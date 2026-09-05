#!/usr/bin/env python3
"""Hold a fixed concurrency on a myllmbox LLM serve for a time window and report avg / min / max.

The LLM speed test: fire exactly C
streams ONCE, together, with a token budget big enough to decode for the whole window — NO re-firing, because a
fresh request mid-window drags a prefill into the decode measurement. Sample the engine's /metrics every
--sample seconds; a sample is STEADY only when all C streams are running AND the prompt-token counter did not
move (pure decode, the bench/README rule). The window runs UNTIL THE FIRST STREAM FINISHES (c is no longer c) —
or, by default, after --warmup + --seconds (30 + 600 s): at high c nobody wants to wait for the first of 32 pasture
answers, and either way the window's end ABORTS every remaining stream (requests are streamed; closing them makes vLLM drop
them) so the next rung starts at once — nobody waits for stragglers. --seconds 0 = no cap. The watched period is
reported. avg/min/max over the steady samples of

    gen tok/s      (Δ vllm:generation_tokens_total / Δt          — aggregate, all streams)
    per-stream     (gen tok/s ÷ running requests)
    steps/s        (Δ vllm:spec_decode_num_drafts_total / Δt / running — ENGINE steps: the drafts counter is
                    per sequence per step, so at c=8 it runs 8× the engine rate; we divide it back out)
    acc len        (1 + Δaccepted / Δdrafts                       — content/checkpoint quality)
    running        (vllm:num_requests_running gauge — proves the rung was actually held)
    prefill-free   (Δ vllm:prompt_tokens_total == 0 in the sample — otherwise the sample is dropped)

Thinking is a switch: --thinking on|off → chat_template_kwargs.enable_thinking (the model's own toggle;
completion_tokens counts reasoning + answer either way). Prompts: the built-in code prompt (default),
the two gauntlet texts (pasture/fish), or any file. A nonce per request defeats prefix caching.

  ./bench/test.py --c 8 --thinking off --prompt pasture                        # one rung: 30 s warmup + 600 s measured
  ./bench/test.py --c 8,12,16,24,32 --thinking off --prompt pasture --seconds 240   # ladder, 4 min measured per rung

The served model is DETECTED from GET /v1/models (--model only to override). Stdlib only. Run it ON the
serving box (the management LAN blocks raw ports) or via the proxy with --token.
Every run is saved as bench/results/<model>/<timestamp>-c<N>.json (gitignored): "params" first, then summary, samples, requests.
"""
import argparse, json, os, re, sys, threading, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_PROMPT = ("Write a complete, production-quality Python implementation of an LRU cache with O(1) get and put, "
               "full type hints, docstrings, thread-safety, and a pytest test suite covering eviction, update, and "
               "capacity edge cases. Then explain the design and its complexity in detail.")

M_GEN, M_DRAFTS, M_DTOK, M_ACC = ("vllm:generation_tokens_total", "vllm:spec_decode_num_drafts_total",
                                  "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total")
M_RUN, M_WAIT, M_PROMPT = "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:prompt_tokens_total"
M_KV = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")   # name changed across vLLM versions


def _headers(token):
    return {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}


def _metrics(url, token):
    try:
        req = urllib.request.Request(url + "/metrics", headers=_headers(token))
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read().decode()
    except Exception:
        return ""


def _sum(text, name):
    tot, found = 0.0, False
    for m in re.finditer(rf'^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$', text, re.M):
        tot += float(m.group(1)); found = True
    return tot if found else None


def _first(text, names):
    for n in names:
        v = _sum(text, n)
        if v is not None:
            return v
    return None


def detect_model(url, token):
    """The served model name, from GET /v1/models (first entry). No guessing, no flag needed."""
    try:
        req = urllib.request.Request(url + "/v1/models", headers=_headers(token))
        with urllib.request.urlopen(req, timeout=15) as r:
            ids = [m.get("id") for m in (json.load(r).get("data") or []) if m.get("id")]
    except Exception as e:
        sys.exit(f"✗ cannot read {url}/v1/models ({e}) — is the serve up? (or pass --model)")
    if not ids:
        sys.exit(f"✗ {url}/v1/models lists no models — pass --model")
    if len(ids) > 1:
        print(f"  ⚠ several models served {ids} — using the first (override with --model)", flush=True)
    return ids[0]


def load_prompt(spec):
    if spec == "code":
        return CODE_PROMPT
    p = HERE / f"{spec}-text.txt" if spec in ("pasture", "fish") else Path(spec)
    return p.read_text()


class Worker(threading.Thread):
    """One stream: ONE streamed request, fired at start, decoding until it finishes — or until the stop event
    (the --seconds cap) makes us close the connection, which makes vLLM abort the request server-side."""
    def __init__(self, i, a, prompt, stop, log):
        super().__init__(daemon=True)
        self.i, self.a, self.prompt, self.stop, self.log = i, a, prompt, stop, log

    def run(self):
        nonce = f"{self.a.tag}-s{self.i}-{int(time.time()*1000)}"
        body = {"model": self.a.model, "temperature": self.a.temperature, "top_p": self.a.top_p,
                "max_tokens": self.a.max_tokens, "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": self.a.thinking == "on"},
                "messages": [{"role": "user", "content": f"{self.prompt}\n\n(request id {nonce}, answer fully)"}]}
        t0 = time.time(); rec = {"stream": self.i, "t0": t0, "ok": True, "completion_tokens": 0, "prompt_tokens": 0,
                                 "finish": None, "reasoning_chars": 0, "answer_chars": 0, "chunks": 0}
        try:
            req = urllib.request.Request(self.a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         method="POST", headers=_headers(self.a.token))
            with urllib.request.urlopen(req, timeout=self.a.timeout) as r:
                for raw in r:                       # SSE lines; the socket read is what the cap interrupts (we close)
                    if self.stop.is_set():
                        rec["finish"] = "aborted"; break
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    try:
                        ev = json.loads(line[5:])
                    except Exception:
                        continue
                    if ev.get("usage"):
                        rec["completion_tokens"] = ev["usage"].get("completion_tokens", rec["completion_tokens"])
                        rec["prompt_tokens"] = ev["usage"].get("prompt_tokens", rec["prompt_tokens"])
                    for ch in ev.get("choices") or []:
                        d = ch.get("delta") or {}
                        rec["chunks"] += 1
                        rec["reasoning_chars"] += len(d.get("reasoning_content") or "")
                        rec["answer_chars"] += len(d.get("content") or "")
                        if ch.get("finish_reason"):
                            rec["finish"] = ch["finish_reason"]
        except Exception as e:
            if self.stop.is_set():
                rec["finish"] = "aborted"
            else:
                rec.update(ok=False, error=str(e)[:200])
        rec["t1"] = time.time()
        self.log.append(rec)


def sample_loop(a, t_start, deadline, samples):
    prev = None
    while True:
        now = time.time()
        if now >= deadline:
            print(f"· cap reached at {round(now - t_start)}s (warmup {a.warmup} + {a.seconds} measured) — window ends here; aborting the streams", flush=True)
            break
        m = _metrics(a.url, a.token)
        cur = {"t": now, "gen": _sum(m, M_GEN), "drafts": _sum(m, M_DRAFTS), "dtok": _sum(m, M_DTOK),
               "acc": _sum(m, M_ACC), "running": _sum(m, M_RUN), "waiting": _sum(m, M_WAIT), "kv": _first(m, M_KV),
               "prompt": _sum(m, M_PROMPT)}
        if prev and cur["gen"] is not None and prev["gen"] is not None:
            dt = cur["t"] - prev["t"]
            s = {"t": round(now - t_start), "dt": round(dt, 1), "running": cur["running"], "waiting": cur["waiting"],
                 "kv_pct": (cur["kv"] * 100 if cur["kv"] is not None and cur["kv"] <= 1 else cur["kv"]),
                 "gen_tps": (cur["gen"] - prev["gen"]) / dt,
                 "prompt_tok": (cur["prompt"] - prev["prompt"]) if cur["prompt"] is not None and prev["prompt"] is not None else None}
            if cur["drafts"] is not None and prev["drafts"] is not None and cur["drafts"] > prev["drafts"]:
                dd = cur["drafts"] - prev["drafts"]
                s["seq_steps_ps"] = dd / dt                                   # per-sequence drafts/s (= c × engine)
                s["steps_ps"] = dd / dt / (cur["running"] or 1)               # ENGINE steps/s
                s["acc_len"] = 1 + (cur["acc"] - prev["acc"]) / dd if cur["acc"] is not None else None
                s["k"] = (cur["dtok"] - prev["dtok"]) / dd if cur["dtok"] is not None else None
            elif cur["dtok"] is not None and prev["dtok"] is not None and a.k:
                s["steps_ps"] = (cur["dtok"] - prev["dtok"]) / a.k / dt / (cur["running"] or 1)   # fallback: drafted ÷ K ÷ running
                s["acc_len"] = None
            else:
                s["steps_ps"] = s["acc_len"] = None
            s["per_stream"] = s["gen_tps"] / s["running"] if s["running"] else None
            s["steady"] = bool(s["t"] >= a.warmup and (s["running"] or 0) >= a.c and not s["prompt_tok"])
            samples.append(s)
            print(f"  t={s['t']:4d}s  run={int(s['running'] or 0):2d} wait={int(s['waiting'] or 0):2d}  "
                  f"gen {s['gen_tps']:6.1f} tok/s  per-stream {(s['per_stream'] or 0):5.1f}  "
                  f"steps {(s['steps_ps'] or 0):5.1f}/s  acc {(s['acc_len'] or 0):4.2f}  kv {(s['kv_pct'] or 0):4.1f}%"
                  f"{'  prefill ' + str(int(s['prompt_tok'])) + ' tok' if s['prompt_tok'] else ''}{'' if s['steady'] else '  (not steady)'}",
                  flush=True)
            if s["t"] >= a.warmup and (s["running"] or 0) < a.c:
                print(f"· a stream finished (running {int(s['running'] or 0)} < c={a.c}) — window ends here at {s['t']}s", flush=True)
                break
        prev = cur
        time.sleep(max(0.0, a.sample - (time.time() - now)))


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"avg": sum(xs) / len(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def fmt(st, nd=1):
    return "—" if not st else f"{st['avg']:.{nd}f} ({st['min']:.{nd}f}–{st['max']:.{nd}f})"


def run_hold(a, prompt, c):
    a.c = c   # the rung being held (Worker/sample_loop read a.c)
    m0 = _metrics(a.url, a.token)
    if not m0:
        sys.exit(f"✗ no /metrics at {a.url} — is the serve up (and reachable from here)?")
    have = {n: _sum(m0, n) is not None for n in (M_GEN, M_DRAFTS, M_DTOK, M_ACC, M_RUN)}
    missing = [n for n, ok in have.items() if not ok]
    print(f"# test  {a.url}  model={a.model}  c={c}  window={'warmup ' + str(a.warmup) + 's + ' + str(a.seconds) + 's measured (or first stream to finish)' if a.seconds > 0 else 'until the first stream finishes (warmup ' + str(a.warmup) + 's excluded)'}  "
          f"thinking={a.thinking}  max_tokens={a.max_tokens}  prompt={a.prompt}", flush=True)
    if missing:
        print(f"  ⚠ metrics missing: {', '.join(missing)}" + ("  (steps/s via drafted÷K needs --k)" if M_DRAFTS in missing and not a.k else ""), flush=True)

    t_start = time.time(); deadline = t_start + (a.warmup + a.seconds if a.seconds > 0 else 10**9)   # cap = warmup + measured
    log, samples = [], []
    stop = threading.Event()
    workers = [Worker(i, a, prompt, stop, log) for i in range(c)]
    [w.start() for w in workers]
    sample_loop(a, t_start, deadline, samples)
    # window over (cap hit OR a stream finished) → ABORT the rest at once: close every stream so vLLM drops the
    # requests, then wait until the engine reports 0 running before the next rung. Nobody waits for stragglers.
    print("· window over — aborting the remaining streams", flush=True)
    stop.set()
    for w in workers:
        w.join(timeout=30)
    for _ in range(60):
        m = _metrics(a.url, a.token)
        if not (_sum(m, M_RUN) or 0):
            break
        time.sleep(1)

    steady = [s for s in samples if s["steady"]]
    held = [s for s in samples if s["t"] >= a.warmup]
    ok = [r for r in log if r["ok"]]
    truncated = sum(1 for r in ok if r["finish"] == "length")
    aborted = sum(1 for r in ok if r["finish"] == "aborted")
    summary = {
        "gen_tps": stats([s["gen_tps"] for s in steady]),
        "per_stream": stats([s["per_stream"] for s in steady]),
        "steps_ps": stats([s["steps_ps"] for s in steady]),
        "acc_len": stats([s["acc_len"] for s in steady]),
        "running": stats([s["running"] for s in held]),
        "kv_pct": stats([s["kv_pct"] for s in held]),
        "samples_total": len(samples), "samples_steady": len(steady),
        "requests_completed": len(ok), "requests_failed": len(log) - len(ok), "requests_truncated": truncated,
        "requests_aborted_by_cap": aborted,
        "completion_tokens_total": sum(r["completion_tokens"] for r in ok),
        "avg_completion_tokens": (sum(r["completion_tokens"] for r in ok) / len(ok)) if ok else None,
        "avg_request_s": (sum(r["t1"] - r["t0"] for r in ok) / len(ok)) if ok else None,
    }

    t_end = time.time()
    clock = lambda t: time.strftime("%H:%M:%S", time.localtime(t))
    st_ts = [t_start + s["t"] for s in steady]
    period = {"start": clock(t_start), "end": clock(t_end), "seconds": round(t_end - t_start),
              "steady_from": clock(min(st_ts)) if st_ts else None, "steady_to": clock(max(st_ts)) if st_ts else None,
              "steady_seconds": round(max(st_ts) - min(st_ts) + a.sample) if st_ts else 0}
    summary["period"] = period
    print(f"\n# watched {period['start']} → {period['end']} ({period['seconds']} s); steady part "
          f"{period['steady_from']} → {period['steady_to']} ({period['steady_seconds']} s, {len(steady)} samples)", flush=True)
    print(f"# summary (steady = after warmup, all {c} running, zero prefill in the sample):", flush=True)
    print(f"  samples {summary['samples_steady']}/{summary['samples_total']} steady · requests {len(ok)} ok, "
          f"{summary['requests_failed']} failed, {truncated} hit max_tokens, {aborted} aborted by cap · avg {summary['avg_completion_tokens'] or 0:.0f} tok / "
          f"{summary['avg_request_s'] or 0:.0f} s per request", flush=True)
    print(f"  gen tok/s   avg (min–max): {fmt(summary['gen_tps'])}", flush=True)
    print(f"  per-stream  avg (min–max): {fmt(summary['per_stream'])}", flush=True)
    print(f"  steps/s     avg (min–max): {fmt(summary['steps_ps'])}   (engine steps; ms/step avg {1000/summary['steps_ps']['avg']:.0f})" if summary['steps_ps'] else "  steps/s     —", flush=True)
    print(f"  acc len     avg (min–max): {fmt(summary['acc_len'], 2)}", flush=True)
    print(f"  running     avg (min–max): {fmt(summary['running'])}   kv% {fmt(summary['kv_pct'])}", flush=True)
    if summary["samples_steady"] < max(3, len(held) // 2):
        print("  ⚠ few steady samples — rung not held (max-num-seqs below c? streams finished early → raise --max-tokens; prefill spilling past --warmup → raise it)", flush=True)
    if truncated:
        print(f"  ⚠ {truncated} request(s) hit max_tokens — with thinking on, raise --max-tokens or the numbers are reasoning-only", flush=True)

    date = time.strftime("%Y-%m-%d %H:%M")
    print("\n# reports.md row:", flush=True)
    print(f"| {date} | | {c} | {a.prompt} thinking {a.thinking} (steady {period['steady_seconds']}s of {period['seconds']}s watched) | {fmt(summary['gen_tps'], 0)} | "
          f"{fmt(summary['steps_ps'])} | {fmt(summary['acc_len'], 2)} | | per-stream {fmt(summary['per_stream'], 0)}; test.py |", flush=True)

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", a.model).strip("-")            # Qwen/Qwen3.8-Flash-Next → Qwen-Qwen3.8-Flash-Next
    out = Path(a.json) if a.json else HERE / "results" / slug / f"{time.strftime('%Y%m%d-%H%M%S')}-c{c}.json"
    if a.json and "," in str(a.levels_raw):   # a ladder with one --json name → one file per rung
        out = out.with_name(f"{out.stem}-c{c}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    params = {"date": date, "url": a.url, "model": a.model, "c": c, "watched": period, "cap_seconds": a.seconds, "warmup": a.warmup,
              "sample": a.sample, "thinking": a.thinking, "prompt": a.prompt, "max_tokens": a.max_tokens,
              "temperature": a.temperature, "top_p": a.top_p, "tag": a.tag}
    out.write_text(json.dumps({"params": params, "summary": summary, "samples": samples, "requests": log}, indent=1))
    print(f"· saved {out}", flush=True)

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="", help="served model name (default: detected from /v1/models)")
    ap.add_argument("--token", default="")
    ap.add_argument("--c", required=True, help="concurrency to hold, or a comma list run one after another (e.g. 8,12,16)")
    ap.add_argument("--seconds", type=int, default=600, help="measured window AFTER warmup; then the streams are ABORTED and the next rung starts (default 600; 0 = no cap, run until the first stream finishes)")
    ap.add_argument("--warmup", type=int, default=30, help="seconds excluded at the start — covers the c prefills (default 30)")
    ap.add_argument("--sample", type=int, default=10, help="metrics sampling period (default 10 s, = the engine's own window)")
    ap.add_argument("--max-tokens", type=int, default=32768, help="per-stream budget — must outlast the window (~tok/s × seconds)")
    ap.add_argument("--thinking", choices=["on", "off"], default="on")
    ap.add_argument("--prompt", default="code", help="code | pasture | fish | <path to a text file>")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=0, help="num_speculative_tokens — only needed if the engine lacks the drafts counter")
    ap.add_argument("--timeout", type=int, default=1800, help="per-request timeout (s)")
    ap.add_argument("--tag", default="test")
    ap.add_argument("--json", default="", help="output path (default bench/results/<model>/<timestamp>-c<N>.json)")
    a = ap.parse_args()

    prompt = load_prompt(a.prompt)
    a.levels_raw = str(a.c)
    levels = [int(x) for x in a.levels_raw.split(",") if x.strip()]
    if not a.model:
        a.model = detect_model(a.url, a.token)
        print(f"· model detected: {a.model}", flush=True)
    rows = []
    for i, c in enumerate(levels):
        if i:
            print(f"\n· rung c={levels[i-1]} fully drained — starting c={c}\n", flush=True)
        rows.append((c, run_hold(a, prompt, c)))
    if len(rows) > 1:
        print("\n# ladder (avg (min–max) over steady samples):", flush=True)
        print("| c | gen tok/s | per-stream | steps/s | acc len | steady s / watched s |", flush=True)
        print("|---|---|---|---|---|---|", flush=True)
        for c, sm in rows:
            pr = sm["period"]
            print(f"| {c} | {fmt(sm['gen_tps'], 0)} | {fmt(sm['per_stream'], 0)} | {fmt(sm['steps_ps'])} | "
                  f"{fmt(sm['acc_len'], 2)} | {pr['steady_seconds']} / {pr['seconds']} |", flush=True)


if __name__ == "__main__":
    main()
