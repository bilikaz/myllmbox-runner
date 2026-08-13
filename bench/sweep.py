#!/usr/bin/env python3
"""Concurrency sweep for a myllmbox LLM serve — the honest A/B harness.

Fires ONE fixed prompt (+ a per-request nonce so prefix-caching can't inflate throughput) at increasing
concurrency, measures END-TO-END aggregate output tok/s CLIENT-SIDE (sum completion_tokens / wall — not the
engine's rolling avg), plus spec-decode acceptance from /metrics deltas. Prints ready-to-paste reports.md rows.

Same prompt + same model + same spec on both serves ⇒ same acceptance ⇒ isolates per-step kernel speed.

  ./bench/sweep.py --url http://127.0.0.1:8000 --model deepseek-ai/DeepSeek-V4-Flash \
                   [--token T] [--c 1,2,4,8,16] [--max-tokens 512] [--rounds 2]

Stdlib only (urllib + threads) so it runs on any box with python3, no venv.
"""
import argparse, json, re, threading, time, urllib.request

PROMPT = ("Write a complete, production-quality Python implementation of an LRU cache with O(1) get and put, "
          "full type hints, docstrings, thread-safety, and a pytest test suite covering eviction, update, and "
          "capacity edge cases. Then explain the design and its complexity in detail.")  # draftable/structured


def _headers(token):
    return {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}


def _post(url, token, body, timeout=900):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 method="POST", headers=_headers(token))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


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


def _one(url, token, model, max_tokens, nonce, out):
    body = {"model": model, "temperature": 0.6, "top_p": 0.95, "max_tokens": max_tokens, "stream": False,
            "messages": [{"role": "user", "content": f"{PROMPT}\n\n(request id {nonce}, answer fully)"}]}
    t0 = time.time()
    try:
        r = _post(url, token, body)
        ct = (r.get("usage") or {}).get("completion_tokens", 0)
        out.append(ct)
    except Exception as e:
        out.append(0); print(f"    req {nonce} error: {e}", flush=True)


def sweep(url, token, model, c, max_tokens, rounds, tag):
    results = []
    for rnd in range(rounds):
        m0 = _metrics(url, token)
        a0, d0 = _sum(m0, "vllm:spec_decode_num_accepted_tokens_total"), _sum(m0, "vllm:spec_decode_num_draft_tokens_total")
        out, ths = [], []
        base = int(time.time() * 1000)
        for i in range(c):
            ths.append(threading.Thread(target=_one, args=(url, token, model, max_tokens, f"{tag}-{rnd}-{i}-{base}", out)))
        s = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; wall = time.time() - s
        toks = sum(out)
        tps = toks / wall if wall > 0 else 0.0
        m1 = _metrics(url, token)
        a1, d1 = _sum(m1, "vllm:spec_decode_num_accepted_tokens_total"), _sum(m1, "vllm:spec_decode_num_draft_tokens_total")
        acc = (100.0 * (a1 - a0) / (d1 - d0)) if (a0 is not None and d1 and (d1 - d0) > 0) else None
        results.append((tps, toks, wall, acc))
        accs = f"{acc:.0f}%" if acc is not None else "n/a"
        print(f"  c={c:<3} round {rnd+1}/{rounds}: {tps:6.1f} tok/s  ({toks} tok / {wall:.1f}s)  draft-accept {accs}", flush=True)
    tpss = sorted(r[0] for r in results)
    med = tpss[len(tpss) // 2]
    accs = [r[3] for r in results if r[3] is not None]
    acc_med = sorted(accs)[len(accs) // 2] if accs else None
    return med, acc_med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--token", default="")
    ap.add_argument("--c", default="1,2,4,8,16")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    levels = [int(x) for x in a.c.split(",") if x.strip()]

    print(f"# sweep {a.url}  model={a.model}  max_tokens={a.max_tokens}  rounds={a.rounds}", flush=True)
    print("· warmup (1 request, absorbs first-shape JIT)…", flush=True)
    sweep(a.url, a.token, a.model, 1, a.max_tokens, 1, "warm")

    rows = []
    for c in levels:
        med, acc = sweep(a.url, a.token, a.model, c, a.max_tokens, a.rounds, f"c{c}")
        rows.append((c, med, acc))

    print("\n# paste into reports.md (aggregate tok/s = median of rounds; per-req = agg/c):", flush=True)
    for c, med, acc in rows:
        accs = f"{acc:.0f}%" if acc is not None else "—"
        print(f"| DATE | {c} | code (fixed) | {med:.0f} | {med/c:.0f} | — | {accs} | — | sweep.py |", flush=True)


if __name__ == "__main__":
    main()
