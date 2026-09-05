#!/usr/bin/env python3
"""Summarise bench/test.py results: pool every run of a model into one ladder table.

Reads bench/results/<model>/*.json (what test.py writes), keeps the runs inside --start/--end, groups them by
rung (c) and pools the STEADY samples of all runs in a group:

    runs · steady samples · gen tok/s · per-stream · engine steps/s (ms/step) · acc len — each avg (min–max)

Repeated rungs (test.py --c 1,1,1,1) therefore become one row with 4 runs behind it.
Thinking OFF runs are the **code** band, thinking ON the **thinking** band. --thinking off|on prints that band's
table with every metric; --thinking both (default) prints ONE combined table, a row per rung:
  c · MAX tok/s · AVERAGE tok/s · MIN tok/s · AVERAGE per-stream · code avg (min–max) + /stream · thinking avg
  (min–max) + /stream · steps/s avg (min–max) · acceptance avg (min–max)
where every AVERAGE = (code band avg + thinking band avg) / 2 (arithmetic — each band weighs the same whatever its
sample count), per-stream = AVERAGE / c, and MIN/MAX = the extremes reported by either band.
The report-ready view, no hand compiling. Markdown; --json dumps the numbers.

  ./bench/summary.py                                       # one model folder present → it's picked
  ./bench/summary.py --model Qwen/Qwen3.8-Flash-Next       # served name or folder slug
  ./bench/summary.py --start 2026-09-05 --end 2026-09-05 --prompt pasture              # code + thinking + average
  ./bench/summary.py --thinking off --prompt pasture                                  # the code band only
  ./bench/summary.py --start "2026-09-05 13:00"            # times allowed; --end defaults to now

Stdlib only.
"""
import argparse, json, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def slug(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def parse_when(s, end=False):
    """'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM[:SS]' → epoch. A bare date as --end means the END of that day."""
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            t = time.mktime(time.strptime(s, fmt))
            return t + 86399 if (fmt == "%Y-%m-%d" and end) else t
        except ValueError:
            pass
    sys.exit(f"✗ bad date '{s}' (use YYYY-MM-DD or 'YYYY-MM-DD HH:MM')")


def run_time(path, params):
    """When the run happened: the filename timestamp (YYYYmmdd-HHMMSS) — falls back to params.date."""
    m = re.match(r"(\d{8}-\d{6})", path.stem)
    if m:
        return time.mktime(time.strptime(m.group(1), "%Y%m%d-%H%M%S"))
    d = params.get("date")
    return time.mktime(time.strptime(d, "%Y-%m-%d %H:%M")) if d else path.stat().st_mtime


def engine_steps(sample):
    """Engine steps/s. Early test.py wrote per-sequence drafts/s into steps_ps (no seq_steps_ps key) → ÷ running."""
    v = sample.get("steps_ps")
    if v is None:
        return None
    if "seq_steps_ps" in sample:
        return v
    return v / (sample.get("running") or 1)


def st(xs):
    xs = [x for x in xs if x is not None]
    return {"avg": sum(xs) / len(xs), "min": min(xs), "max": max(xs), "n": len(xs)} if xs else None


def fmt(s, nd=1):
    return "—" if not s else f"{s['avg']:.{nd}f} ({s['min']:.{nd}f}–{s['max']:.{nd}f})"


def pick_model(name):
    if not RESULTS.is_dir():
        sys.exit(f"✗ no results dir {RESULTS}")
    folders = sorted(p for p in RESULTS.iterdir() if p.is_dir() and any(p.glob("*.json")))
    if name:
        want = slug(name)
        hit = [p for p in folders if p.name == want or p.name == name]
        if not hit:
            sys.exit(f"✗ no results for '{name}' — have: {', '.join(p.name for p in folders) or 'none'}")
        return hit[0]
    if len(folders) == 1:
        return folders[0]
    sys.exit("✗ several models have results — pick one with --model: " + ", ".join(p.name for p in folders))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="", help="served model name or results folder slug (auto if only one)")
    ap.add_argument("--start", default="", help="include runs from this date/time (YYYY-MM-DD [HH:MM])")
    ap.add_argument("--end", default="", help="…up to this date/time (default now; a bare date = whole day)")
    ap.add_argument("--thinking", choices=["on", "off", "both"], default="both",
                    help="off = the code band, on = the thinking band, both (default) = code + thinking + average tables")
    ap.add_argument("--prompt", default="", help="only runs with this prompt (code | pasture | fish | path)")
    ap.add_argument("--json", default="", help="also write the summary as JSON to this path")
    a = ap.parse_args()

    folder = pick_model(a.model)
    t0, t1 = parse_when(a.start), parse_when(a.end, end=True) or time.time()

    runs, skipped = [], 0
    for p in sorted(folder.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            skipped += 1; continue
        prm, smp = d.get("params") or {}, d.get("samples") or []
        when = run_time(p, prm)
        if (t0 and when < t0) or when > t1:
            continue
        if a.thinking != "both" and prm.get("thinking") != a.thinking:
            continue
        if a.prompt and prm.get("prompt") != a.prompt:
            continue
        steady = [s for s in smp if s.get("steady")]
        runs.append({"file": p.name, "when": when, "c": prm.get("c"), "thinking": prm.get("thinking"),
                     "prompt": prm.get("prompt"), "steady": steady, "summary": d.get("summary") or {}})

    if not runs:
        sys.exit(f"✗ no runs in {folder.name} match (window {a.start or 'any'} → {a.end or 'now'}"
                 f"{', thinking ' + a.thinking if a.thinking else ''}{', prompt ' + a.prompt if a.prompt else ''})")

    split_prompt = len({r["prompt"] for r in runs}) > 1

    def band_rows(band_runs):
        """rows per (c[, prompt]) for one band, pooling the steady samples of every run in the group."""
        groups = {}
        for r in band_runs:
            groups.setdefault((r["c"], r["prompt"] if split_prompt else ""), []).append(r)
        rows = []
        for k in sorted(groups):
            rs = groups[k]; smp = [s for r in rs for s in r["steady"]]
            rows.append({
                "c": k[0], "prompt": k[1] or rs[0]["prompt"], "runs": len(rs), "samples": len(smp),
                "gen_tps": st([s.get("gen_tps") for s in smp]),
                "per_stream": st([s.get("per_stream") for s in smp]),
                "steps_ps": st([engine_steps(s) for s in smp]),
                "acc_len": st([s.get("acc_len") for s in smp]),
                "kv_pct": st([s.get("kv_pct") for s in smp]),
                "finished": sum((r["summary"].get("requests_completed") or 0) - (r["summary"].get("requests_aborted_by_cap") or 0) for r in rs),
                "aborted": sum(r["summary"].get("requests_aborted_by_cap") or 0 for r in rs),
            })
        return rows

    def merge(x, y):
        """average of two band stats: avg = mean of the avgs, min/max = the extremes of both."""
        if not x or not y:
            return None
        return {"avg": (x["avg"] + y["avg"]) / 2, "min": min(x["min"], y["min"]), "max": max(x["max"], y["max"]), "n": x["n"] + y["n"]}

    def average_rows(code, think):
        ci = {(r["c"], r["prompt"]): r for r in code}; ti = {(r["c"], r["prompt"]): r for r in think}
        rows = []
        for k in sorted(set(ci) | set(ti)):
            c, t = ci.get(k), ti.get(k)
            if not (c and t):
                rows.append({"c": k[0], "prompt": k[1], "runs": (c or t)["runs"], "samples": (c or t)["samples"],
                             "gen_tps": None, "per_stream": None, "steps_ps": None, "acc_len": None, "kv_pct": None,
                             "finished": (c or t)["finished"], "aborted": (c or t)["aborted"], "note": "code only" if c else "thinking only"})
                continue
            rows.append({"c": k[0], "prompt": k[1], "runs": c["runs"] + t["runs"], "samples": c["samples"] + t["samples"],
                         "gen_tps": merge(c["gen_tps"], t["gen_tps"]), "per_stream": merge(c["per_stream"], t["per_stream"]),
                         "steps_ps": merge(c["steps_ps"], t["steps_ps"]), "acc_len": merge(c["acc_len"], t["acc_len"]),
                         "kv_pct": merge(c["kv_pct"], t["kv_pct"]), "finished": c["finished"] + t["finished"], "aborted": c["aborted"] + t["aborted"]})
        return rows

    def table(title, rows):
        print(f"\n## {title}")
        ex = "| prompt " if split_prompt else ""
        print(f"| c {ex}| runs | samples | gen tok/s avg (min–max) | per-stream | engine steps/s | ms/step | acc len | kv % | finished/aborted |")
        print(f"|---{'|---' if split_prompt else ''}|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            e = f"| {r['prompt']} " if split_prompt else ""
            if r.get("note"):
                print(f"| {r['c']} {e}| {r['runs']} | {r['samples']} | — ({r['note']}) | — | — | — | — | — | {r['finished']}/{r['aborted']} |")
                continue
            ms = f"{1000 / r['steps_ps']['avg']:.0f}" if r["steps_ps"] else "—"
            print(f"| {r['c']} {e}| {r['runs']} | {r['samples']} | {fmt(r['gen_tps'])} | {fmt(r['per_stream'])} | "
                  f"{fmt(r['steps_ps'])} | {ms} | {fmt(r['acc_len'], 2)} | {fmt(r['kv_pct'])} | {r['finished']}/{r['aborted']} |")

    lo, hi = min(r["when"] for r in runs), max(r["when"] for r in runs)
    clock = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
    code = band_rows([r for r in runs if r["thinking"] == "off"])
    think = band_rows([r for r in runs if r["thinking"] == "on"])
    print(f"# {folder.name} — {len(runs)} runs ({sum(r['runs'] for r in code)} code / {sum(r['runs'] for r in think)} thinking), "
          f"{sum(len(r['steady']) for r in runs)} steady samples, {clock(lo)} → {clock(hi)}"
          f"{'  prompt=' + a.prompt if a.prompt else ''}{'  (' + str(skipped) + ' unreadable files skipped)' if skipped else ''}")
    tables = {}
    if a.thinking == "off" and code:
        tables["code"] = code; table("code  (thinking off)", code)
    elif a.thinking == "on" and think:
        tables["thinking"] = think; table("thinking  (thinking on)", think)
    elif a.thinking == "both":
        # ONE combined table: code | thinking | average per rung (no runs/samples/kv — those live in the band tables)
        tables["code"], tables["thinking"] = code, think
        tables["average"] = average_rows(code, think) if (code and think) else []
        ci = {(r["c"], r["prompt"]): r for r in code}; ti = {(r["c"], r["prompt"]): r for r in think}
        av = {(r["c"], r["prompt"]): r for r in tables["average"]}
        ex = "| prompt " if split_prompt else ""
        print(f"\n| c {ex}| MAX tok/s | AVERAGE tok/s | MIN tok/s | AVERAGE /stream | "
              f"code tok/s avg (min–max) | code /stream | thinking tok/s avg (min–max) | thinking /stream | "
              f"steps/s avg (min–max) | acc avg (min–max) |")
        print(f"|---{'|---' if split_prompt else ''}|---|---|---|---|---|---|---|---|---|---|")
        f1 = lambda st, nd=1: "—" if not st else f"{st['avg']:.{nd}f}"
        for k in sorted(set(ci) | set(ti)):
            c, t = ci.get(k), ti.get(k)
            e = f"| {k[1]} " if split_prompt else ""
            both = c and t
            g = merge(c["gen_tps"], t["gen_tps"]) if both else None          # avg of band avgs; min/max = extremes of both
            sp = merge(c["steps_ps"], t["steps_ps"]) if both else None
            ac = merge(c["acc_len"], t["acc_len"]) if both else None
            print(f"| {k[0]} {e}| {g['max']:.0f} | **{g['avg']:.0f}** | {g['min']:.0f} | **{g['avg'] / k[0]:.1f}** | " if g else
                  f"| {k[0]} {e}| — | — | — | — | ", end="")
            print(f"{fmt(c['gen_tps']) if c else '—'} | {f1(c['per_stream']) if c else '—'} | "
                  f"{fmt(t['gen_tps']) if t else '—'} | {f1(t['per_stream']) if t else '—'} | "
                  f"{fmt(sp) if sp else (fmt(c['steps_ps']) if c else fmt(t['steps_ps']))} | "
                  f"{fmt(ac, 2) if ac else (fmt(c['acc_len'], 2) if c else fmt(t['acc_len'], 2))} |")
        if not (code and think):
            print(f"\n(only the {'code' if code else 'thinking'} band has runs — AVERAGE column empty)")
    rows = tables

    if a.json:
        Path(a.json).write_text(json.dumps({"model": folder.name, "runs": len(runs), "window": [clock(lo), clock(hi)],
                                            "filters": {"thinking": a.thinking, "prompt": a.prompt},
                                            "tables": rows}, indent=1))
        print(f"· saved {a.json}")


if __name__ == "__main__":
    main()
