#!/usr/bin/env python3
"""Acceptance report for ANY window of a serve — from the engine log, not from test.py.

For sessions the bench tool did not drive (the visual gauntlet, a real client, a long agent run): read the vLLM
container log between two clock times and report, weighted by drafted tokens:
    gen tok/s · engine steps/s · acceptance length · P(draft position i accepted) for i = 1..K  — avg (min–max)
so the per-position picture ("was it the drafter's 3rd/4th guess that failed, or the 1st?") is a number, not a paste.

  ./bench/accept.py --since 13:05 [--until 13:40] [--host 192.168.1.66] [--container mbx-vllm] [--tz-offset -3]
  ./bench/accept.py --since "2026-09-06 13:05:47" --until now

Clock times are LOCAL (yours); the engine logs UTC — --tz-offset is the hours to ADD to a log time to get local
(default: auto from this machine). Reads `docker logs` over ssh when --host is given, locally otherwise. Stdlib only.
"""
import argparse, datetime as dt, re, subprocess, sys, time

L_GEN = re.compile(r"INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[loggers\.py:\d+\] Engine \d+: Avg prompt throughput: ([\d.]+) tokens/s, "
                   r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs")
L_SPEC = re.compile(r"INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[metrics\.py:\d+\] SpecDecoding metrics: Mean acceptance length: ([\d.]+), "
                    r".*?Accepted: (\d+) tokens, Drafted: (\d+) tokens, Per-position acceptance rate: ([\d., ]+?), Avg Draft")


def parse_when(s, today):
    s = s.strip()
    if s == "now":
        return time.time()
    for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%H:%M:%S", "%H:%M"):
        try:
            t = dt.datetime.strptime(s, f)
            if f.startswith("%H"):
                t = t.replace(year=today.year, month=today.month, day=today.day)
            return t.timestamp()
        except ValueError:
            pass
    sys.exit(f"✗ bad time {s!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True); ap.add_argument("--until", default="now")
    ap.add_argument("--host", default="", help="ssh host running the container (default: local docker)")
    ap.add_argument("--container", default="mbx-vllm")
    ap.add_argument("--tz-offset", type=float, default=None, help="hours to add to log (UTC) times → local; default auto")
    ap.add_argument("--min-running", type=int, default=1, help="only count 10 s bins with at least this many running requests")
    a = ap.parse_args()
    today = dt.datetime.now()
    t0, t1 = parse_when(a.since, today), parse_when(a.until, today)
    off = a.tz_offset if a.tz_offset is not None else -time.timezone / 3600 + (1 if time.localtime().tm_isdst else 0)
    cmd = ["docker", "logs", "--since", dt.datetime.fromtimestamp(t0 - 60, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), a.container]
    if a.host:
        cmd = ["ssh", a.host, " ".join(cmd)]
    raw = subprocess.run(cmd, capture_output=True, text=True).stdout + "\n"
    # pair each SpecDecoding line with the Engine line of the same second
    gen = {}
    for m in L_GEN.finditer(raw):
        ts = dt.datetime(today.year, int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5])).timestamp() + off * 3600
        gen[round(ts)] = (float(m[7]), int(m[8]), float(m[6]))
    rows = []
    for m in L_SPEC.finditer(raw):
        ts = dt.datetime(today.year, int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5])).timestamp() + off * 3600
        if not (t0 <= ts <= t1):
            continue
        g = gen.get(round(ts)) or gen.get(round(ts) - 1) or gen.get(round(ts) + 1)
        if not g or g[1] < a.min_running or g[2] > 0:      # skip idle bins and bins with prefill in them
            continue
        drafted, accepted = int(m[8]), int(m[7])
        pos = [float(x) for x in m[9].split(",")]
        if drafted == 0:
            continue
        rows.append({"ts": ts, "gen": g[0], "running": g[1], "acc_len": float(m[6]), "drafted": drafted, "accepted": accepted, "pos": pos})
    if not rows:
        sys.exit("✗ no decode bins in that window (idle, or the container/host is wrong)")
    K = len(rows[0]["pos"]); W = sum(r["drafted"] for r in rows)
    wavg = lambda f: sum(f(r) * r["drafted"] for r in rows) / W
    mm = lambda f: (min(f(r) for r in rows), max(f(r) for r in rows))
    steps = lambda r: r["drafted"] / 10 / K / r["running"]
    span = (rows[-1]["ts"] - rows[0]["ts"] + 10)
    print(f"# acceptance report  {a.host or 'local'}/{a.container}  {time.strftime('%H:%M:%S', time.localtime(rows[0]['ts']))} → "
          f"{time.strftime('%H:%M:%S', time.localtime(rows[-1]['ts']))}  ({span:.0f} s, {len(rows)} decode bins of 10 s, K={K}, weighted by drafted tokens)")
    print(f"  running      avg (min–max): {sum(r['running'] for r in rows)/len(rows):.1f} ({mm(lambda r: r['running'])[0]}–{mm(lambda r: r['running'])[1]})")
    print(f"  gen tok/s    avg (min–max): {sum(r['gen'] for r in rows)/len(rows):.1f} ({mm(lambda r: r['gen'])[0]:.1f}–{mm(lambda r: r['gen'])[1]:.1f})")
    print(f"  steps/s      avg (min–max): {sum(steps(r) for r in rows)/len(rows):.2f} ({steps(min(rows, key=steps)):.2f}–{steps(max(rows, key=steps)):.2f})   (engine steps = drafted ÷ 10 s ÷ K ÷ running)")
    print(f"  acc len      avg (min–max): {wavg(lambda r: r['acc_len']):.2f} ({mm(lambda r: r['acc_len'])[0]:.2f}–{mm(lambda r: r['acc_len'])[1]:.2f})")
    for i in range(K):
        print(f"  P(pos {i+1} ok)  avg (min–max): {wavg(lambda r: r['pos'][i]):.3f} ({mm(lambda r: r['pos'][i])[0]:.3f}–{mm(lambda r: r['pos'][i])[1]:.3f})")
    print(f"  drafted {W:,} · accepted {sum(r['accepted'] for r in rows):,} · overall draft acceptance {100*sum(r['accepted'] for r in rows)/W:.1f}%")


if __name__ == "__main__":
    main()
