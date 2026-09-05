#!/usr/bin/env python3
"""Score fixed texts with the served model: average log-probability per token (and perplexity).

The instrument the render gauntlet lacks: a checkpoint/table/quant change that the eye can't see shows up here in
the third decimal. Same texts, same server flags, different weights → compare the averages. Uses the completions
API with echo + logprobs (prompt_logprobs), no generation involved, so it does not disturb a serving box.

  ./bench/logprob.py                                # both gauntlet texts (pasture + fish), model auto-detected
  ./bench/logprob.py --text README.md --text notes.txt [--url U] [--token T] [--model M] [--tag ple4]

Prints per text: tokens · avg logprob · ppl; then the pooled average. Saves bench/results/<model>/logprob-<ts>.json
("params" first). Stdlib only.
"""
import argparse, json, math, re, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _headers(token):
    return {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}


def detect_model(url, token):
    req = urllib.request.Request(url + "/v1/models", headers=_headers(token))
    with urllib.request.urlopen(req, timeout=15) as r:
        ids = [m["id"] for m in json.load(r).get("data", []) if m.get("id")]
    if not ids:
        sys.exit("✗ /v1/models lists nothing — pass --model")
    return ids[0]


def score(url, token, model, text):
    body = {"model": model, "prompt": text, "max_tokens": 1, "echo": True, "logprobs": 0, "temperature": 0}
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), method="POST", headers=_headers(token))
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.load(r)
    lp = (res["choices"][0].get("logprobs") or {}).get("token_logprobs") or []
    vals = [v for v in lp[:-1] if v is not None]          # drop the generated token; first prompt token has no logprob
    if not vals:
        sys.exit("✗ server returned no prompt logprobs — needs a vLLM completions endpoint with echo+logprobs")
    return len(vals), sum(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default="")
    ap.add_argument("--model", default="", help="served name (default: detected from /v1/models)")
    ap.add_argument("--text", action="append", default=[], help="text file to score (repeatable; default: the two gauntlet prompts)")
    ap.add_argument("--tag", default="", help="free label for this checkpoint/config (e.g. int3, ple4, bf16-table)")
    a = ap.parse_args()
    texts = a.text or [str(HERE / "pasture-text.txt"), str(HERE / "fish-text.txt")]
    if not a.model:
        a.model = detect_model(a.url, a.token)
    print(f"# logprob  {a.url}  model={a.model}{'  tag=' + a.tag if a.tag else ''}", flush=True)
    rows, tot_n, tot_lp = [], 0, 0.0
    for t in texts:
        txt = Path(t).read_text()
        n, lp = score(a.url, a.token, a.model, txt)
        rows.append({"text": t, "tokens": n, "avg_logprob": lp / n, "ppl": math.exp(-lp / n)})
        tot_n += n; tot_lp += lp
        print(f"  {Path(t).name:24s} tokens {n:6d}  avg logprob {lp/n:8.4f}  ppl {math.exp(-lp/n):8.3f}", flush=True)
    avg = tot_lp / tot_n
    print(f"  {'POOLED':24s} tokens {tot_n:6d}  avg logprob {avg:8.4f}  ppl {math.exp(-avg):8.3f}", flush=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", a.model).strip("-")
    out = HERE / "results" / slug / f"logprob-{time.strftime('%Y%m%d-%H%M%S')}{'-' + a.tag if a.tag else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"params": {"date": time.strftime("%Y-%m-%d %H:%M"), "url": a.url, "model": a.model, "tag": a.tag, "texts": texts},
                               "pooled": {"tokens": tot_n, "avg_logprob": avg, "ppl": math.exp(-avg)}, "per_text": rows}, indent=1))
    print(f"· saved {out}", flush=True)


if __name__ == "__main__":
    main()
