#!/usr/bin/env python3
"""Video-model gauntlet runner — the standing quality test, one command per serve.

Fires the two canonical scene prompts (bench/pasture-video.txt, bench/fish-video.txt — the video
adaptations, MOTION literal) at a /v1/videos server, once per CANONICAL SEED (123123123, 456456456,
789789789 — never the server default). Results land in bench/results/<name>/ (gitignored):
<prompt>[-WxHxF]-s<seed>.mp4 + run.json with timings. The user picks the best of 3 per prompt and
commits the winners as recipes/<recipe>/tests/{pasture,fish}.mp4.

  ./bench/gauntlet-video.py 127.0.0.1:8000                       # canonical clip: 1280x704, 121f @ 24
  ./bench/gauntlet-video.py https://hub-x.myllmbox.com --token $BINDING_TOKEN --name ltx

Stdlib only — runs on any box with python3. Run it ON the serving box (127.0.0.1) or through the
tunnel URL; the management LAN blocks raw ports. Server snaps sizes to /64 and frames to 8k+1.
"""
import argparse
import base64
import json
import os
import time
import urllib.request
import uuid

SEEDS = [123123123, 456456456, 789789789]
PROMPTS = ["pasture", "fish"]
BENCH = os.path.dirname(os.path.abspath(__file__))


def multipart(fields: dict):
    b = uuid.uuid4().hex
    out = b""
    for k, v in fields.items():
        out += (f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    out += f"--{b}--\r\n".encode()
    return out, f"multipart/form-data; boundary={b}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="ip:port or full URL of the serve (proxy or direct)")
    ap.add_argument("--model", default=None, help="results naming only (default: first entry of /v1/models)")
    ap.add_argument("--name", default=None, help="results subfolder name (default: the model id tail)")
    ap.add_argument("--token", default=None, help="bearer token (needed through the proxy/tunnel)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--size", default="1280x704",
                    help="clip WxH (default 1280x704, the canonical video tier; dims snap to /64)")
    ap.add_argument("--frames", type=int, default=121, help="num_frames (snaps to 8k+1; 121 @ 24fps = ~5s)")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()

    base = a.target if a.target.startswith("http") else f"http://{a.target}"
    auth = {"Authorization": f"Bearer {a.token}"} if a.token else {}

    model = a.model
    if not model:
        with urllib.request.urlopen(urllib.request.Request(base + "/v1/models", headers=auth),
                                    timeout=30) as r:
            model = json.load(r)["data"][0]["id"]
    name = a.name or model.split("/")[-1]
    outdir = os.path.join(BENCH, "results", name)
    os.makedirs(outdir, exist_ok=True)

    w, h = a.size.lower().split("x")
    tag = f"-{w}x{h}x{a.frames}"
    report = {"target": base, "model": model, "size": a.size, "frames": a.frames,
              "fps": a.fps, "runs": []}
    for prompt_name in a.prompts.split(","):
        prompt = open(os.path.join(BENCH, f"{prompt_name}-video.txt")).read()
        for seed in (int(s) for s in a.seeds.split(",")):
            t0 = time.time()
            body, ctype = multipart({"prompt": prompt, "seed": seed, "width": w, "height": h,
                                     "num_frames": a.frames, "frame_rate": a.fps})
            req = urllib.request.Request(base + "/v1/videos", data=body, method="POST",
                                         headers={"Content-Type": ctype, **auth})
            try:
                with urllib.request.urlopen(req, timeout=a.timeout) as r:
                    resp = json.load(r)
                if "error" in resp:
                    raise RuntimeError(resp["error"])
                mp4 = base64.b64decode(resp["data"][0]["b64_json"])
                dt = time.time() - t0
                path = os.path.join(outdir, f"{prompt_name}{tag}-s{seed}.mp4")
                open(path, "wb").write(mp4)
                print(f"  {prompt_name} seed {seed}: {dt:.0f}s, {len(mp4)//1024}KB -> {path}")
                report["runs"].append({"prompt": prompt_name, "seed": seed,
                                       "seconds": round(dt, 1), "file": os.path.basename(path)})
            except Exception as e:  # noqa: BLE001
                print(f"  {prompt_name} seed {seed}: FAILED — {e}")
                report["runs"].append({"prompt": prompt_name, "seed": seed, "error": str(e)})

    json.dump(report, open(os.path.join(outdir, "run.json"), "w"), indent=1)
    print(f"report: {outdir}/run.json — pick the best of 3 per prompt and commit as "
          f"recipes/<recipe>/tests/{{pasture,fish}}.mp4")


if __name__ == "__main__":
    main()
