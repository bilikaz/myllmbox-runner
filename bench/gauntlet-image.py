#!/usr/bin/env python3
"""Image-model gauntlet runner — the standing quality test, one command per serve.

Fires the two canonical scene prompts (bench/pasture-image.txt, bench/fish-image.txt) at an
OpenAI-Images server, once per CANONICAL SEED (123123123, 456456456, 789789789 — never the server
default: a fixed default seed returns the identical image N times). Results land in
bench/results/<name>/ (gitignored): <prompt>-s<seed>.png + run.json with timings. The user picks
the best of 3 per prompt and commits the winners as recipes/<recipe>/tests/{pasture,fish}.png.

  ./bench/gauntlet-image.py 127.0.0.1:8000                      # edit-mode server (qwen-image-edit)
  ./bench/gauntlet-image.py 127.0.0.1:8000 --mode generate      # pure text->image (flux2-dev)
  ./bench/gauntlet-image.py https://hub-x.myllmbox.com --token $BINDING_TOKEN --name flux2

Stdlib only (hand-rolled multipart + a generated blank-canvas PNG) — runs on any box with python3.
Run it ON the serving box (127.0.0.1) or through the tunnel URL; the management LAN blocks raw ports.
"""
import argparse
import json
import os
import struct
import sys
import time
import urllib.request
import uuid
import zlib

SEEDS = [123123123, 456456456, 789789789]
PROMPTS = ["pasture", "fish"]
BENCH = os.path.dirname(os.path.abspath(__file__))


def blank_canvas_png(w: int = 1920, h: int = 1080) -> bytes:
    """Minimal near-white PNG (the edit-mode reference input), no PIL needed."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    row = b"\x00" + b"\xfa\xfa\xf8" * w
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * h, 6))
            + chunk(b"IEND", b""))


def multipart(fields: dict, file_field: str, filename: str, blob: bytes):
    b = uuid.uuid4().hex
    out = b""
    for k, v in fields.items():
        out += (f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    out += (f"--{b}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
            f"filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n").encode()
    out += blob + f"\r\n--{b}--\r\n".encode()
    return out, f"multipart/form-data; boundary={b}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="ip:port or full URL of the serve (proxy or direct)")
    ap.add_argument("--mode", choices=["edit", "generate"], default="edit",
                    help="edit = /v1/images/edits with a blank reference (default); generate = /v1/images/generations")
    ap.add_argument("--model", default=None, help="model id (default: first entry of /v1/models)")
    ap.add_argument("--name", default=None, help="results subfolder name (default: the model id tail)")
    ap.add_argument("--token", default=None, help="bearer token (needed through the proxy/tunnel)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--timeout", type=int, default=420)
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

    canvas = blank_canvas_png()
    report = {"target": base, "model": model, "mode": a.mode, "runs": []}
    for prompt_name in a.prompts.split(","):
        prompt = open(os.path.join(BENCH, f"{prompt_name}-image.txt")).read()
        for seed in (int(s) for s in a.seeds.split(",")):
            t0 = time.time()
            if a.mode == "edit":
                body, ctype = multipart({"model": model, "prompt": prompt, "seed": seed},
                                        "image", "canvas.png", canvas)
                req = urllib.request.Request(base + "/v1/images/edits", data=body, method="POST",
                                             headers={"Content-Type": ctype, **auth})
            else:
                req = urllib.request.Request(
                    base + "/v1/images/generations", method="POST",
                    data=json.dumps({"model": model, "prompt": prompt, "seed": seed,
                                     "size": "1920x1080"}).encode(),
                    headers={"Content-Type": "application/json", **auth})
            try:
                with urllib.request.urlopen(req, timeout=a.timeout) as r:
                    resp = json.load(r)
                import base64
                png = base64.b64decode(resp["data"][0]["b64_json"])
                dt = time.time() - t0
                path = os.path.join(outdir, f"{prompt_name}-s{seed}.png")
                open(path, "wb").write(png)
                print(f"  {prompt_name} seed {seed}: {dt:.0f}s -> {path}")
                report["runs"].append({"prompt": prompt_name, "seed": seed,
                                       "seconds": round(dt, 1), "file": os.path.basename(path)})
            except Exception as e:  # noqa: BLE001
                print(f"  {prompt_name} seed {seed}: FAILED — {e}")
                report["runs"].append({"prompt": prompt_name, "seed": seed, "error": str(e)})

    json.dump(report, open(os.path.join(outdir, "run.json"), "w"), indent=1)
    print(f"report: {outdir}/run.json — pick the best of 3 per prompt and commit as "
          f"recipes/<recipe>/tests/{{pasture,fish}}.png")


if __name__ == "__main__":
    main()
