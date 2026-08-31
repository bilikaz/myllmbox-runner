#!/usr/bin/env python3
"""Repack a hibrid checkpoint into a CLEAN, stock-loadable HF repo.

Our hibrid converters hardlink the source shards and re-point replaced tensors via the index —
fast and disk-cheap, but the stale originals remain physically inside the old shard files, and
stock vLLM iterates FILES, not the index (packed 4-bit replacements shape-assert on the stale
bf16 copies; our images carry MBX_INDEX_STRICT to skip them). A public checkpoint must load with
ZERO patches, so this tool walks model.safetensors.index.json and re-serializes every tensor into
fresh shards — each tensor exists exactly once, exactly where the index says.

  ./scripts/export-hf-checkpoint.py /models/myllmbox/Qwen3.8-Flash-Next-hibrid46 \
      /models/export/Qwen3.8-Flash-Next-hibrid46-hf [--shard-gb 4]

Copies every non-shard file (config.json with its quantization_config, tokenizer, README if
present) verbatim. Output is upload-ready: hf upload <repo> <outdir>. CPU-only, streams one
tensor at a time (safe alongside a serve); needs free disk = checkpoint size.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="hibrid checkpoint dir (with model.safetensors.index.json)")
    ap.add_argument("dst", help="output dir for the clean repo")
    ap.add_argument("--shard-gb", type=float, default=4.0, help="target shard size (GB, default 4)")
    a = ap.parse_args()

    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    weight_map: dict[str, str] = idx["weight_map"]
    os.makedirs(a.dst, exist_ok=True)

    # group tensors by their OWNING file (per the index — the single source of truth),
    # preserving index order so related tensors stay adjacent.
    by_file: dict[str, list[str]] = {}
    for name, f in weight_map.items():
        by_file.setdefault(f, []).append(name)

    limit = int(a.shard_gb * 2**30)
    new_map: dict[str, str] = {}
    total = 0
    shard_i = 0
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0

    def flush():
        nonlocal shard_i, buf, buf_bytes
        if not buf:
            return
        shard_i += 1
        fname = f"model-{shard_i:05d}.safetensors"  # renamed at the end once count is known
        save_file(buf, os.path.join(a.dst, fname))
        for n in buf:
            new_map[n] = fname
        print(f"  wrote {fname}: {len(buf)} tensors, {buf_bytes/2**30:.2f} GiB")
        buf, buf_bytes = {}, 0

    for src_file, names in sorted(by_file.items()):
        with safe_open(os.path.join(a.src, src_file), framework="pt") as f:
            present = set(f.keys())
            for name in names:
                if name not in present:
                    raise SystemExit(f"index maps {name} -> {src_file} but the tensor is not there")
                t = f.get_tensor(name)
                nbytes = t.numel() * t.element_size()
                if buf and buf_bytes + nbytes > limit:
                    flush()
                buf[name] = t
                buf_bytes += nbytes
                total += nbytes
    flush()

    # final names carry the count (HF convention), rewrite map accordingly
    renames = {f"model-{i:05d}.safetensors": f"model-{i:05d}-of-{shard_i:05d}.safetensors"
               for i in range(1, shard_i + 1)}
    for old, new in renames.items():
        os.rename(os.path.join(a.dst, old), os.path.join(a.dst, new))
    new_map = {n: renames[f] for n, f in new_map.items()}
    json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
              open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)

    for f in os.listdir(a.src):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            continue
        p = os.path.join(a.src, f)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(a.dst, f))
            print(f"  copied {f}")

    print(f"done: {shard_i} clean shards, {total/2**30:.2f} GiB -> {a.dst}")
    print("every tensor exists exactly once (stock-loader safe); upload with: hf upload <repo> " + a.dst)


if __name__ == "__main__":
    main()
