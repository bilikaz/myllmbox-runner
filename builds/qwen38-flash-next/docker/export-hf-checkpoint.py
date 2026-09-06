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
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="hibrid checkpoint dir (with model.safetensors.index.json)")
    ap.add_argument("dst", help="output dir for the clean repo")
    ap.add_argument("--shard-gb", type=float, default=4.0, help="target shard size (GB, default 4)")
    ap.add_argument("--strip", default=None,
                    help="regex: tensors whose NAME matches are dropped from the repo entirely "
                         "(shards AND index). e.g. the 128 bf16 PLE bank shards when the repo "
                         "ships the table pre-quantized (see --qbits-*).")
    ap.add_argument("--qbits-dir", default=None,
                    help="dir with raw quantized-PLE artifacts (*.packed/*.scales/*.mins) to embed "
                         "as ORDINARY checkpoint tensors (<prefix>.qbits_packed/_scales/_mins) and "
                         "declare in config.json as ple_quantization — the standard shipping format.")
    ap.add_argument("--qbits-prefix",
                    default="model.language_model.layers.1.ple.ple_embedding.ngram_embedding",
                    help="tensor-name prefix for the embedded qbits tensors")
    ap.add_argument("--qbits-bits", type=int, default=3)
    ap.add_argument("--qbits-group", type=int, default=160)
    ap.add_argument("--qbits-rows", type=int, default=320001536)
    ap.add_argument("--qbits-dim", type=int, default=160)
    a = ap.parse_args()

    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    weight_map: dict[str, str] = idx["weight_map"]
    if a.strip:
        rx = re.compile(a.strip)
        dropped = [n for n in weight_map if rx.search(n)]
        for n in dropped:
            del weight_map[n]
        print(f"  --strip {a.strip!r}: dropping {len(dropped)} tensors from the repo")
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

    if a.qbits_dir:
        import glob
        V, D, bits, group = a.qbits_rows, a.qbits_dim, a.qbits_bits, a.qbits_group
        bpr = {2: D // 4, 3: (D // 8) * 3, 4: D // 2}[bits]
        G = D // group
        find = lambda ext: glob.glob(os.path.join(a.qbits_dir, f"*.{ext}"))[0]  # noqa: E731
        emit = {
            f"{a.qbits_prefix}.qbits_packed":
                torch.from_file(find("packed"), shared=True, size=V * bpr, dtype=torch.uint8).view(V, bpr),
            f"{a.qbits_prefix}.qbits_scales":
                torch.from_file(find("scales"), shared=True, size=V * G, dtype=torch.float16).view(V, G),
            f"{a.qbits_prefix}.qbits_mins":
                torch.from_file(find("mins"), shared=True, size=V * G, dtype=torch.float16).view(V, G),
        }
        for name, t in emit.items():
            nbytes = t.numel() * t.element_size()
            if buf and buf_bytes + nbytes > limit:
                flush()
            buf[name] = t
            buf_bytes += nbytes
            total += nbytes
            print(f"  embedding {name}: {nbytes/2**30:.2f} GiB")
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

    if a.qbits_dir:
        cfg_p = os.path.join(a.dst, "config.json")
        cfg = json.load(open(cfg_p))
        cfg["ple_quantization"] = {"bits": a.qbits_bits, "group": a.qbits_group,
                                   "rows": a.qbits_rows, "dim": a.qbits_dim}
        json.dump(cfg, open(cfg_p, "w"), indent=2)
        print("  config.json: ple_quantization declared")

    print(f"done: {shard_i} clean shards, {total/2**30:.2f} GiB -> {a.dst}")
    print("every tensor exists exactly once (stock-loader safe); upload with: hf upload <repo> " + a.dst)


if __name__ == "__main__":
    main()
