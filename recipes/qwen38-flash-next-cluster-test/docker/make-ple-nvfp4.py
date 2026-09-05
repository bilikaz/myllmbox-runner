#!/usr/bin/env python3
"""make-ple-nvfp4.py — re-quantize the Qwen3.8-Flash-Next PLE n-gram table from bf16 to NVFP4, as a checkpoint.

INPUT   --int3  the hibrid46 checkpoint (everything except the table is reused by HARDLINK — 21 shard files,
                of which 20 and 21 hold ONLY the int3 table tensors and are simply not linked)
        --bf16  the hibrid46-off checkpoint (its ONE 95 GiB shard file holds the 128 bf16 table shards
                `…ngram_embedding.shard_N.weight`, [2,500,012 × 160] each = 320,001,536 rows)
OUTPUT  --dst   a new self-describing checkpoint: hardlinks + 8 new shard files `ple-nvfp4-0000k-of-00008.safetensors`
                (16 bf16 shards each = 40,000,192 rows) holding
                    …ngram_embedding.nvfp4_shard_k.packed   uint8          [rows, 80]   two e2m1 codes per byte (even value = low nibble)
                    …ngram_embedding.nvfp4_shard_k.scales   float8_e4m3fn  [rows, 10]   one block scale per 16 values, in units of the global
                    …ngram_embedding.nvfp4_global           float32        []           table amax / (6 × 448)
                + rewritten model.safetensors.index.json + config.json `ple_quantization`
                  = {"format":"nvfp4","block":16,"rows":R,"dim":160,"shards":8,"shard_rows":S}

NVFP4 here = the standard layout: e2m1 codes {0, .5, 1, 1.5, 2, 3, 4, 6} × sign, FP8-e4m3 block scale per 16
values, one FP32 per-tensor global scale (block scale ≤ 448 by construction). Encoding uses the FP8-ROUNDED block
scale so codes are nearest-to-effective, not nearest-to-ideal. ~4.5 bits/value → 28.6 GiB (int3 was 17.9).
Two streaming passes over the bf16 source (global amax, then quantize), 250k-row chunks, ~200 MB working set —
runs on a serving box. Resumable per output shard (a .ok marker per file).

  docker run --rm -v /home/valdas/spark-vllm-docker/models:/models -v $PWD/docker/make-ple-nvfp4.py:/x.py:ro \\
      --entrypoint python3 myllmbox/qwen38-flash-next-vllm:v1 /x.py \\
      --int3 /models/myllmbox/Qwen3.8-Flash-Next-hibrid46 --bf16 /models/myllmbox/Qwen3.8-Flash-Next-hibrid46-off \\
      --dst /models/myllmbox/Qwen3.8-Flash-Next-hibrid46-ple4
"""
import argparse, json, os, re, shutil, struct, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])   # midpoints → nearest e2m1 magnitude
BLOCK, FP8MAX, E2M1MAX = 16, 448.0, 6.0


def header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def quantize(x: torch.Tensor, gscale: float):
    """x [n, D] float32 → packed uint8 [n, D/2], scales fp8 [n, D/16]."""
    n, D = x.shape
    b = x.view(n, D // BLOCK, BLOCK)
    amax = b.abs().amax(-1)                                       # [n, D/16]
    sf8 = (amax / E2M1MAX / gscale).to(torch.float8_e4m3fn)       # block scale in units of the global, fp8-rounded
    eff = sf8.to(torch.float32) * gscale                           # the scale the DEQUANT will use
    a = b / eff.clamp_min(1e-30).unsqueeze(-1)                     # normalised values, |a| ≲ 6 (zero blocks → 0)
    idx = torch.bucketize(a.abs(), BOUNDS).to(torch.uint8)        # 0..7
    code = idx | ((a < 0).to(torch.uint8) << 3)                    # sign in bit 3
    code = code.view(n, D)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed, sf8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--int3", required=True); ap.add_argument("--bf16", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=250_000, help="rows per quantize chunk")
    a = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))

    # ---- source geometry ------------------------------------------------------------------------
    ix3 = json.load(open(f"{a.int3}/model.safetensors.index.json"))["weight_map"]
    int3_names = [k for k in ix3 if ".ngram_embedding.qbits_" in k]
    assert int3_names, "no int3 table in --int3"
    prefix = int3_names[0].split(".ngram_embedding.")[0] + ".ngram_embedding"       # …layers.1.ple.ple_embedding.ngram_embedding
    int3_files = {ix3[k] for k in int3_names}
    for f in int3_files:
        assert all(ix3[k] in int3_files or ".qbits_" not in k for k in ix3), "unexpected"
        others = [k for k, v in ix3.items() if v == f and ".qbits_" not in k]
        assert not others, f"{f} also holds non-table tensors {others[:3]} — refusing (would need a rewrite)"

    ixb = json.load(open(f"{a.bf16}/model.safetensors.index.json"))["weight_map"]
    shard_names = sorted((k for k in ixb if ".ngram_embedding.shard_" in k), key=lambda k: int(re.search(r"shard_(\d+)", k).group(1)))
    assert shard_names, "no bf16 table shards in --bf16"
    bf16_file = {ixb[k] for k in shard_names}; assert len(bf16_file) == 1, "bf16 shards span several files — extend me"
    bf16_file = f"{a.bf16}/{bf16_file.pop()}"
    h = header(bf16_file)
    rows_per_bf16 = h[shard_names[0]]["shape"][0]; D = h[shard_names[0]]["shape"][1]
    assert all(h[k]["shape"] == [rows_per_bf16, D] for k in shard_names), "uneven bf16 shards — extend me"
    n_bf16 = len(shard_names); R = n_bf16 * rows_per_bf16
    assert n_bf16 % a.shards == 0, f"{n_bf16} bf16 shards not divisible into {a.shards} outputs"
    per_out = n_bf16 // a.shards; S = per_out * rows_per_bf16
    assert D % BLOCK == 0 and D % 2 == 0
    cfg3 = json.load(open(f"{a.int3}/config.json"))
    q3 = cfg3.get("ple_quantization") or {}
    assert int(q3.get("rows", R)) == R and int(q3.get("dim", D)) == D, f"int3 declares {q3}, bf16 source is {R}x{D}"
    print(f"table: {n_bf16} bf16 shards × {rows_per_bf16} rows × {D} = {R:,} rows → {a.shards} NVFP4 shards × {S:,} rows "
          f"({R*D/2/2**30:.1f} GiB packed + {R*D/BLOCK/2**30:.1f} GiB scales)", flush=True)

    # ---- destination: hardlink everything from the int3 checkpoint except its table files -------------------
    os.makedirs(a.dst, exist_ok=True)
    for name in sorted(os.listdir(a.int3)):
        src = f"{a.int3}/{name}"; dst = f"{a.dst}/{name}"
        if name in int3_files or name in ("model.safetensors.index.json", "config.json") or os.path.isdir(src):
            continue
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    print(f"· linked {len(os.listdir(a.dst))} files from {a.int3}", flush=True)

    out_name = lambda k: f"ple-nvfp4-{k+1:05d}-of-{a.shards:05d}.safetensors"
    st = safe_open(bf16_file, "pt")

    # ---- pass 1: global amax (skipped if every output shard already exists) ---------------------------------
    gpath = f"{a.dst}/.nvfp4_global.json"
    if os.path.exists(gpath):
        gscale = json.load(open(gpath))["global"]
        print(f"· global scale (cached): {gscale:.6e}", flush=True)
    else:
        t0 = time.time(); amax = 0.0
        for i, k in enumerate(shard_names):
            sl = st.get_slice(k)
            for r in range(0, rows_per_bf16, a.chunk):
                amax = max(amax, sl[r:min(rows_per_bf16, r + a.chunk)].to(torch.float32).abs().amax().item())
            if i % 16 == 15:
                print(f"  amax pass {i+1}/{n_bf16}  ({time.time()-t0:.0f}s)  running amax {amax:.4f}", flush=True)
        gscale = amax / (E2M1MAX * FP8MAX)
        json.dump({"amax": amax, "global": gscale}, open(gpath, "w"))
        print(f"· table amax {amax:.4f} → global scale {gscale:.6e}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- pass 2: quantize into 8 output shards ----------------------------------------------------------------
    t0 = time.time()
    for k in range(a.shards):
        f = f"{a.dst}/{out_name(k)}"
        if os.path.exists(f + ".ok"):
            print(f"· shard {k+1}/{a.shards} present — skip", flush=True); continue
        packed = torch.empty((S, D // 2), dtype=torch.uint8)
        scales = torch.empty((S, D // BLOCK), dtype=torch.float8_e4m3fn)
        row = 0
        for j in range(k * per_out, (k + 1) * per_out):
            sl = st.get_slice(shard_names[j])
            for r in range(0, rows_per_bf16, a.chunk):
                x = sl[r:min(rows_per_bf16, r + a.chunk)].to(torch.float32)
                p, s = quantize(x, gscale)
                packed[row:row + x.shape[0]] = p; scales[row:row + x.shape[0]] = s
                row += x.shape[0]
        assert row == S
        tensors = {f"{prefix}.nvfp4_shard_{k}.packed": packed, f"{prefix}.nvfp4_shard_{k}.scales": scales}
        if k == 0:
            tensors[f"{prefix}.nvfp4_global"] = torch.tensor(gscale, dtype=torch.float32)
        save_file(tensors, f, metadata={"format": "pt"})
        open(f + ".ok", "w").write("ok")
        print(f"· shard {k+1}/{a.shards} written  {os.path.getsize(f)/2**30:.2f} GiB  ({time.time()-t0:.0f}s elapsed)", flush=True)

    # ---- index + config --------------------------------------------------------------------------------------
    idx = json.load(open(f"{a.int3}/model.safetensors.index.json"))
    wm = {k: v for k, v in idx["weight_map"].items() if ".qbits_" not in k}
    for k in range(a.shards):
        wm[f"{prefix}.nvfp4_shard_{k}.packed"] = out_name(k); wm[f"{prefix}.nvfp4_shard_{k}.scales"] = out_name(k)
    wm[f"{prefix}.nvfp4_global"] = out_name(0)
    total = sum(os.path.getsize(f"{a.dst}/{f}") for f in set(wm.values()))
    json.dump({"metadata": {**(idx.get("metadata") or {}), "total_size": total}, "weight_map": dict(sorted(wm.items()))},
              open(f"{a.dst}/model.safetensors.index.json", "w"), indent=1)
    cfg3["ple_quantization"] = {"format": "nvfp4", "block": BLOCK, "rows": R, "dim": D, "shards": a.shards, "shard_rows": S}
    json.dump(cfg3, open(f"{a.dst}/config.json", "w"), indent=1)
    for k in range(a.shards):
        p = f"{a.dst}/{out_name(k)}.ok"
        if os.path.exists(p): os.unlink(p)
    if os.path.exists(gpath): os.unlink(gpath)
    print(f"✓ {a.dst}: {len(set(wm.values()))} shard files, {total/2**30:.1f} GiB, ple_quantization={cfg3['ple_quantization']}", flush=True)


if __name__ == "__main__":
    main()
