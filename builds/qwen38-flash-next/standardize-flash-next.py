#!/usr/bin/env python3
"""standardize-flash-next.py — take ANY Qwen3.8-Flash-Next checkpoint and put it in the myllmbox layout (hibrid47 standard).

The standard = the body as the source quantized it (vLLM loads modelopt and compressed-tensors alike) + the PLE n-gram
table as 8 NVFP4 shards (`ple-nvfp4-0000k-of-00008.safetensors`, e2m1 codes + fp8 block scales per 16 + one fp32
global) declared in config.json `ple_quantization`, so the serving image holds the table resident on the GPU (cluster
image patch 03) or demand-paged from disk (solo image patches 04/05). Config field names normalized to what the vendor
image expects. Every input file is HARDLINKED, never rewritten (checkpoint dirs here are hardlink families).

What it does with the table, by what it finds in --src:
  bf16 (128 `…ngram_embedding.shard_N.weight`)  → with --table-ref: sample rows, re-quantize them with our quantizer and
                                                  compare to the reference's codes; identical → link the reference's
                                                  8 shards (seconds). Different (a fine-tune touched the table) or no
                                                  --table-ref → quantize it fresh: 2 streaming passes, ~200 MB RAM,
                                                  resumable per shard (about an hour on a Spark).
  nvfp4 (`ple_quantization.format == "nvfp4"`)  → already standard; linked as-is.
  int3 (`…ngram_embedding.qbits_*`)              → refused: int3 cannot be re-quantized without the bf16 source
                                                  (use make-hibrid47.py with --bf16 for that case).
Body files that hold ONLY table tensors are dropped; a file that mixes table and body tensors is re-saved without
the table tensors (streamed, one tensor at a time).

  python3 builds/qwen38-flash-next/standardize-flash-next.py \
      --src models/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4 \
      --table-ref models/myllmbox/Qwen3.8-Flash-Next-hibrid47 \
      --dst models/myllmbox/Qwen3.8-Flash-Next-hibrid47-uncensored
  --check      validate, print the plan and the size table, write nothing
  --requantize force a fresh table quantization even when --table-ref matches
Runs on the host (python3 + safetensors + torch; the vendor image has all three: docker run … --entrypoint python3).
"""
import argparse, hashlib, importlib.util, json, os, re, shutil, struct, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTER = os.path.join(HERE, "cluster", "docker", "make-hibrid47.py")   # quantize(), BLOCK, E2M1MAX, FP8MAX live there
COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
              "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json", "README.md")
BF16_TABLE = ".ngram_embedding.shard_"
INT3_TABLE = ".ngram_embedding.qbits_"
NVFP4_TABLE = (".ngram_embedding.nvfp4_shard_", ".ngram_embedding.nvfp4_global")
# vendor-image field names (the image's model code) vs what newer transformers write
LAYER_TYPE_MAP = {"qwen_sparse_attention": "full_attention"}
SAMPLE_ROWS = 64          # per sampled range, for the identity test
SAMPLES_PER_TABLE = 6     # ranges spread over different bf16 shards


def header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]


def converter():
    spec = importlib.util.spec_from_file_location("mk", CONVERTER)
    mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
    return mk


def tier(name):
    if any(s in name for s in (BF16_TABLE, INT3_TABLE) + NVFP4_TABLE): return "PLE table"
    if name.startswith("mtp."): return "MTP drafter"
    if ".visual." in name or name.startswith("model.visual"): return "vision tower"
    if ".mlp.experts." in name: return "routed experts"
    if ".mlp.shared_expert" in name: return "shared experts"
    if ".self_attn." in name: return "QSA attention"
    if ".linear_attn." in name: return "GDN linear attention"
    if "embed_tokens" in name or name.startswith("lm_head"): return "embeddings / lm_head"
    return "other (norms, hyper-connections, PLE layer, router)"


def size_table(dst_or_src, weight_map, missing_ok=False):
    """bytes per tier from the safetensors headers (data_offsets), for files that exist."""
    sizes, hdr_cache = {}, {}
    for name, f in weight_map.items():
        p = os.path.join(dst_or_src, f)
        if f not in hdr_cache:
            if not os.path.exists(p):
                if missing_ok: hdr_cache[f] = None; continue
                raise FileNotFoundError(p)
            hdr_cache[f] = header(p)[0]
        h = hdr_cache[f]
        if h is None or name not in h: continue
        a, b = h[name]["data_offsets"]; sizes[tier(name)] = sizes.get(tier(name), 0) + (b - a)
    return sizes


def print_sizes(title, sizes):
    tot = sum(sizes.values())
    print(f"  {title}: {tot/2**30:.1f} GiB")
    for k, v in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<48s} {v/2**30:7.2f} GiB  {100*v/tot:5.1f} %")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--table-ref", help="a hibrid47-style checkpoint whose NVFP4 table is linked when the source's bf16 table matches it")
    ap.add_argument("--check", action="store_true"); ap.add_argument("--requantize", action="store_true")
    ap.add_argument("--aux-from", choices=("ref", "src"), default=None,
                    help="where tokenizer/generation/preprocessor files come from (default: ref when --table-ref is given — the "
                         "files the serving image is known to load; a checkpoint re-saved by a newer transformers carries "
                         "tokenizer_config keys and a re-serialized tokenizer.json the vendor image was never tested with)")
    ap.add_argument("--shards", type=int, default=8); ap.add_argument("--chunk", type=int, default=250_000)
    a = ap.parse_args()
    src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
    ref = os.path.abspath(a.table_ref) if a.table_ref else None

    idx_path = os.path.join(src, "model.safetensors.index.json")
    if not os.path.exists(idx_path): sys.exit(f"✗ {idx_path} missing")
    sidx = json.load(open(idx_path))["weight_map"]
    scfg = json.load(open(os.path.join(src, "config.json")))
    stc = scfg.get("text_config", scfg)
    qm = (scfg.get("quantization_config") or stc.get("quantization_config") or {})
    print(f"source: {src}\n  {len(sidx)} tensors in {len(set(sidx.values()))} files; body quantization: "
          f"{qm.get('quant_method') or qm.get('quant_algo') or 'none'}; transformers {scfg.get('transformers_version')}")

    # ---- what kind of table --------------------------------------------------------------------------------------
    bf16_names = sorted((k for k in sidx if BF16_TABLE in k), key=lambda k: int(re.search(r"shard_(\d+)", k).group(1)))
    int3_names = [k for k in sidx if INT3_TABLE in k]
    nvfp4_names = sorted(k for k in sidx if any(s in k for s in NVFP4_TABLE))
    if int3_names and not bf16_names:
        sys.exit("✗ int3 table and no bf16 table in the source — needs the bf16 source: builds/…/make-hibrid47.py --int3 … --bf16 …")
    if nvfp4_names and scfg.get("ple_quantization", {}).get("format") == "nvfp4":
        mode = "nvfp4-passthrough"; table_names = nvfp4_names
    elif bf16_names:
        mode = "bf16"; table_names = bf16_names
    else:
        sys.exit("✗ no PLE table found in the source index")
    prefix = table_names[0].split(".ngram_embedding.")[0] + ".ngram_embedding"
    table_files = sorted({sidx[k] for k in table_names})
    by_file = {}
    for k, f in sidx.items(): by_file.setdefault(f, []).append(k)
    table_only = [f for f in table_files if all(k in table_names for k in by_file[f])]
    mixed = [f for f in table_files if f not in table_only]
    body_files = sorted(f for f in by_file if f not in table_only)
    print(f"  table: {mode}, {len(table_names)} tensors in {table_files}; table-only files dropped: {table_only}; mixed files (re-saved without the table): {mixed}")

    # ---- inputs present? ----------------------------------------------------------------------------------------
    missing = [f for f in body_files if not os.path.exists(os.path.join(src, f))]
    if mode == "bf16":
        missing += [f for f in table_files if not os.path.exists(os.path.join(src, f))]
    if missing:
        print(f"  ✗ {len(missing)} source files missing (download still running?): {missing[:5]}{' …' if len(missing) > 5 else ''}")

    # ---- decide the table plan ---------------------------------------------------------------------------------
    plan = mode
    ref_idx = ref_cfg = None
    if mode == "bf16":
        if ref:
            ref_idx = json.load(open(os.path.join(ref, "model.safetensors.index.json")))["weight_map"]
            ref_cfg = json.load(open(os.path.join(ref, "config.json")))
            if ref_cfg.get("ple_quantization", {}).get("format") != "nvfp4": sys.exit("✗ --table-ref has no NVFP4 table")
            if a.requantize:
                plan = "quantize"
            elif not [f for f in table_files if os.path.exists(os.path.join(src, f))]:
                plan = "identity-test (waiting for the table file)"
            else:
                plan = "link-ref" if table_matches_ref(src, sidx, bf16_names, ref, ref_idx, ref_cfg, converter()) else "quantize"
        else:
            plan = "quantize"
    print(f"  plan: table → {plan}")

    # ---- size table (source, from headers that exist) ------------------------------------------------------------
    try:
        print_sizes("source on disk", size_table(src, sidx, missing_ok=True))
    except Exception as e:
        print(f"  (size table skipped: {e})")
    if a.check or missing or "waiting" in plan:
        if not a.check: sys.exit(1)
        print("check only — nothing written"); return

    # ---- build ------------------------------------------------------------------------------------------------------
    os.makedirs(dst, exist_ok=False)
    new_map = {}
    for f in body_files:
        if f in mixed:
            resave_without(src, f, dst, set(table_names), by_file[f], new_map)
        else:
            os.link(os.path.join(src, f), os.path.join(dst, f))
            for k in by_file[f]: new_map[k] = f
    if plan == "nvfp4-passthrough":
        pleq = scfg["ple_quantization"]
    elif plan == "link-ref":
        ref_names = [k for k in ref_idx if any(s in k for s in NVFP4_TABLE)]
        for f in sorted({ref_idx[k] for k in ref_names}):
            os.link(os.path.join(ref, f), os.path.join(dst, f))
        for k in ref_names:
            new_map[prefix + k.split(".ngram_embedding")[1]] = ref_idx[k]     # re-prefix in case the PLE layer id differs
        pleq = ref_cfg["ple_quantization"]
    else:
        pleq = quantize_table(src, sidx, bf16_names, prefix, dst, new_map, a.shards, a.chunk, converter())
    aux = ref if (a.aux_from or ("ref" if ref else "src")) == "ref" else src
    for f in COPY_FILES:
        p = os.path.join(aux if f != "README.md" else src, f)      # README always the source's (its model card / license terms)
        if os.path.exists(p): shutil.copy2(p, os.path.join(dst, f))
    print(f"  tokenizer / generation / preprocessor files from: {aux}")
    total = sum(os.path.getsize(os.path.join(dst, f)) for f in set(new_map.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_map.items()))},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)

    # ---- config: source's, normalized to the vendor image's names, + ple_quantization -----------------------------
    cfg = json.loads(json.dumps(scfg)); tc = cfg.get("text_config", cfg)
    if "layer_types" in tc:
        tc["layer_types"] = [LAYER_TYPE_MAP.get(t, t) for t in tc["layer_types"]]
    cfg["ple_quantization"] = pleq
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)

    # ---- verify -------------------------------------------------------------------------------------------------------
    seen = {}
    for f in sorted(set(new_map.values())):
        for n in header(os.path.join(dst, f))[0]:
            if n == "__metadata__": continue
            if n in seen: sys.exit(f"✗ {n} in both {seen[n]} and {f}")
            seen[n] = f
    bad = [n for n, f in new_map.items() if seen.get(n) != f]; extra = [n for n in seen if n not in new_map]
    left = [n for n in seen if BF16_TABLE in n or INT3_TABLE in n]
    if bad or extra or left:
        sys.exit(f"✗ verify: {len(bad)} misplaced, {len(extra)} unindexed, {len(left)} old table tensors left")
    n_nv = sum(1 for n in seen if any(s in n for s in NVFP4_TABLE))
    assert n_nv == 2 * pleq["shards"] + 1, f"NVFP4 table tensors {n_nv} != {2*pleq['shards']+1}"
    print_sizes("standardized checkpoint", size_table(dst, new_map))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        p, q = os.path.join(dst, f), os.path.join(src, f)
        if os.path.exists(p) and os.path.exists(q) and sha16(p) != sha16(q):
            print(f"  note: {f} differs between the source and the checkpoint written (aux files taken from {'the reference' if aux == ref else 'the source'})")
    print(f"✓ {dst}: {len(new_map)} tensors, {len(set(new_map.values()))} files, {total/2**30:.1f} GiB; ple_quantization={pleq}")


def table_matches_ref(src, sidx, bf16_names, ref, ref_idx, ref_cfg, mk):
    """Re-quantize sampled rows of the source's bf16 table with OUR quantizer and compare to the reference's codes."""
    import torch
    from safetensors import safe_open
    q = ref_cfg["ple_quantization"]; rows_per_out = q["shard_rows"]
    ref_names = [k for k in ref_idx if ".nvfp4_shard_0.packed" in k]; gname = [k for k in ref_idx if ".nvfp4_global" in k][0]
    hb = header(os.path.join(src, sidx[bf16_names[0]]))[0]
    rows_per_bf16 = hb[bf16_names[0]]["shape"][0]
    with safe_open(os.path.join(ref, ref_idx[gname]), "pt") as f: g = f.get_tensor(gname).item()
    n = len(bf16_names); picks = [(j, r0) for j, r0 in zip(range(0, n, max(1, n // SAMPLES_PER_TABLE)), (0, 1_000_000, 777_777, 2_400_000, 123_456, 50_000))]
    same = True
    for j, r0 in picks:
        r0 = min(r0, rows_per_bf16 - SAMPLE_ROWS); k = bf16_names[j]
        with safe_open(os.path.join(src, sidx[k]), "pt") as f: x = f.get_slice(k)[r0:r0 + SAMPLE_ROWS].to(torch.float32)
        p, s = mk.quantize(x, g)
        grow = j * rows_per_bf16 + r0; out_k, rr = divmod(grow, rows_per_out)
        pk = [k2 for k2 in ref_idx if f".nvfp4_shard_{out_k}.packed" in k2][0]
        with safe_open(os.path.join(ref, ref_idx[pk]), "pt") as f:
            rp = f.get_slice(pk)[rr:rr + SAMPLE_ROWS]; rs = f.get_slice(pk.replace(".packed", ".scales"))[rr:rr + SAMPLE_ROWS]
        ok = torch.equal(p.view(torch.uint8), rp.view(torch.uint8)) and torch.equal(s.view(torch.uint8), rs.view(torch.uint8))
        print(f"    identity test: bf16 shard {j} rows {r0}-{r0+SAMPLE_ROWS} → {'identical' if ok else 'DIFFERENT'}")
        same &= ok
    print(f"  table {'matches' if same else 'does NOT match'} the reference → {'link its 8 shards' if same else 'quantize fresh'}")
    return same


def resave_without(src, f, dst, drop, names, new_map):
    """Copy a shard minus the table tensors it also holds (one tensor at a time)."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    keep = [k for k in names if k not in drop]
    out = {}
    with safe_open(os.path.join(src, f), "pt") as fh:
        for k in keep: out[k] = fh.get_tensor(k)
    save_file(out, os.path.join(dst, f), metadata={"format": "pt"})
    for k in keep: new_map[k] = f
    print(f"  re-saved {f} without {len(names)-len(keep)} table tensors ({len(keep)} kept)")


def quantize_table(src, sidx, bf16_names, prefix, dst, new_map, shards, chunk, mk):
    """make-hibrid47's two-pass streaming quantizer, generalized to bf16 shards spread over any files. Resumable."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    hb = {}
    for k in bf16_names:
        f = sidx[k]
        if f not in hb: hb[f] = header(os.path.join(src, f))[0]
    rows_per_bf16 = hb[sidx[bf16_names[0]]][bf16_names[0]]["shape"][0]; D = hb[sidx[bf16_names[0]]][bf16_names[0]]["shape"][1]
    assert all(hb[sidx[k]][k]["shape"] == [rows_per_bf16, D] for k in bf16_names), "uneven bf16 shards"
    n = len(bf16_names); assert n % shards == 0; per_out = n // shards; S = per_out * rows_per_bf16; R = n * rows_per_bf16
    out_name = lambda k: f"ple-nvfp4-{k+1:05d}-of-{shards:05d}.safetensors"
    print(f"  quantizing table: {n} bf16 shards × {rows_per_bf16} × {D} = {R:,} rows → {shards} NVFP4 shards ({R*D/2/2**30:.1f} GiB packed + {R*D/mk.BLOCK/2**30:.1f} GiB scales)", flush=True)
    handles = {}
    def sl(k):
        f = sidx[k]
        if f not in handles: handles[f] = safe_open(os.path.join(src, f), "pt")
        return handles[f].get_slice(k)
    gpath = os.path.join(dst, ".nvfp4_global.json"); t0 = time.time()
    if os.path.exists(gpath):
        g = json.load(open(gpath))["global"]
    else:
        amax = 0.0
        for i, k in enumerate(bf16_names):
            s = sl(k)
            for r in range(0, rows_per_bf16, chunk): amax = max(amax, s[r:min(rows_per_bf16, r + chunk)].to(torch.float32).abs().amax().item())
            if i % 16 == 15: print(f"    amax pass {i+1}/{n} ({time.time()-t0:.0f}s) running amax {amax:.4f}", flush=True)
        g = amax / (mk.E2M1MAX * mk.FP8MAX); json.dump({"amax": amax, "global": g}, open(gpath, "w"))
    print(f"  global scale {g:.6e}", flush=True)
    for k in range(shards):
        f = os.path.join(dst, out_name(k))
        if os.path.exists(f + ".ok"): continue
        packed = torch.empty((S, D // 2), dtype=torch.uint8); scales = torch.empty((S, D // mk.BLOCK), dtype=torch.float8_e4m3fn); row = 0
        for j in range(k * per_out, (k + 1) * per_out):
            s = sl(bf16_names[j])
            for r in range(0, rows_per_bf16, chunk):
                x = s[r:min(rows_per_bf16, r + chunk)].to(torch.float32); p, sc = mk.quantize(x, g)
                packed[row:row + x.shape[0]] = p; scales[row:row + x.shape[0]] = sc; row += x.shape[0]
        assert row == S
        t = {f"{prefix}.nvfp4_shard_{k}.packed": packed, f"{prefix}.nvfp4_shard_{k}.scales": scales}
        if k == 0: t[f"{prefix}.nvfp4_global"] = torch.tensor(g, dtype=torch.float32)
        save_file(t, f, metadata={"format": "pt"}); open(f + ".ok", "w").write("ok")
        print(f"    shard {k+1}/{shards} written ({time.time()-t0:.0f}s)", flush=True)
    for k in range(shards):
        new_map[f"{prefix}.nvfp4_shard_{k}.packed"] = out_name(k); new_map[f"{prefix}.nvfp4_shard_{k}.scales"] = out_name(k)
        p = os.path.join(dst, out_name(k) + ".ok")
        if os.path.exists(p): os.unlink(p)
    new_map[f"{prefix}.nvfp4_global"] = out_name(0)
    if os.path.exists(gpath): os.unlink(gpath)
    return {"format": "nvfp4", "block": mk.BLOCK, "rows": R, "dim": D, "shards": shards, "shard_rows": S}


if __name__ == "__main__":
    main()
