"""Build the HIBRID45 checkpoint: Inferact NVFP4 experts + NVFP4 dense EXCEPT attention.

hibrid4 (all 300 dense modules at nvfp4) hit 53.5 tok/s code but FAILED the quality gate:
prose acceptance sagged to ~0.60 first-position and the pasture grew defects (black sun rays,
detached eyes) — a felt intelligence drop. Diagnosis: QSA q/k/v/o are COMPARATORS (q.k dot
products rank tokens; rankings flip on small weight noise) — the slice Alibaba refused to
quantize at all. hibrid45 = one-variable step: QSA q/k/v/o stay UNTOUCHED BF16 (not even
converted — original hardlinked shards), GDN in_proj/out_proj + shared expert stay nvfp4.
Byte map: dense/step 1.64 GiB (vs 1.35 hibrid4 / 2.71 fp8 champion) -> ~13.2-13.3 steps
projected, code ~51-52. If quality STILL fails, the GDN projections are convicted and the
fp8 champion stands. If it passes, optional +2%: QSA to the champion's proven fp8.

The fp8 hibrid proved the thesis (+30% steps, pasture-gated, 50.0 peak): the decode-path dense is the
one slice nobody quantizes. This goes one notch further: the SAME 300 modules to NVFP4 W4A16
(4-bit weights, BF16 activations, group-16 fp8 scales + fp32 per-tensor global) — another ~1.3 GiB
off the ~4.7 GiB step. vLLM's mixed dispatcher handles W4A16_NVFP4 NATIVELY (Marlin kernel, forced
via use_a16 — no dispatch splice needed at all, unlike fp8). QUALITY IS THE OPEN QUESTION: 4-bit on
q/k/v/o + GDN projections is far sharper than fp8 — pasture gate + accept-rate watch decide.

Format (mirrors vLLM's ModelOptNvFp4W4A16LinearMethod exactly, verified from the loader):
  weight          uint8   [out, in/2]  packed nibbles along input; LOW nibble = even element,
                                       sign bit 0x08, magnitude index 0x07 -> [0,.5,1,1.5,2,3,4,6]
  weight_scale    fp8-e4m3 [out, in/16] per-group scale IN UNITS OF weight_scale_2
  weight_scale_2  fp32    scalar        global = amax / (6.0 * 448.0)  (ModelOpt convention, amax/2688)
Fused groups (qkv, gate_up, in_proj_qkvz) get a SHARED weight_scale_2 (max amax over members) — the
loader warns and loses accuracy when fused partitions carry different globals; we control the
converter, so we don't.

Output = models/myllmbox/Qwen3.8-Flash-Next-hibrid4 (originals hardlinked + one new fp4 shard +
rewritten index + the same three config folds as the fp8 hibrid: MTP renumbering, namespace
tripling, ignore-list strip).
"""
import json, os, re, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/models/Inferact/Qwen3.8-Flash-Next-NVFP4"
DST = "/models/myllmbox/Qwen3.8-Flash-Next-hibrid46"
GROUP = 16
FP4_MAX = 6.0
FP8_MAX = 448.0
FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
ALGO = "W4A16_NVFP4"
SHARD_NAME = "model-hibrid46-nvfp4.safetensors"

# hibrid46: shared expert BACK to bf16 (buys ~0.4% steps at unmeasured risk — and Alibaba's own
# FP8 release excludes it too). GDN stays nvfp4 (~3.5% steps, A/B-validated).
PATTERNS = [
    r"^model\.language_model\.layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
]
FUSED_GROUPS = {
    "linear_attn.in_proj_qkvz": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
}


def _quant_nvfp4(wf: torch.Tensor, global_amax: float):
    """-> (packed u8 [out,in/2], scale fp8 [out,in/16], scale2 fp32 scalar, mean rel err)."""
    out_f, in_f = wf.shape
    assert in_f % GROUP == 0, f"in_features {in_f} not group-divisible"
    scale2 = max(global_amax / (FP4_MAX * FP8_MAX), 1e-12)
    g = wf.reshape(out_f, in_f // GROUP, GROUP)
    gmax = g.abs().amax(dim=-1)
    s_fp8 = (gmax / FP4_MAX / scale2).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)
    s_real = (s_fp8.to(torch.float32) * scale2).clamp(min=1e-20)   # decode-symmetric encode
    scaled = g / s_real[..., None]
    cb = torch.tensor(FP4_VALUES, dtype=torch.float32)
    idx = (scaled.abs().unsqueeze(-1) - cb).abs().argmin(dim=-1).to(torch.uint8)
    sign = ((scaled < 0) & (idx > 0)).to(torch.uint8)              # never emit -0
    nib = (idx | (sign << 3)).reshape(out_f, in_f)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()     # LOW nibble = even element
    deq = (cb[idx.long()] * torch.where(sign.bool(), -1.0, 1.0) * s_real[..., None]).reshape(out_f, in_f)
    rel = ((deq - wf).abs().mean() / (wf.abs().mean() + 1e-12)).item()
    return packed, s_fp8, torch.tensor(scale2, dtype=torch.float32), rel


def _complete() -> bool:
    try:
        qc = json.load(open(f"{DST}/config.json")).get("quantization_config") or {}
        if qc.get("quant_algo") != "MIXED_PRECISION" or not qc.get("quantized_layers"):
            return False
        idx = json.load(open(f"{DST}/model.safetensors.index.json"))
        shard = f"{DST}/{SHARD_NAME}"
        return (
            any(v == SHARD_NAME for v in idx["weight_map"].values())
            and os.path.exists(shard) and os.path.getsize(shard) > 2**30
        )
    except Exception:
        return False


def main() -> None:
    if _complete():
        print(f"hibrid46 checkpoint already complete: {DST} — skipping conversion")
        return
    idx_path = f"{SRC}/model.safetensors.index.json"
    if not os.path.exists(idx_path):
        raise SystemExit(
            f"source checkpoint missing: {SRC}\n"
            "fetch it first (on the host): ./download.sh Inferact/Qwen3.8-Flash-Next-NVFP4"
        )
    index = json.load(open(idx_path))
    wmap = dict(index["weight_map"])

    targets = sorted(
        n[: -len(".weight")] for n in wmap
        if n.endswith(".weight") and any(re.match(p, n[: -len(".weight")]) for p in PATTERNS)
    )
    print(f"candidate dense modules: {len(targets)}")

    os.makedirs(DST, exist_ok=True)
    for f in os.listdir(SRC):
        s, d = f"{SRC}/{f}", f"{DST}/{f}"
        if os.path.isfile(s) and not os.path.exists(d):
            os.link(s, d)

    handles: dict[str, object] = {}

    def _tensor(base):
        shard = wmap[base + ".weight"]
        if shard not in handles:
            handles[shard] = safe_open(f"{SRC}/{shard}", framework="pt", device="cpu")
        return handles[shard].get_tensor(base + ".weight")

    # pass 1 — per-module amax, then SHARED global per fused group (loader-accuracy rule)
    amax: dict[str, float] = {b: _tensor(b).to(torch.float32).abs().max().item() for b in targets}
    shared = dict(amax)
    tset = set(targets)
    for base in targets:
        for _fused, sfxs in FUSED_GROUPS.items():
            if any(base.endswith("." + s) for s in sfxs):
                parent = base
                for s in sfxs:
                    if parent.endswith("." + s):
                        parent = parent[: -len("." + s)]
                        break
                members = [f"{parent}.{s}" for s in sfxs]
                if all(m in tset for m in members):
                    gmax = max(amax[m] for m in members)
                    for m in members:
                        shared[m] = gmax

    # pass 2 — quantize
    new_tensors: dict[str, torch.Tensor] = {}
    quantized_layers: dict[str, dict] = {}
    done_bytes, worst_rel = 0, 0.0
    for base in targets:
        wf = _tensor(base).to(torch.float32)
        packed, s_fp8, s2, rel = _quant_nvfp4(wf, shared[base])
        new_tensors[base + ".weight"] = packed
        new_tensors[base + ".weight_scale"] = s_fp8
        new_tensors[base + ".weight_scale_2"] = s2
        quantized_layers[base] = {"quant_algo": ALGO, "group_size": GROUP}
        done_bytes += packed.numel()
        worst_rel = max(worst_rel, rel)
        assert rel < 0.25, f"{base}: nvfp4 mean rel err {rel:.3f} implausibly high"
    print(f"quantized: {len(quantized_layers)} modules, {done_bytes/2**30:.2f} GiB fp4-packed; "
          f"worst mean-rel-err {worst_rel:.4f}")
    assert quantized_layers, "nothing quantized — pattern mismatch?"

    save_file(new_tensors, f"{DST}/{SHARD_NAME}")
    for name in new_tensors:
        wmap[name] = SHARD_NAME
    index["weight_map"] = wmap
    os.unlink(f"{DST}/model.safetensors.index.json")
    json.dump(index, open(f"{DST}/model.safetensors.index.json", "w"))

    # fused-module entries (rule-1 direct lookup — full fused suffix, see the fp8 hibrid's bug history)
    fused_added = 0
    quantized_set = set(quantized_layers)
    for base in list(quantized_set):
        for fused_suffix, shard_suffixes in FUSED_GROUPS.items():
            if any(base.endswith("." + sfx) for sfx in shard_suffixes):
                parent = base
                for sfx in shard_suffixes:
                    if parent.endswith("." + sfx):
                        parent = parent[: -len("." + sfx)]
                        break
                if all(f"{parent}.{sfx}" in quantized_set for sfx in shard_suffixes):
                    key = f"{parent}.{fused_suffix}"
                    if key not in quantized_layers:
                        quantized_layers[key] = {"quant_algo": ALGO, "group_size": GROUP}
                        fused_added += 1
    print(f"fused-module entries added: {fused_added}")

    parents = sorted({n.split(".experts.")[0] for n in wmap if ".experts." in n})
    for p in parents:
        quantized_layers[f"{p}.experts.up_proj"] = {"quant_algo": "NVFP4", "group_size": 16}
    print(f"expert parents declared NVFP4: {len(parents)}")

    cfg = json.load(open(f"{SRC}/config.json"))

    # the same three folds as the fp8 hibrid (see its make-hibrid.py for the war stories):
    mtp_idx = (cfg.get("text_config") or cfg)["num_hidden_layers"]
    for k in [k for k in quantized_layers if k.startswith("mtp.layers.0.")]:
        quantized_layers[k.replace("mtp.layers.0.", f"mtp.layers.{mtp_idx}.", 1)] = dict(quantized_layers[k])
    LM_P = "model.language_model."
    expanded: dict[str, dict] = {}
    for k, v in quantized_layers.items():
        if k.startswith(LM_P):
            rest = k[len(LM_P):]
            for kk in (k, f"language_model.model.{rest}", f"model.{rest}"):
                expanded[kk] = dict(v)
        elif k.startswith("mtp."):
            rest = k[len("mtp."):]
            for kk in (k, f"mtp.model.{rest}", f"model.mtp.{rest}", rest):
                expanded[kk] = dict(v)
        else:
            expanded[k] = dict(v)
    for k in [k for k in expanded if re.match(rf"^layers\.{mtp_idx}\.", k)]:
        for pre in (LM_P, "language_model.model.", "model."):
            expanded[pre + k] = dict(expanded[k])
    quantized_layers = expanded
    print(f"quantized_layers after namespace/MTP expansion: {len(quantized_layers)}")

    qc = cfg.get("quantization_config") or {}
    if qc.get("ignore"):
        before = len(qc["ignore"])
        qc["ignore"] = [e for e in qc["ignore"] if quantized_layers.get(e, {}).get("quant_algo") != ALGO]
        print(f"ignore list: {before} -> {len(qc['ignore'])} (our dense modules removed)")

    qc.update({"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
               "group_size": 16, "quantized_layers": quantized_layers})
    cfg["quantization_config"] = qc
    os.unlink(f"{DST}/config.json")
    json.dump(cfg, open(f"{DST}/config.json", "w"), indent=1)

    os.system(f"chmod -R ug+rwX,o+rX {DST}")
    try:
        st = os.stat("/models")
        for root, dirs, files in os.walk(DST):
            for n in dirs + files:
                os.chown(os.path.join(root, n), st.st_uid, st.st_gid)
        os.chown(DST, st.st_uid, st.st_gid)
    except OSError:
        pass
    print(f"HIBRID45 checkpoint ready: {DST}")


if __name__ == "__main__":
    sys.exit(main())
