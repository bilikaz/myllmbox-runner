"""Build the HIBRID checkpoint: Inferact NVFP4 experts + OUR block-FP8 dense projections.

Nobody ships fp8 dense for this model (verified 2026-08-29: the official "FP8" checkpoint quantizes
ONLY experts; every dense projection is BF16 there AND in Inferact). This script converts the
decode-path dense linears to ModelOpt FP8_PB_WO format — per-128x128-block weight scales, DYNAMIC
per-token activation quant → deterministic weight-only conversion, ZERO calibration.

Output = models/myllmbox/Qwen3.8-Flash-Next-hibrid:
  · every original Inferact file HARDLINKED (no copy, no extra 100G)
  · one new shard with the fp8 weights + 4D weight_scale tensors ([out/128,1,in/128,1], ModelOpt shape)
  · rewritten index (fp8 names -> new shard; stale bf16 bytes in old shards are simply unreferenced)
  · config.json quantization_config -> quant_algo MIXED_PRECISION + quantized_layers map
    (FP8_PB_WO for our dense set, NVFP4 declared per experts-parent so vLLM's
    ModelOptMixedPrecisionConfig routes each layer natively; the serve image only needs the tiny
    FP8_PB_WO branch splice — see this recipe's Dockerfile).

Vendor-respect list: lm_head, embeddings, mHC mixers, norms, vision tower, MTP, PLE stay BF16
(Alibaba excluded the sensitive ones from their own fp8; we mirror that).
Layers whose shapes don't divide 128 are skipped (logged) — FP8_PB_WO requires it.
"""
import json, os, re, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/models/Inferact/Qwen3.8-Flash-Next-NVFP4"
DST = "/models/myllmbox/Qwen3.8-Flash-Next-hibrid"
BLOCK = 128
FP8_MAX = 448.0

# decode-path dense modules to convert (module name = tensor name minus ".weight")
# FULL decode-path set minus in_proj_a/b (48 rows — not 128-divisible; they fuse together into
# in_proj_ba, so keeping BOTH bf16 keeps that fused layer algo-consistent). vLLM fuses shards into
# MergedColumnParallelLinear modules (qkv_proj, gate_up_proj, in_proj_qkvz) — the mixed config resolves
# the FUSED runtime name, so quantized_layers must declare those names too (emitted below); the fused
# crash of 2026-08-29 was that resolution failing, not the scale loaders.
PATTERNS = [
    r"^model\.language_model\.layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
    r"^model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
    r"^model\.language_model\.layers\.\d+\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)$",
]
# fused-module name -> its checkpoint shards (must ALL be quantized for the entry to be emitted)
FUSED_GROUPS = {
    "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "mlp.shared_expert.gate_up_proj": ("mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"),
    "linear_attn.in_proj_qkvz": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
}

def main() -> None:
    idx_path = f"{SRC}/model.safetensors.index.json"
    index = json.load(open(idx_path))
    wmap = dict(index["weight_map"])

    targets = []
    for name in wmap:
        if not name.endswith(".weight"):
            continue
        base = name[: -len(".weight")]
        if any(re.match(p, base) for p in PATTERNS):
            targets.append(base)
    targets.sort()
    print(f"candidate dense modules: {len(targets)}")

    os.makedirs(DST, exist_ok=True)
    # hardlink every original file (index/config rewritten below overwrite their links)
    for f in os.listdir(SRC):
        s, d = f"{SRC}/{f}", f"{DST}/{f}"
        if os.path.isfile(s) and not os.path.exists(d):
            os.link(s, d)

    new_tensors: dict[str, torch.Tensor] = {}
    quantized_layers: dict[str, dict] = {}
    handles: dict[str, object] = {}
    skipped, done_bytes = [], 0
    for base in targets:
        shard = wmap[base + ".weight"]
        if shard not in handles:
            handles[shard] = safe_open(f"{SRC}/{shard}", framework="pt", device="cpu")
        w = handles[shard].get_tensor(base + ".weight")
        out_f, in_f = w.shape
        if out_f % BLOCK or in_f % BLOCK:
            skipped.append((base, tuple(w.shape)))
            continue
        wf = w.to(torch.float32)
        blocks = wf.reshape(out_f // BLOCK, BLOCK, in_f // BLOCK, BLOCK)
        amax = blocks.abs().amax(dim=(1, 3), keepdim=False).clamp(min=1e-12)   # [ob, ib]
        scale = amax / FP8_MAX
        q = (blocks / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
        q = q.reshape(out_f, in_f).to(torch.float8_e4m3fn)
        new_tensors[base + ".weight"] = q
        # ModelOpt FP8_PB_WO exports weight_scale as [out_blk, 1, in_blk, 1]
        new_tensors[base + ".weight_scale"] = scale[:, None, :, None].contiguous()
        quantized_layers[base] = {"quant_algo": "FP8_PB_WO"}
        done_bytes += q.numel()
        # roundtrip sanity on a sample of blocks
        rt = q.to(torch.float32).reshape(out_f // BLOCK, BLOCK, in_f // BLOCK, BLOCK) * scale[:, None, :, None]
        err = (rt - blocks).abs().max() / (amax.max() + 1e-12)
        assert err < 0.05, f"{base}: fp8 roundtrip rel err {err:.4f} too high"
    print(f"quantized: {len(quantized_layers)} modules, {done_bytes/2**30:.2f} GiB fp8; skipped: {skipped}")
    assert quantized_layers, "nothing quantized — pattern/shape mismatch?"

    shard_name = "model-hibrid-fp8.safetensors"
    save_file(new_tensors, f"{DST}/{shard_name}")
    for name in new_tensors:
        wmap[name] = shard_name
    index["weight_map"] = wmap
    os.unlink(f"{DST}/model.safetensors.index.json")
    json.dump(index, open(f"{DST}/model.safetensors.index.json", "w"))

    # declare the FUSED runtime module names too (rule-1 direct lookup in the mixed config)
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
                    # fused module full name = shards' parent + the FULL fused suffix
                    # (BUG HISTORY 2026-08-29: appending only the last component dropped
                    # "linear_attn."/"self_attn." -> resolution miss -> vLLM's getattr(self, name, SELF)
                    # fallback bound the LAYER as the param -> "'MergedColumnParallelLinear' has no
                    # attribute 'data'". Keys must be the exact runtime module path.)
                    key = f"{parent}.{fused_suffix}"
                    if key not in quantized_layers:
                        quantized_layers[key] = {"quant_algo": "FP8_PB_WO"}
                        fused_added += 1
    print(f"fused-module entries added: {fused_added}")

    # declare experts NVFP4 per parent so the mixed config routes RoutedExperts natively
    parents = sorted({n.split(".experts.")[0] for n in wmap if ".experts." in n})
    for p in parents:
        quantized_layers[f"{p}.experts.up_proj"] = {"quant_algo": "NVFP4", "group_size": 16}
    print(f"expert parents declared NVFP4: {len(parents)}")

    cfg = json.load(open(f"{SRC}/config.json"))

    # ---- boot-saga folds (2026-08-29; each was a live-config surgery before it lived here) ----
    # (a) MTP RENUMBERING: the checkpoint stores the MTP draft layer as mtp.layers.0.*, but vLLM
    #     renumbers it to layers.<num_hidden_layers> at runtime (mtp_start_layer_idx) — resolution
    #     asked for 'mtp.layers.48.mlp.experts' and found nothing -> w2_input_scale crash. Emit BOTH
    #     layer indices for every mtp key.
    mtp_idx = (cfg.get("text_config") or cfg)["num_hidden_layers"]
    for k in [k for k in quantized_layers if k.startswith("mtp.layers.0.")]:
        quantized_layers[k.replace("mtp.layers.0.", f"mtp.layers.{mtp_idx}.", 1)] = dict(quantized_layers[k])

    # (b) NAMESPACE TRIPLING: the mixed config's prefix-candidate translation only covers SOME of the
    #     checkpoint<->runtime prefix pairs (runtime modules resolve as language_model.model.* while the
    #     checkpoint says model.language_model.*; MTP resolves under several roots during load vs run).
    #     Emitting every variant makes rule-1 direct lookup hit regardless of which namespace asks.
    #     This reproduces EXACTLY the live-validated 1343-entry map of the first successful boot:
    #       {model.language_model., language_model.model., model.} x (300 fp8 + 96 fused + 49 experts)
    #       + {mtp., mtp.model., model.mtp., <bare>} x layers.{0,48} MTP experts.
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
    # the renumbered MTP layer ALSO appears inside the language-model stack at runtime (layers.48):
    for k in [k for k in expanded if re.match(rf"^layers\.{mtp_idx}\.", k)]:
        for pre in (LM_P, "language_model.model.", "model."):
            expanded[pre + k] = dict(expanded[k])
    quantized_layers = expanded
    print(f"quantized_layers after namespace/MTP expansion: {len(quantized_layers)}")

    qc = cfg.get("quantization_config") or {}

    # (c) IGNORE-LIST STRIP: Inferact's config carries an "ignore" list naming every module IT left
    #     unquantized — which includes all 300 of ours. vLLM's is_layer_excluded consults it BEFORE any
    #     quantized_layers lookup, so our fp8 modules silently loaded as UnquantizedLinearMethod (the
    #     'MergedColumnParallelLinear has no attribute data' crash). Drop exactly our modules from it.
    if qc.get("ignore"):
        before = len(qc["ignore"])
        qc["ignore"] = [e for e in qc["ignore"] if quantized_layers.get(e, {}).get("quant_algo") != "FP8_PB_WO"]
        print(f"ignore list: {before} -> {len(qc['ignore'])} (our fp8 modules removed)")

    qc.update({"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
               "group_size": 16, "quantized_layers": quantized_layers})
    cfg["quantization_config"] = qc
    os.unlink(f"{DST}/config.json")
    json.dump(cfg, open(f"{DST}/config.json", "w"), indent=1)

    os.system(f"chmod -R a+r {DST}")
    print(f"HIBRID checkpoint ready: {DST}")

if __name__ == "__main__":
    sys.exit(main())
