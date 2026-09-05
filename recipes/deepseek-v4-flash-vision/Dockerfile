# syntax=docker/dockerfile:1.6
# ------------------------------------------------------------------------------------------------------
# mbx-deepseek-v4-flash-vision — DeepSeek-V4-Flash-Vision-Exp on 2× DGX Spark (GB10 / SM121).
#
# BASE = the official per-model vLLM image (arm64, CUDA 13.0, built 2026-09-01 from the #54566 branch at
# commit 5ab628dd1 = vLLM 0.28.1rc1.dev137; FlashInfer 0.6.18 python+cubin+jit-cache; torch 2.13.0+cu130;
# triton 3.7.1; nccl 2.30.7; tilelang 0.1.12; TORCH_CUDA_ARCH_LIST includes 12.0 + 12.1).
# Pinned by digest (tag = arm64 manifest of vllm/vllm-openai:deepseekv4-flash-vision).
#
# On top: FIVE small, anchor-asserted source patches (docker/patches/*.py — each refuses to apply twice and
# fails the build loudly if upstream moved the anchor). Provenance of every change:
#   01  vLLM PR #54631 (lucamotz, open)  — streaming Vision weight loader (drop the sorted() that
#       materialises the whole 157G checkpoint in host RAM before TP sharding) + DSpark n_predict taken
#       from the checkpoint's dspark_block_size=5 (config says num_nextn_predict_layers=3 → wrong depth).
#       Without it: OOM/ dies on first decode.
#   02  vLLM PR #54815 (merged 2026-09-02) — sliding-window layers use PLAIN RoPE (no YaRN); YaRN only
#       on compressor layers. Correctness fix the image predates.
#   03  vLLM PR #52724 (merged 2026-09-01) — FlashInfer DSV4 sparse-MLA / SWA metadata builders declare
#       AttentionCGSupport.ALWAYS → DSpark adaptive verification is allowed on this backend.
#   04  SM121 adaptive-verification enablers (from r0b0tlab/dsv4-flash-vision-exp-vllm-sm121, reviewed
#       + re-expressed as minimal diffs): indexer flattened-query gate extended to family 120; C128A
#       decode indices kept contiguous; squeeze/contiguous the index tensors before the SM120 TVM call
#       ("eidx must be contiguous" boot crash once adaptive verification runs).
#   05  FlashInfer csrc: dual-cache sparse-MLA PREFILL dispatch for index_topk=512 (stock 0.6.18 only
#       instantiates dual-cache TOPK=128 → "Unsupported sparse-MLA prefill configuration" on the first
#       image request). Same fix as FlashInfer PR #4850 (superseded upstream by the #4802 refactor, which
#       is NOT in 0.6.18); r0b0tlab's FP8-compute variant, validated on 2×GB10. JIT-compiled at first
#       boot (~20 s) into /cache/flashinfer-workspace. ALSO evicts the image's PREBUILT AOT copy of this
#       module (flashinfer_jit_cache/jit_cache/sparse_mla_sm120) — FlashInfer loads an existing AOT .so
#       in preference to JIT source, which silently made the patch dead code on our first boot.
#
# `./run.sh deepseek-v4-flash-vision` builds this → mbx-deepseek-v4-flash-vision (and ships it to box2).
# ------------------------------------------------------------------------------------------------------
FROM vllm/vllm-openai:deepseekv4-flash-vision-arm64-cu130@sha256:8568b4bbc821903d93a0a9c17dd80382fdc0ba78eaa128e3eb5cb71c3bf06b79
LABEL org.opencontainers.image.source="https://github.com/bilikaz/myllmbox" \
      mbx.base="vllm/vllm-openai:deepseekv4-flash-vision-arm64-cu130@sha256:8568b4bb" \
      mbx.patches="pr54631 pr54815 pr52724 sm121-adaptive fi-dualcache-topk512"

COPY docker/patches/ /opt/mbx/patches/
RUN set -e; for p in /opt/mbx/patches/0*.py; do echo "== $p"; python3 "$p"; done

# Sanity: the patched modules import, and every marker is in place.
RUN python3 - <<'PY'
import pathlib, vllm, flashinfer
root = pathlib.Path(vllm.__file__).parent
fi = pathlib.Path(flashinfer.__file__).parent
want = {
    root / "models/deepseek_v4/nvidia/vl_model.py": "MBX: PR #54631",
    root / "config/speculative.py": "_normalize_deepseek_v4_dspark_hf_config",
    root / "models/deepseek_v4/common/rope.py": "Sliding-window layers use plain RoPE",
    root / "models/deepseek_v4/nvidia/flashinfer_sparse.py": "DeepseekSparseSWAFlashInferBackend",
    root / "v1/attention/backends/mla/indexer.py": "is_device_capability_family(120)",
    root / "models/deepseek_v4/sparse_mla.py": "MBX: SM121 adaptive",
    fi / "data/csrc/sparse_mla_sm120_prefill.cu": "DISPATCH_BY_NH_PBSX_512",
}
import importlib.util
aot = pathlib.Path(importlib.util.find_spec("flashinfer_jit_cache").origin).parent / "jit_cache/sparse_mla_sm120"
assert not aot.exists(), f"prebuilt AOT module still present: {aot}"
for p, needle in want.items():
    assert needle in p.read_text(), (p, needle)
import vllm.models.deepseek_v4.nvidia.flashinfer_sparse, vllm.models.deepseek_v4.common.rope, vllm.config.speculative
print("mbx-deepseek-v4-flash-vision: all patches present; vllm", vllm.__version__)
PY
