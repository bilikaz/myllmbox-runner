# syntax=docker/dockerfile:1.6
# ------------------------------------------------------------------------------------------------------
# mbx-qwen38-flash-next-cluster-test — the v1 serving image + GPU-resident NVFP4 PLE table (the lab lane).
#
# BASE = docker.io/myllmbox/qwen38-flash-next-vllm:v1 (our release; pinned by digest).
# Patches (docker/patches/, anchor-asserted, each refuses to apply twice):
#   01  multi-node PLE offload gate (MBX_PLE_MULTINODE) — kept as the FALLBACK path; inert unless the env is set.
#   02  GPU-resident NVFP4 PLE table: when the checkpoint declares ple_quantization.format=nvfp4 (made by
#       docker/make-ple-nvfp4.py) and VLLM_PLE_CPU_OFFLOAD is off, the n-gram table lives on the GPU, row-sharded
#       over TP, gathered + dequantized in-process — no CPU worker, no IPC, no per-step detour.
#
# `./run.sh qwen38-flash-next-cluster-test` builds this → mbx-qwen38-flash-next-cluster-test (ships to box2).
# ------------------------------------------------------------------------------------------------------
FROM myllmbox/qwen38-flash-next-vllm:v1@sha256:92ccd7de6d80b9597844d7a5bffb9ded4067dd323129e18bc24192c00dac09ad
LABEL org.opencontainers.image.source="https://github.com/bilikaz/myllmbox" \
      mbx.base="myllmbox/qwen38-flash-next-vllm:v1@sha256:92ccd7de" \
      mbx.patches="ple-offload-multinode ple-gpu-nvfp4"

COPY docker/patches/ /opt/mbx/patches/
COPY docker/make-ple-nvfp4.py /opt/mbx/make-ple-nvfp4.py
RUN set -e; for p in /opt/mbx/patches/0*.py; do echo "== $p"; python3 "$p"; done

# Sanity: gates present, module imports, the NVFP4 method round-trips a synthetic row on CPU.
RUN python3 - <<'PY'
import pathlib, torch, vllm
root = pathlib.Path(vllm.__file__).parent
src = (root / "models/qwen3_8_flash_next/nvidia/ple_layer.py").read_text()
for needle in ("_MbxNvfp4EmbeddingMethod", "_mbx_nvfp4_gpu", "_mbx_nvfp4_load", '"format", "block", "rows", "dim"'):
    assert needle in src, needle
assert (root / "v1/ple_offload/worker.py").read_text().count("MBX_PLE_MULTINODE") >= 1
import importlib
pl = importlib.import_module("vllm.models.qwen3_8_flash_next.nvidia.ple_layer")
# encode with the converter's math, decode with the method's math → must agree within the fp4 grid
import sys; sys.path.insert(0, "/opt/mbx"); import importlib.util
spec = importlib.util.spec_from_file_location("mk", "/opt/mbx/make-ple-nvfp4.py"); mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
torch.manual_seed(0); x = torch.randn(64, 160) * 0.05
g = x.abs().max().item() / (6 * 448)
packed, scales = mk.quantize(x, g)
class L: pass
lay = L(); lay.nvfp4_packed = packed; lay.nvfp4_scales = scales; lay.nvfp4_global = torch.tensor(g); lay.nvfp4_lut = torch.tensor(pl._MbxNvfp4EmbeddingMethod._LUT); lay.params_dtype = torch.float32
y = pl._MbxNvfp4EmbeddingMethod().embedding(lay, torch.arange(64))
rel = ((y - x).norm() / x.norm()).item()
assert rel < 0.12, f"nvfp4 round-trip error too high: {rel}"
print(f"mbx-qwen38-flash-next-cluster-test: patches present; nvfp4 encode/decode round-trip rel err {rel:.3f}; vllm {vllm.__version__}")
PY
