# syntax=docker/dockerfile:1.6
# ------------------------------------------------------------------------------------------------------
# mbx-qwen38-flash-next-cluster — the PUBLISHED qwen38-flash-next image + three patches (hibrid47 stack).
#
# BASE = docker.io/myllmbox/qwen38-flash-next-vllm:v1 (our own release, 2026-09-02; reproducible from
# builds/qwen38-flash-next/ — vendor SM121 vLLM base + the ple_layer patch set). Pinned by digest.
#
# docker/patches/ (anchor-asserted, each refuses to apply twice):
#   01  multi-node PLE offload gate (MBX_PLE_MULTINODE) — the int3/CPU-offload path, kept as the FALLBACK for
#       hibrid46-style checkpoints; inert unless the env is set.
#   02  loader page-cache hygiene: posix_fadvise(DONTNEED) per shard after it is consumed (kit livelocks 2026-09-05).
#   03  GPU-resident NVFP4 PLE table: when the checkpoint declares ple_quantization.format=nvfp4 (hibrid47, made by
#       docker/make-hibrid47.py — shipped in the image at /opt/mbx/) and VLLM_PLE_CPU_OFFLOAD is off, the table is
#       an ordinary resident GPU parameter, row-sharded over TP (or a full copy per rank with MBX_PLE_REPLICATE=1),
#       gathered + dequantized in-forward inside the CUDA graphs. No CPU worker, no IPC. +8–11 % engine steps.
#
# `./run.sh qwen38-flash-next-cluster` builds this → mbx-qwen38-flash-next-cluster and ships it to box2.
# Public twin: builds/qwen38-flash-next/cluster/Dockerfile → myllmbox/qwen38-flash-next-cluster-vllm:v3.
# ------------------------------------------------------------------------------------------------------
FROM myllmbox/qwen38-flash-next-vllm:v1@sha256:92ccd7de6d80b9597844d7a5bffb9ded4067dd323129e18bc24192c00dac09ad
LABEL org.opencontainers.image.source="https://github.com/bilikaz/myllmbox" \
      mbx.base="myllmbox/qwen38-flash-next-vllm:v1@sha256:92ccd7de" \
      mbx.patches="ple-offload-multinode loader-drop-page-cache ple-gpu-nvfp4"

COPY docker/patches/ /opt/mbx/patches/
COPY docker/make-hibrid47.py /opt/mbx/make-hibrid47.py
RUN set -e; for p in /opt/mbx/patches/0*.py; do echo "== $p"; python3 "$p"; done

# Sanity: gates present, modules import, the NVFP4 table method round-trips a synthetic row against the converter.
RUN python3 - <<'PY'
import pathlib, importlib, importlib.util, torch, vllm
root = pathlib.Path(vllm.__file__).parent
for f, n in {"v1/worker/gpu_worker.py": 2, "v1/ple_offload/worker.py": 1, "v1/ple_offload/connector.py": 2}.items():
    assert (root / f).read_text().count("MBX_PLE_MULTINODE") >= n, f
assert "_mbx_fadvise_dontneed" in (root / "model_executor/model_loader/weight_utils.py").read_text()
src = (root / "models/qwen3_8_flash_next/nvidia/ple_layer.py").read_text()
for needle in ("_MbxNvfp4EmbeddingMethod", "_mbx_nvfp4_gpu", "_mbx_nvfp4_load", "_mbx_ple_replicate", '"format", "block", "rows", "dim"'):
    assert needle in src, needle
pl = importlib.import_module("vllm.models.qwen3_8_flash_next.nvidia.ple_layer")
spec = importlib.util.spec_from_file_location("mk", "/opt/mbx/make-hibrid47.py"); mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
torch.manual_seed(0); x = torch.randn(64, 160) * 0.05; g = x.abs().max().item() / (6 * 448)
packed, scales = mk.quantize(x, g)
class L: pass
lay = L(); lay.nvfp4_packed = packed; lay.nvfp4_scales = scales; lay.nvfp4_global = torch.tensor(g); lay.nvfp4_lut = torch.tensor(pl._MbxNvfp4EmbeddingMethod._LUT); lay.params_dtype = torch.float32
y = pl._MbxNvfp4EmbeddingMethod().embedding(lay, torch.arange(64)); rel = ((y - x).norm() / x.norm()).item()
assert rel < 0.12, rel
y2 = pl._MbxNvfp4EmbeddingMethod().embedding(lay, torch.arange(64).view(4, 16)); assert tuple(y2.shape) == (4, 16, 160)
import vllm.v1.ple_offload.worker, vllm.v1.ple_offload.connector
print(f"mbx-qwen38-flash-next-cluster: patches 01-03 present; nvfp4 round-trip rel err {rel:.3f}; vllm {vllm.__version__}")
PY
