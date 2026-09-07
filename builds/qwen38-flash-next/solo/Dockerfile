# syntax=docker/dockerfile:1.6
# ------------------------------------------------------------------------------------------------------
# myllmbox/qwen38-flash-next-vllm:v2 — the v1 serving image + six patches: the hibrid47 NVFP4 table DEMAND-PAGED from NVMe + fp8 KV (single Spark).
# Public build of recipes/qwen38-flash-next/Dockerfile (byte-identical patches); ledger: ../vllm-v2.md.
# Build:  docker build -t myllmbox/qwen38-flash-next-vllm:v2 builds/qwen38-flash-next/solo/
#
# BASE = docker.io/myllmbox/qwen38-flash-next-vllm:v1 (our release; pinned by digest).
# Patches (docker/patches/, anchor-asserted, each refuses to apply twice):
#   01  multi-node PLE offload gate (MBX_PLE_MULTINODE) — kept as the FALLBACK path; inert unless the env is set.
#   02  GPU-resident NVFP4 PLE table: when the checkpoint declares ple_quantization.format=nvfp4 (made by
#       docker/make-ple-nvfp4.py) and VLLM_PLE_CPU_OFFLOAD is off, the n-gram table lives on the GPU, row-sharded
#       over TP, gathered + dequantized in-process — no CPU worker, no IPC, no per-step detour.
#   03  loader drops each shard from the page cache after it is loaded (UMA: the GPU driver wants FREE pages).
#   04  MBX_PLE_MMAP=1: the table is NOT allocated — the 8 ple-nvfp4 shard files are mmapped read-only and the GPU
#       gathers straight out of the mapping (GB10 ATS; DLPack capsule over the host pointer). Boot needs 72 GiB, not
#       99; the table fills the page cache on demand (evictable). MBX_PLE_MMAP_PREWARM=0|1|<seconds>, trigger file
#       /cache/mbx-ple-prewarm = populate the whole table later. See builds/qwen38-flash-next/ple-mmap-probe/.
#   05  dual-path gather: vllm::mbx_ple_gather is an eager SPLITTING op; MBX_PLE_MMAP_MODE=auto reads rows with O_DIRECT
#       preads (no page cache, no GPU faults) during boot and flips to the mmap/ATS gather when warm-up + capture finish.
#       Flags: /cache/mbx-ple-mmap, /cache/mbx-ple-nvme; PREWARM=auto populates right after the flip.
#   06  fp8_e4m3 / nvfp4 KV cache on the QSA path = upstream PR #54846 ported (docker/overlays/, whole-file overlays;
#       docker/tests/test_qsa_kv_quant.py = the PR's tests, run on the GPU after the build). --kv-cache-dtype fp8.
#
# Consumed by solo/qwen38-flash-next-recipe v2 (the single-Spark kit) and, as mbx-qwen38-flash-next, by the recipe.
# ------------------------------------------------------------------------------------------------------
FROM myllmbox/qwen38-flash-next-vllm:v1@sha256:92ccd7de6d80b9597844d7a5bffb9ded4067dd323129e18bc24192c00dac09ad
LABEL org.opencontainers.image.source="https://github.com/bilikaz/myllmbox" \
      mbx.base="myllmbox/qwen38-flash-next-vllm:v1@sha256:92ccd7de" \
      mbx.patches="ple-offload-multinode ple-gpu-nvfp4 loader-drop-page-cache ple-mmap ple-mmap-dual qsa-fp8-nvfp4-kv"

COPY docker/patches/ /opt/mbx/patches/
COPY docker/overlays/ /opt/mbx/overlays/
COPY docker/tests/ /opt/mbx/tests/
COPY docker/make-ple-nvfp4.py /opt/mbx/make-ple-nvfp4.py
RUN set -e; for p in /opt/mbx/patches/0*.py; do echo "== $p"; python3 "$p"; done

# Sanity: gates present, module imports, the NVFP4 method round-trips a synthetic row on CPU.
RUN python3 - <<'PY'
import pathlib, torch, vllm
root = pathlib.Path(vllm.__file__).parent
src = (root / "models/qwen3_8_flash_next/nvidia/ple_layer.py").read_text()
for needle in ("_MbxNvfp4EmbeddingMethod", "_mbx_nvfp4_gpu", "_mbx_nvfp4_load", '"format", "block", "rows", "dim"',
               "_mbx_ple_mmap", "_mbx_mmap_attach", "_mbx_mmap_gather", "_MbxDLPack", "mbx_ple_gather", "_mbx_nvme_gather"):
    assert needle in src, needle
assert '"vllm::mbx_ple_gather"' in (root / "config/compilation.py").read_text(), "splitting op not registered"
qsa = (root / "models/qwen3_8_flash_next/nvidia/qsa.py").read_text(); ops = (root / "models/qwen3_8_flash_next/nvidia/ops/qsa.py").read_text()
for needle in ("_QSA_KV_CACHE_DTYPES", "kv_cache_fp8", "do_kv_cache_update"): assert needle in qsa, needle
for needle in ("KV_QUANT_FP8", "KV_QUANT_NVFP4", "def _qsa_mqa_paged_kernel(", "def qsa_select_paged_tokens("): assert needle in ops, needle
import py_compile; [py_compile.compile(str(root / f), doraise=True) for f in ("models/qwen3_8_flash_next/nvidia/qsa.py", "models/qwen3_8_flash_next/nvidia/ops/qsa.py", "platforms/interface.py")]
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
# the F.embedding contract: a [tokens, 16] index must come back as [tokens, 16, D] (VocabParallelEmbedding masks it so)
y2 = pl._MbxNvfp4EmbeddingMethod().embedding(lay, torch.arange(64).view(4, 16))
assert tuple(y2.shape) == (4, 16, 160), f"embedding must keep the input shape, got {tuple(y2.shape)}"
assert torch.equal(y2.reshape(64, 160), y), "reshaped gather differs from flat gather"
# demand-paged table plumbing, CPU-side (no GPU at build time): a synthetic 2-shard checkpoint → header parse →
# DLPack capsule over the mmap → tensors must equal the file bytes; the per-shard gather must pick the right rows.
import json, os, struct, tempfile, numpy as np
from safetensors.numpy import save_file
S, D = 64, 160
tmp = tempfile.mkdtemp(); wm = {}
rng = np.random.default_rng(0); ref_p, ref_s = [], []
for k in range(2):
    pk = rng.integers(0, 256, (S, D // 2), dtype=np.uint8); sc = rng.integers(0, 256, (S, D // 16), dtype=np.uint8)
    ref_p.append(pk); ref_s.append(sc)
    names = {f"model.x.ngram_embedding.nvfp4_shard_{k}.packed": pk, f"model.x.ngram_embedding.nvfp4_shard_{k}.scales": sc}
    save_file(names, f"{tmp}/ple-{k}.safetensors"); wm.update({n: f"ple-{k}.safetensors" for n in names})
json.dump({"weight_map": wm}, open(f"{tmp}/model.safetensors.index.json", "w"))
cfg = {"format": "nvfp4", "block": 16, "rows": 2 * S, "dim": D, "shards": 2, "shard_rows": S}
packed, scales, keep, meta = pl._mbx_mmap_table(tmp, cfg, cuda=False, want_meta=True)
assert len(meta) == 4 and all(len(v) == 4 for v in meta.values())
for k in range(2):
    assert packed[k].dtype == torch.uint8 and tuple(packed[k].shape) == (S, D // 2)
    assert torch.equal(packed[k], torch.from_numpy(ref_p[k])), f"shard {k} packed bytes differ through DLPack"
    assert torch.equal(scales[k].view(torch.uint8), torch.from_numpy(ref_s[k])), f"shard {k} scales differ"
class M: pass
lay = M(); lay._mbx_mm_packed = packed; lay._mbx_mm_scales = scales; lay._mbx_mm_S = S; lay._mbx_mm_n = 2
ids = torch.tensor([0, 5, S - 1, S, S + 7, 2 * S - 1])
gp, gs = pl._mbx_mmap_gather(lay, ids)
want = torch.from_numpy(np.concatenate(ref_p))[ids]
assert torch.equal(gp, want), "per-shard gather picked wrong rows"
assert torch.equal(gs.view(torch.uint8), torch.from_numpy(np.concatenate(ref_s))[ids]), "per-shard scale gather wrong"
# the nvme path (O_DIRECT, buffered fallback on overlayfs) must return the same rows; then the eager op in both modes
lay._mbx_mm_meta = meta; lay._mbx_mm_cols_p = D // 2; lay._mbx_mm_cols_s = D // 16; lay._mbx_name = "t"; lay._mbx_mm_keep = keep
npk, nsc = pl._mbx_nvme_gather(lay, ids)
assert torch.equal(npk, want), "nvme gather packed rows differ"
assert torch.equal(nsc.view(torch.uint8), gs.view(torch.uint8)), "nvme gather scales differ"
assert hasattr(torch.ops.vllm, "mbx_ple_gather"), "custom op not registered"
pl._MBX_PLE_LAYERS["t"] = lay
for mode in ("nvme", "mmap"):
    pl._MBX_PLE_STATE["mode"] = mode
    po = torch.empty((len(ids), D // 2), dtype=torch.uint8); so = torch.empty((len(ids), D // 16), dtype=torch.float8_e4m3fn)
    torch.ops.vllm.mbx_ple_gather(ids, po, so, "t")
    assert torch.equal(po, want) and torch.equal(so.view(torch.uint8), gs.view(torch.uint8)), f"op wrong in {mode} mode"
print(f"mbx-qwen38-flash-next: patches present; nvfp4 round-trip rel err {rel:.3f}; mmap+DLPack CPU path OK; nvme path + eager op OK in both modes; vllm {vllm.__version__}")
PY
