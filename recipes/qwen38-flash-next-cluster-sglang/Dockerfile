# mbx-qwen38-flash-next-cluster-sglang — DUAL SGLANG LANE (same patch set as solo-sglang; own folder per house rule) (2026-08-29): the stock qwen38flashnext sglang
# image + the FULL DSpark SM121 kernel set from MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks (MIT,
# measured 64.4 tok/s single-stream TP=2 on 2x Spark) + our MBX_PLE_MMAP NVMe-table patch.
#
# Why the v1 lane (retired 2026-08-27) lost and what changed:
#   v1: QSA fell back to our hand-carried Triton kernel, spec decode was broken (fp8 head accept 0.00 /
#       nvfp4 draft head unloadable) -> 11-12 tok/s dual, solo never measured. Retired.
#   v2: RadixArk checkpoint has a BF16 MTP head -> NEXTN spec WORKS (Mia validated, CUDA-graphed);
#       sglang#36806+#36845 kernel work is now carried as her production patch set; her NVFP4 KV cache
#       patch gives ~3.1x KV tokens on the QSA layers (NIAH-verified through 128k).
#
# The four patch layers (order matters):
#   1. sglang#36845: SM121 packed one-query Triton varlen kernel for QSA decode (FA4 CuTe does not
#      compile on GB10 — MLIR "weakly congruent"; the kernel reads cu_seqlens on-device so CUDA-graph
#      replay stays valid; fp8 K/V upcast allowed in-kernel).
#   2. sglang#36806: force _resolve_trtllm_sparse_decode -> None on SM121 — that path silently emits
#      token id 0 at long context (120k-210k) on GB10 while returning HTTP 200.
#   3. NVFP4 KV cache for QSA layers (--kv-cache-dtype nvfp4): pool stores packed FP4 + per-block FP8
#      scales, gather paths compact + dequant via flashinfer's nvfp4_kv kernels (CUDA-graph safe).
#   4. MBX_PLE_MMAP=1 (ours, from the v1 lane): sglang's PLE offload PINS the table in host RAM — on
#      GB10 UMA that is the same pool as the GPU, so solo can NOT fit RadixArk (84G weights + 52G
#      pinned = 136G > 119G). This patch makes the offload wrapper adopt the checkpoint's own
#      mmap-backed CPU tensor (zero-copy view of the safetensors on NVMe — RadixArk's table is fp8 AT
#      REST, dtype matches, so adoption short-circuits: no copy, no bank, rows page in on first gather
#      and stay evictable). Fallback alloc path backs the wrapper buffer with an unlinked NVMe file.
#
# digest of lmsysorg/sglang:qwen38flashnext pinned below (arm64, PR #36497 baked, SM121-verified base)
FROM lmsysorg/sglang:qwen38flashnext@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1

COPY docker/sm121_varlen.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa/sm121_varlen.py
COPY docker/qsa_nvfp4_kv.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa_nvfp4_kv.py
COPY docker/apply_nvfp4_patches.py /tmp/apply_nvfp4_patches.py

# ---- patches 1+2: QSA backend routing (Mia's splices, verbatim logic) ----
RUN python3 - <<'PYEOF'
p = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
s = open(p).read()

# sglang#36845: SM121 packed-varlen Triton fallback (not FA4 CuTe).
if "qsa.sm121_varlen" not in s:
    anchor = "    try:\n        from flash_attn import flash_attn_varlen_func"
    assert anchor in s, "flash_attn varlen anchor not found — upstream image layout changed"
    insert = (
        "    from sglang.srt.utils import is_sm121\n"
        "\n"
        "    if is_sm121():\n"
        "        from sglang.srt.layers.attention.qsa.sm121_varlen import (\n"
        "            qsa_sm121_varlen_attention,\n"
        "        )\n"
        "\n"
        "        return qsa_sm121_varlen_attention\n"
    )
    s = s.replace(anchor, insert + anchor, 1)

# sglang#36806: never take FlashInfer TRT-LLM sparse decode on SM121 (silent token-0 at long context).
marker = "dspark: SM121 must not use TRT-LLM sparse decode"
if marker not in s:
    fn = "def _resolve_trtllm_sparse_decode():"
    i = s.find(fn)
    assert i >= 0, "trtllm resolver not found — upstream image layout changed"
    ds = s.find('"""', i)
    assert ds > 0, "trtllm resolver docstring not found"
    ds_end = s.find('"""', ds + 3)
    assert ds_end > 0, "trtllm resolver docstring unterminated"
    ds_end += 3
    s = s[:ds_end] + (
        "\n    from sglang.srt.utils import is_sm121\n"
        "\n"
        "    # dspark: SM121 must not use TRT-LLM sparse decode\n"
        "    # (sglang#36806 / #36845). That path silently emits token id 0\n"
        "    # at long context on GB10.\n"
        "    if is_sm121():\n"
        "        return None\n"
    ) + s[ds_end:]

open(p, "w").write(s)
print("qwen_sparse_attn_backend.py patched for SM121 (sglang#36806 + #36845)")
PYEOF

# ---- patch 3: NVFP4 KV cache wiring (Mia's apply script; asserts on layout drift) ----
RUN python3 /tmp/apply_nvfp4_patches.py && rm -f /tmp/apply_nvfp4_patches.py

# ---- patch 4: MBX_PLE_MMAP (ours, v1 lane verbatim) — PLE table from NVMe, not pinned RAM ----
RUN python3 - <<'PYEOF'
p = "/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py"
s = open(p).read()
anchor = "    def allocate_output(\n        self, shape: Tuple[int, ...], device: torch.device\n    ) -> torch.Tensor:"
method = '''    def weight_loader(self, param, loaded_weight):
        import os as _os, logging as _logging
        _log = _logging.getLogger(__name__)
        if (
            _os.environ.get("MBX_PLE_MMAP", "0") == "1"
            and loaded_weight.device.type == "cpu"
            and not self.use_presharded_weights
            and getattr(param, "packed_dim", None) is None
            and loaded_weight.dim() == 2
            and loaded_weight.dtype == param.dtype
        ):
            _start = self.shard_indices.org_vocab_start_index
            _end = self.shard_indices.org_vocab_end_index
            if 0 <= _start < _end <= loaded_weight.shape[0]:
                _shard = loaded_weight.narrow(0, _start, _end - _start)
                _new = torch.nn.Parameter(_shard, requires_grad=False)
                for _n, _v in vars(param).items():
                    setattr(_new, _n, _v)
                _new.weight_loader = self.weight_loader
                self._parameters["weight"] = _new
                _log.info(
                    "MBX_PLE_MMAP: adopted mmap-backed PLE shard rows [%d:%d] %s %s - no RAM copy",
                    _start, _end, tuple(_shard.shape), _shard.dtype,
                )
                return
            _log.warning("MBX_PLE_MMAP: unexpected PLE weight layout, falling back to pinned copy")
        return super().weight_loader(param, loaded_weight)

'''
assert anchor in s, "allocate_output anchor not found — upstream layout changed"
assert s.count(anchor) == 1, "anchor not unique"
assert "MBX_PLE_MMAP" not in s, "already patched"
s = s.replace(anchor, method + anchor, 1)
alloc_anchor = '''        cpu_weight = nn.Parameter(
            torch.empty(
                source_weight.shape,
                dtype=source_weight.dtype,
                device="cpu",
                pin_memory=True,
            ),
            requires_grad=False,
        )'''
alloc_new = '''        import os as _os
        if _os.environ.get("MBX_PLE_MMAP", "0") == "1":
            import tempfile as _tf
            _dir = _os.environ.get("MBX_PLE_MMAP_DIR", "/cache/ple-mmap")
            _os.makedirs(_dir, exist_ok=True)
            _f = _tf.NamedTemporaryFile(dir=_dir, delete=False); _path = _f.name; _f.close()
            _numel = 1
            for _s in source_weight.shape:
                _numel *= int(_s)
            _t = torch.from_file(
                _path, shared=True, size=_numel, dtype=source_weight.dtype
            ).view(tuple(source_weight.shape))
            _os.unlink(_path)  # mapping stays valid; blocks freed on process exit
            cpu_weight = nn.Parameter(_t, requires_grad=False)
        else:
            cpu_weight = nn.Parameter(
                torch.empty(
                    source_weight.shape,
                    dtype=source_weight.dtype,
                    device="cpu",
                    pin_memory=True,
                ),
                requires_grad=False,
            )'''
assert s.count(alloc_anchor) == 1, "cpu_weight alloc anchor not found/unique"
s = s.replace(alloc_anchor, alloc_new, 1)
open(p, "w").write(s)
print("qwen4_exp.py patched: PLE table served from NVMe mmap under MBX_PLE_MMAP=1")
PYEOF

# ---- patch 5: log hygiene — GENERATION DATA ONLY. The dashboard (sparkdash LlmProbe) polls
# /get_server_info + /get_model_info every ~2s and sglang answers each poll with TWO deprecation
# nags + an access line = 3 log lines / 2s of pure noise, drowning the decode metrics. Drop ALL
# successful http access lines + the deprecation nags; failures (4xx/5xx) still log, and the
# scheduler's decode-batch / spec-accept metric lines are untouched.
RUN python3 - <<'PYEOF'
p = "/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py"
s = open(p).read()
assert "MbxHttpNoiseFilter" not in s, "already patched"
s += '''

# mbx: generation-data-only logs (see recipe Dockerfile patch 5)
import logging as _mbx_logging

class _MbxHttpNoiseFilter(_mbx_logging.Filter):
    def filter(self, record):
        m = record.getMessage()
        if "is deprecated and will be removed" in m:
            return False
        return not (" 200 " in m or m.rstrip().endswith("200 OK"))

for _mbx_name in ("uvicorn.access", __name__):
    _mbx_logging.getLogger(_mbx_name).addFilter(_MbxHttpNoiseFilter())
'''
open(p, "w").write(s)
print("http_server.py: generation-data-only log filter installed")
PYEOF

# ---- patch 6: vLLM-style param tolerance (MBX_STRIP_SPEC_BREAKERS=1). sglang SILENTLY runs a
# request without speculative decoding when it carries fields the draft verifier can't check —
# logprobs/top_logprobs being the classic client-tool default (measured: dashboard/viewer requests
# decoded at 1 token/step while a bare curl drafted fine). vLLM just ignores what it can't honor;
# this middleware does the same: strip those fields at the door and LOG what was stripped (which
# also tells us exactly which field the client sent). Response degrades gracefully (no per-token
# logprobs) instead of silently halving throughput.
RUN python3 - <<'PYEOF'
p = "/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py"
s = open(p).read()
assert "MBX_STRIP_SPEC_BREAKERS" not in s, "already patched"
s += '''

# mbx: strip spec-breaking request fields (see recipe Dockerfile patch 6)
import json as _mbx_json
import os as _mbx_os
from starlette.requests import Request as _MbxRequest

_MBX_STRIP = _mbx_os.environ.get("MBX_STRIP_SPEC_BREAKERS", "0") == "1"
_MBX_SPEC_BREAKERS = ("logprobs", "top_logprobs")
_mbx_stripped_seen = set()

@app.middleware("http")
async def _mbx_strip_spec_breakers(request, call_next):
    if (
        _MBX_STRIP
        and request.method == "POST"
        and request.url.path in ("/v1/chat/completions", "/v1/completions")
    ):
        body = await request.body()
        try:
            data = _mbx_json.loads(body)
            hit = [k for k in _MBX_SPEC_BREAKERS if data.pop(k, None) not in (None, False, 0)]
            if hit:
                for k in hit:
                    if k not in _mbx_stripped_seen:
                        _mbx_stripped_seen.add(k)
                        _mbx_logging.getLogger(__name__).warning(
                            "MBX_STRIP_SPEC_BREAKERS: dropped %r from a request (it disables "
                            "speculative decoding); further drops of this field are silent", k)
                body = _mbx_json.dumps(data).encode()
                request.scope["headers"] = [
                    (n, v) for n, v in request.scope["headers"] if n != b"content-length"
                ] + [(b"content-length", str(len(body)).encode())]
        except Exception:
            pass
        _saved = {"type": "http.request", "body": body, "more_body": False}
        async def _receive():
            return _saved
        request = _MbxRequest(request.scope, _receive)
    return await call_next(request)
'''
open(p, "w").write(s)
print("http_server.py: spec-breaker stripping middleware installed (MBX_STRIP_SPEC_BREAKERS=1)")
PYEOF
