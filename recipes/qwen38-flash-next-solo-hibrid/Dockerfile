# mbx-qwen38-flash-next-solo-hibrid — solo-vllm content + FP8_PB_WO-in-mixed splice (hibrid ckpt).
# vLLM's VLLM_PLE_CPU_OFFLOAD runs the n-gram table in a separate CPU worker process, but keeps the table
# RESIDENT in host RAM (95.4G bf16 for Inferact) — on unified memory that's the same pool as the GPU, so a
# single Spark can't fit it. This patch (MBX_PLE_MMAP=1) backs the table with an unlinked file on /cache
# (NVMe): the checkpoint-shard copies write through to disk once and pages stay evictable — the same
# mechanism as our sglang image's v2 patch. Offload itself is single-node only upstream (nnodes=2 refused).
FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

RUN python3 - <<'PYEOF'
p = "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
s = open(p).read()
anchor = """        self.ngram_embedding = VocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
            quant_method=_get_ple_embedding_quant_method(
                quant_config, f"{prefix}.ngram_embedding"
            ),
        )"""
patch = anchor + """
        # MBX_PLE_MMAP: back the CPU-worker table with an NVMe file so shard copies write through to
        # disk and pages stay evictable — UMA boxes can't afford 95G resident (see Dockerfile).
        # With MBX_PLE_MMAP_TAG set, the file is PERSISTENT and reused across boots: first boot writes
        # the 95G once and marks it complete; later boots mmap it and SKIP the copy entirely (load in
        # seconds, no NVMe wear). Without a tag: throwaway unlinked file, rewritten each boot (safe default).
        # Reset a stale table with: rm /cache/ple-mmap/*  (e.g. after switching checkpoints under one tag).
        import os as _os
        if (
            _os.environ.get("MBX_PLE_MMAP", "0") == "1"
            and envs.VLLM_PLE_CPU_OFFLOAD
            and is_offload_process()
        ):
            _w = self.ngram_embedding.weight
            _dir = _os.environ.get("MBX_PLE_MMAP_DIR", "/cache/ple-mmap")
            _os.makedirs(_dir, exist_ok=True)
            _numel = 1
            for _s in _w.shape:
                _numel *= int(_s)
            _tag = _os.environ.get("MBX_PLE_MMAP_TAG", "")
            self._mbx_ple_prewarmed = False
            self._mbx_ple_marker = None
            if _tag:
                _safe_prefix = str(prefix).replace("/", "_").replace(".", "_")
                _p2 = _os.path.join(
                    _dir,
                    f"{_tag}-{_safe_prefix}-{'x'.join(str(int(_s)) for _s in _w.shape)}-{str(_w.dtype).split('.')[-1]}.bin",
                )
                _marker = _p2 + ".ok"
                _esz = _w.element_size() if hasattr(_w, "element_size") else torch.empty((), dtype=_w.dtype).element_size()
                _want = _numel * _esz
                if _os.path.exists(_marker) and _os.path.getsize(_p2) == _want:
                    self._mbx_ple_prewarmed = True
                    __import__("logging").getLogger("mbx.ple").info("MBX_PLE_MMAP: reusing persistent PLE table %s (copy will be SKIPPED)", _p2)
                else:
                    self._mbx_ple_marker = _marker
                    __import__("logging").getLogger("mbx.ple").info("MBX_PLE_MMAP: writing persistent PLE table %s (one-time)", _p2)
            else:
                import tempfile as _tf
                _f = _tf.NamedTemporaryFile(dir=_dir, delete=False); _p2 = _f.name; _f.close()
            _t = torch.from_file(
                _p2, shared=True, size=_numel, dtype=_w.dtype
            ).view(tuple(_w.shape))
            if not _tag:
                _os.unlink(_p2)
            _new = torch.nn.Parameter(_t, requires_grad=False)
            for _n, _v in vars(_w).items():
                setattr(_new, _n, _v)
            self.ngram_embedding.weight = _new"""
assert s.count(anchor) == 1, "ngram_embedding constructor anchor not found/unique — upstream changed"
assert "MBX_PLE_MMAP" not in s, "already patched"
s = s.replace(anchor, patch, 1)

# load_weights: on a prewarmed table SKIP the shard copies (identical bytes — pure NVMe wear); after a
# fresh table finishes its copies, sync and write the completion marker so the next boot can skip.
la = """                copy_ple_embedding_shard_(
                    embedding.weight.data,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                    tp_start=embedding.shard_indices.org_vocab_start_index,
                    tp_end=embedding.shard_indices.org_vocab_end_index,
                )
                loaded.add("ngram_embedding.weight")
                continue"""
lp = """                if getattr(self, "_mbx_ple_prewarmed", False):
                    loaded.add("ngram_embedding.weight")
                    continue
                copy_ple_embedding_shard_(
                    embedding.weight.data,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                    tp_start=embedding.shard_indices.org_vocab_start_index,
                    tp_end=embedding.shard_indices.org_vocab_end_index,
                )
                self._mbx_ple_shards_done = getattr(self, "_mbx_ple_shards_done", 0) + 1
                if (
                    getattr(self, "_mbx_ple_marker", None)
                    and self._mbx_ple_shards_done >= self.split_ngram_parts
                ):
                    import os as _os
                    _os.sync()
                    open(self._mbx_ple_marker, "w").write("ok")
                    # world-readable: the (non-root) runner may copy the bank (table_server).
                    # _p2 is an __init__ local — derive the table path from the stored marker.
                    _os.chmod(self._mbx_ple_marker[: -len(".ok")], 0o644)
                    _os.chmod(self._mbx_ple_marker, 0o644)
                    __import__("logging").getLogger("mbx.ple").info("MBX_PLE_MMAP: persistent PLE table complete (marker written)")
                loaded.add("ngram_embedding.weight")
                continue"""
assert s.count(la) == 1, "shard-copy anchor not found/unique — upstream changed"
s = s.replace(la, lp, 1)
open(p, "w").write(s)
print("ple_layer.py patched: PLE file-backed + write-once persistence under MBX_PLE_MMAP_TAG")
PYEOF

# ---- HIBRID: FP8_PB_WO branch in the MIXED_PRECISION dispatcher (2026-08-29) ----
# vLLM's ModelOptMixedPrecisionConfig already routes per-layer algos (its docstring names our exact
# case: "FP8 for dense layers and NVFP4 for MoE experts") — but its LinearBase branch only dispatches
# the per-tensor "FP8" flavor. Our hibrid checkpoint uses FP8_PB_WO (per-128x128-block weight-only,
# DYNAMIC per-token activations — the zero-calibration flavor). One anchored branch adds it.
RUN python3 - <<'PYEOF'
p = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py"
s = open(p).read()
a = """            if quant_algo == "NVFP4":
                return ModelOptNvFp4LinearMethod(self.nvfp4_config)
            if quant_algo == "W4A16_NVFP4":
                return ModelOptNvFp4W4A16LinearMethod(self.w4a16_nvfp4_config)"""
assert s.count(a) == 1, "mixed-precision LinearBase dispatch anchor not found/unique"
s = s.replace(a, """            if quant_algo == "NVFP4":
                return ModelOptNvFp4LinearMethod(self.nvfp4_config)
            if quant_algo == "W4A16_NVFP4":
                return ModelOptNvFp4W4A16LinearMethod(self.w4a16_nvfp4_config)
            if quant_algo == "FP8_PB_WO":
                # mbx hibrid: per-block weight-only fp8 dense (dynamic act quant, no calibration).
                # ONE shared config instance: every ModelOptFp8Config() construction logs the
                # "experimental format" warning — per-layer instantiation spammed ~400 rows at load.
                if not hasattr(self, "_mbx_pbwo_cfg"):
                    self._mbx_pbwo_cfg = ModelOptFp8Config(
                        "FP8_PB_WO", True, self.kv_cache_quant_method, []
                    )
                return ModelOptFp8PbWoLinearMethod(self._mbx_pbwo_cfg)""", 1)
open(p, "w").write(s)
print("modelopt.py: FP8_PB_WO wired into MIXED_PRECISION dispatch")
PYEOF

# ---- HIBRID A/B: b12x native SM121 dense block-fp8 GEMM (2026-08-29) ----
# b12x (local-inference-lab) ships a native SM120/121 CuTe dense block-FP8 GEMM in exactly our format
# (128x128 weight blocks, dynamic per-token-group activations). vLLM already carries the kernel class
# (B12xFp8BlockScaledMMKernel) but it sits BELOW CUTLASS in the priority list, so installing the pip
# package alone changes nothing — activation is explicit via MBX_FP8_LINEAR_KERNEL (patch below).
# MoE is untouched: vLLM auto-selection deliberately excludes the b12x MoE backend (opt-in flag only).
RUN pip install --no-cache-dir b12x==1.3.0

RUN python3 - <<'PYEOF'
# (1) MBX_FP8_LINEAR_KERNEL=b12x|cutlass|triton|marlin|deepgemm — force the block-fp8 dense kernel
#     from the recipe yaml for A/B runs. Unset = stock priority dispatch (CUTLASS on GB10 today).
#     A forced kernel that can't be implemented logs a note and falls back to normal dispatch.
p = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/__init__.py"
s = open(p).read()
a = """    if activation_quant_key.scale.group_shape.is_per_group():
        kernel_type = choose_scaled_mm_linear_kernel("""
assert s.count(a) == 1, "block-fp8 dispatch anchor not found/unique — upstream changed"
assert "MBX_FP8_LINEAR_KERNEL" not in s, "already patched"
patch = """    if activation_quant_key.scale.group_shape.is_per_group():
        # MBX: A/B the block-fp8 dense kernel from the recipe yaml (see Dockerfile).
        import os as _os
        _mbx = _os.environ.get("MBX_FP8_LINEAR_KERNEL", "").strip().lower()
        if force_kernel is None and _mbx:
            force_kernel = {
                "b12x": B12xFp8BlockScaledMMKernel,
                "cutlass": CutlassFp8BlockScaledMMKernel,
                "triton": TritonFp8BlockScaledMMKernel,
                "marlin": MarlinFP8ScaledMMLinearKernel,
                "deepgemm": DeepGemmFp8BlockScaledMMKernel,
            }.get(_mbx)
        kernel_type = choose_scaled_mm_linear_kernel("""
s = s.replace(a, patch, 1)
open(p, "w").write(s)
print("kernels/linear/__init__.py: MBX_FP8_LINEAR_KERNEL force hook installed")

# (2) Compile-cache guard: which GEMM op gets traced into compiled graphs depends on these envs, but
#     vLLM does NOT key its torch-compile cache on them — a cached graph kept launching DeepGEMM after
#     VLLM_USE_DEEP_GEMM=0 (2026-08-29, cost one crashed boot + a manual cache purge). Key the hash.
p = "/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py"
s = open(p).read()
a = "        vllm_factors.append(__version__)"
assert s.count(a) == 1, "compute_hash anchor not found/unique — upstream changed"
s = s.replace(a, a + """
        # MBX: kernel-selection envs change traced graphs — must be part of the compile-cache key.
        import os as _os_mbx
        vllm_factors.append(
            "mbx-kernel-envs:"
            + _os_mbx.environ.get("MBX_FP8_LINEAR_KERNEL", "")
            + ","
            + _os_mbx.environ.get("VLLM_USE_DEEP_GEMM", "1")
        )""", 1)
open(p, "w").write(s)
print("config/vllm.py: kernel-selection envs keyed into the compile cache")

# (3) torch.compile safety for the b12x kernel itself: upstream's _run_b12x_fp8_block_scaled_mm does
#     a lazy importlib call INSIDE the traced code path -> dynamo dies with "Attempted to call
#     function marked as skipped" the moment the first fp8 linear region compiles (boot crash,
#     2026-08-29). Wrap the mm in an opaque custom op (same pattern as mbx_vocab_gemv) and rebind
#     the module-level function — the class method resolves the global at call time.
p = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/scaled_mm/b12x_block.py"
s = open(p).read()
assert "mbx::b12x_block_fp8_mm" not in s, "already patched"
assert "def _run_b12x_fp8_block_scaled_mm(" in s, "b12x mm anchor not found — upstream changed"
s += '''

# mbx: compile-opaque wrapper (see hibrid Dockerfile) — rebinds _run_b12x_fp8_block_scaled_mm
import torch as _mbx_torch

@_mbx_torch.library.custom_op("mbx::b12x_block_fp8_mm", mutates_args=())
def _mbx_b12x_mm(
    a: _mbx_torch.Tensor,
    weight: _mbx_torch.Tensor,
    a_scale: _mbx_torch.Tensor,
    weight_scale: _mbx_torch.Tensor,
) -> _mbx_torch.Tensor:
    blockscaled = _import_b12x_blockscaled()
    assert blockscaled is not None
    return blockscaled.mm_block_fp8(
        a, a_scale, weight, weight_scale, out_dtype=_mbx_torch.bfloat16
    )

@_mbx_b12x_mm.register_fake
def _mbx_b12x_mm_fake(a, weight, a_scale, weight_scale):
    return a.new_empty((a.shape[0], weight.shape[0]), dtype=_mbx_torch.bfloat16)

def _run_b12x_fp8_block_scaled_mm(a, weight, a_scale, weight_scale, out_dtype):
    out = _mbx_b12x_mm(a, weight, a_scale, weight_scale)
    return out if out_dtype == _mbx_torch.bfloat16 else out.to(out_dtype)
'''
open(p, "w").write(s)
print("b12x_block.py: compile-opaque custom-op wrapper installed")
PYEOF

# ---- OUT-OF-THE-BOX: first-boot self-conversion. The serve container carries the converter and the
# entrypoint wrapper runs it before vllm: checkpoint already complete -> instant skip; missing ->
# built from the Inferact source on /models (CPU-only, ~5-10 min, ONCE); source missing -> clear
# message naming the ./download.sh command. So a fresh user needs exactly:
#   ./download.sh Inferact/Qwen3.8-Flash-Next-NVFP4 && ./build-and-copy.sh <this> && ./run.sh <this>
COPY docker/serve-entry.sh /opt/mbx/serve-entry.sh
COPY docker/make-hibrid.py /opt/mbx/make-hibrid.py
RUN chmod +x /opt/mbx/serve-entry.sh

# ---- MBX_VOCAB_GEMV: M=1 lm_head via Triton GEMV (vendored from b12x master, see the module
# docstring in docker/mbx_vocab_gemv.py — measured 6.77 -> 4.74 ms on this exact head shape).
# Env-gated OFF by default; flip MBX_VOCAB_GEMV=1 in the recipe yaml to A/B.
COPY docker/mbx_vocab_gemv.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mbx_vocab_gemv.py
RUN python3 - <<'PYEOF'
p = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/logits_processor.py"
s = open(p).read()
assert "MBX_VOCAB_GEMV" not in s, "already patched"
a = """        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
            return lm_head.quant_method.apply("""
assert s.count(a) == 1, "_apply_head anchor not found/unique — upstream changed"
s = s.replace(a, """        # MBX_VOCAB_GEMV: M=1 head through the vendored Triton GEMV (draft steps
        # + bs=1 decode); falls through to the stock path on any mismatch.
        if _MBX_VOCAB_GEMV and embedding_bias is None:
            _mbx_out = _mbx_maybe_vocab_gemv(hidden_states, lm_head.weight)
            if _mbx_out is not None:
                return _mbx_out
        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
            return lm_head.quant_method.apply(""", 1)
s += '''

# mbx vocab gemv glue (functions resolve module globals at call time, so end-of-file is fine)
import os as _mbx_os
_MBX_VOCAB_GEMV = _mbx_os.environ.get("MBX_VOCAB_GEMV", "0") == "1"
if _MBX_VOCAB_GEMV:
    from vllm.model_executor.layers.mbx_vocab_gemv import maybe_vocab_gemv as _mbx_maybe_vocab_gemv
else:
    def _mbx_maybe_vocab_gemv(hidden_states, weight):
        return None
'''
open(p, "w").write(s)
print("logits_processor.py: MBX_VOCAB_GEMV fast path installed (default off)")
PYEOF
