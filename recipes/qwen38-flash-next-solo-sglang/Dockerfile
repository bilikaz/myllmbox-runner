# mbx-qwen38-flash-next-sglang — the stock qwen38flashnext sglang image + the SM121 QSA patch.
# The stock image CANNOT serve this model on GB10: the QSA (Qwen sparse attention) backend resolves its
# packed varlen attention to flash-attn-4's CuTe DSL kernels, which fail on SM121 (MLIR layout-congruence
# at warmup; with fp8 KV it dies even earlier on a q/k dtype assert — both measured on this cluster).
# The patch (from MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks, MIT, validated 2-node TP2 on 2× Spark
# incl. CUDA graphs + NEXTN spec decode) swaps in a Triton FlashDecoding-style fallback whose cu_seqlens
# are read on-device (CUDA-graph replay safe).
FROM lmsysorg/sglang:qwen38flashnext@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1

COPY docker/qsa_fa_fallback.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa_fa_fallback.py

# Splice the fallback into the backend's kernel resolution: non-SM100 → Triton fallback (MiaAI's splice,
# verbatim). The asserts make a silently-changed upstream layout fail the BUILD, not the serve.
RUN python3 - <<'PYEOF'
p = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
s = open(p).read()
anchor = "    try:\n        from flash_attn import flash_attn_varlen_func"
patch = (
    "    from sglang.srt.utils import is_sm100_supported\n"
    "    if not is_sm100_supported():\n"
    "        from sglang.srt.layers.attention.qsa_fa_fallback import triton_varlen_attn_func\n"
    "        return triton_varlen_attn_func\n"
) + anchor
assert anchor in s, "anchor not found — upstream image layout changed"
assert "qsa_fa_fallback" not in s, "already patched"
open(p, "w").write(s.replace(anchor, patch, 1))
print("qwen_sparse_attn_backend.py patched for SM121")
PYEOF

# PLE-FROM-DISK (MBX_PLE_MMAP=1): the stock offload copies the ~24G/node n-gram table into PINNED host
# RAM — on GB10 UMA that's the same physical pool as the GPU, so nothing is saved. This patch makes the
# offload wrapper ADOPT the mmap-backed CPU tensor that sglang's safetensors loader already hands it
# (zero-copy view of the file on NVMe) instead of copying: rows page in from disk on first gather and the
# OS evicts them freely (0xBakeer measured 1-3 page faults/token for this pattern on a Spark). GB10's
# coherent page tables make the host mmap pointer directly GPU-readable. Gated by env MBX_PLE_MMAP=1
# (recipe sets it); any shape/dtype surprise falls back to the stock pinned copy with a warning.
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
# in mmap mode, back the wrapper's table buffer with an (unlinked) FILE on /cache (NVMe) instead of
# pinned RAM: EVERY load path — plain weight_loader copies AND the sharded-PLE merge (Inferact's
# ngram_embedding.shard_N, which bypasses weight_loader and silently pinned 95G) — then writes through
# to disk, and pages stay evictable. The adoption loader above still short-circuits to the checkpoint's
# own mmap when it can (no write pass at all).
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
