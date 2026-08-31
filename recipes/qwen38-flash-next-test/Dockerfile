# mbx-qwen38-flash-next — HIBRID45 (champion 2026-08-30): NVFP4 dense except attention.
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

# ---- HIBRID45: dense -> NVFP4 W4A16 except QSA q/k/v/o (bf16). NO dispatch splice is needed —
# vLLM's mixed config routes quant_algo W4A16_NVFP4 natively (ModelOptNvFp4W4A16LinearMethod, Marlin
# via use_a16=True) and the attention stays plain bf16. The image carries only: converter +
# first-boot entrypoint + index-strict loader + the proven vocab GEMV. (The fp8-era dispatch splice,
# b12x wrapper and compile-cache env keying are GONE — nothing block-fp8 left to dispatch.)
COPY docker/serve-entry.sh /opt/mbx/serve-entry.sh
COPY docker/make-hibrid46.py /opt/mbx/make-hibrid46.py
RUN chmod +x /opt/mbx/serve-entry.sh

# ---- MBX_INDEX_STRICT (default ON): trust model.safetensors.index.json when iterating shards ----
# Hibrid checkpoints HARDLINK the source shards and re-point the replaced tensors to a new shard via
# the index — but the stale originals are still physically inside the old files, and vLLM's loader
# iterates FILES, not the index, so every replaced weight arrives twice. The fp8 hibrid survived by
# accident (same [out,in] shape -> assert passes; correct shard sorts last and wins); 4-bit packs to
# [out,in/2] u8, so the stale BF16 copy hard-asserts in load_merged_column_weight. Fix: skip any
# tensor read from a file the index does not map it to. No-op for non-hibrid checkpoints.
RUN python3 - <<'PYEOF'
p = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py"
src = open(p).read()
assert "MBX_INDEX_STRICT" not in src, "already patched"
assert "import json" in src and "import os" in src

anchor = "    leftover_state_dict: dict[str, torch.Tensor] = {}\n"
assert src.count(anchor) == 1, "iterator anchor not found/unique — upstream changed"
inject = anchor + """
    # MBX: index-strict dedupe (see Dockerfile) — skip stale duplicates of tensors the
    # index maps to a different shard file.
    _mbx_index_map: dict[str, str] = {}
    if os.environ.get("MBX_INDEX_STRICT", "1") == "1" and sorted_files:
        _mbx_idx = os.path.join(
            os.path.dirname(sorted_files[0]), "model.safetensors.index.json"
        )
        if os.path.isfile(_mbx_idx):
            try:
                with open(_mbx_idx) as _mbx_f:
                    _mbx_index_map = json.load(_mbx_f).get("weight_map") or {}
            except Exception:
                _mbx_index_map = {}

    def _mbx_stale(name: str, st_file: str) -> bool:
        owner = _mbx_index_map.get(name)
        return owner is not None and owner != os.path.basename(st_file)
"""
src = src.replace(anchor, inject)

old1 = """            for name, param in state_dict.items():
                if not should_skip_weight(name, local_expert_ids):
                    yield name, param"""
new1 = """            for name, param in state_dict.items():
                if not should_skip_weight(name, local_expert_ids) and not _mbx_stale(
                    name, st_file
                ):
                    yield name, param"""
assert src.count(old1) == 1, "eager path anchor not found/unique"
src = src.replace(old1, new1)

old2 = """                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param"""
new2 = """                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids) or _mbx_stale(
                        name, st_file
                    ):
                        continue
                    param = f.get_tensor(name)
                    yield name, param"""
assert src.count(old2) == 1, "default path anchor not found/unique"
src = src.replace(old2, new2)

old3 = """                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    state_dict[name] = f.get_tensor(name)"""
new3 = """                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids) or _mbx_stale(
                        name, st_file
                    ):
                        continue
                    state_dict[name] = f.get_tensor(name)"""
assert src.count(old3) == 1, "torchao path anchor not found/unique"
src = src.replace(old3, new3)

open(p, "w").write(src)
print("weight_utils.py: MBX_INDEX_STRICT stale-duplicate skip installed (default on)")
PYEOF

# ---- MBX_VOCAB_GEMV: championship-proven on the fp8 hibrid (+5.5% steps) — same vendored kernel ----
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

# ---- MBX_PLE_QBITS (TEST LANE): quantize the 95.4G bf16 PLE/n-gram table to 2/3/4-bit
# (group-64 asymmetric) RESIDENT in the CPU offload worker; every gather dequants on read. Built
# lazily at the FIRST gather (engine warmup), streaming the bf16 mmap bank once sequentially --
# after that the bank is never paged again, killing the ~200MB/s NVMe traffic at c>=3.
# Sizes: 4-bit ~26.9G / 3-bit ~19.2G / 2-bit ~14.9G resident. Ladder protocol: bf16 champion is
# the measured top rung; start at the highest width that FITS (int3 at KV 12G), descend only if
# quality holds. PLE feeds the MAIN forward: judged by gauntlet + paired A/B, not just acceptance.
RUN python3 - <<'PYEOF'
p = "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
s = open(p).read()
assert "MBX_PLE_QBITS" not in s, "already patched"

anchor = """        if output_buffer is not None:
            output = output_buffer[:num_tokens, : self.embedding_dim]
            torch.index_select(
                self.ngram_embedding.weight,
                0,
                ngram_ids.reshape(-1),
                out=output.reshape(-1, self.head_dim),
            )
            return output
        return self.ngram_embedding(ngram_ids).flatten(-2)"""
patch = """        if output_buffer is not None:
            output = output_buffer[:num_tokens, : self.embedding_dim]
            if _mbx_ple_qbits() and is_offload_process():
                _mbx_q_gather(self, ngram_ids.reshape(-1), output.reshape(-1, self.head_dim))
                return output
            torch.index_select(
                self.ngram_embedding.weight,
                0,
                ngram_ids.reshape(-1),
                out=output.reshape(-1, self.head_dim),
            )
            return output
        if _mbx_ple_qbits() and is_offload_process():
            _flat = ngram_ids.reshape(-1)
            _out = torch.empty(
                (_flat.numel(), self.head_dim), dtype=self.ngram_embedding.weight.dtype
            )
            _mbx_q_gather(self, _flat, _out)
            return _out.view(*ngram_ids.shape, self.head_dim).flatten(-2)
        return self.ngram_embedding(ngram_ids).flatten(-2)"""
assert s.count(anchor) == 1, "PLE gather anchor not found/unique -- upstream changed"
s = s.replace(anchor, patch, 1)

s += '''

# ---- MBX_PLE_QBITS helpers (module-level; resolve at call time) ----------------------------------
import os as _mbx_os


def _mbx_ple_qbits() -> int:
    """0 = off; 2/3/4 = quantize the PLE table to that many bits (group asymmetric, resident)."""
    try:
        b = int(_mbx_os.environ.get("MBX_PLE_QBITS", "0"))
    except ValueError:
        return 0
    return b if b in (2, 3, 4) else 0


def _mbx_pack(q: "torch.Tensor", bits: int) -> "torch.Tensor":
    """Pack [n, D] uint8 codes into bytes. 2b: 4/byte; 4b: 2/byte; 3b: 8 codes -> 3 bytes."""
    n, D = q.shape
    if bits == 2:
        v = q.view(n, D // 4, 4)
        return v[..., 0] | (v[..., 1] << 2) | (v[..., 2] << 4) | (v[..., 3] << 6)
    if bits == 4:
        v = q.view(n, D // 2, 2)
        return v[..., 0] | (v[..., 1] << 4)
    v = q.view(n, D // 8, 8)
    out = torch.empty((n, D // 8, 3), dtype=torch.uint8)
    out[..., 0] = v[..., 0] | (v[..., 1] << 3) | ((v[..., 2] & 3) << 6)
    out[..., 1] = (v[..., 2] >> 2) | (v[..., 3] << 1) | (v[..., 4] << 4) | ((v[..., 5] & 1) << 7)
    out[..., 2] = (v[..., 5] >> 1) | (v[..., 6] << 2) | (v[..., 7] << 5)
    return out.view(n, (D // 8) * 3)


def _mbx_unpack(pk: "torch.Tensor", bits: int, D: int) -> "torch.Tensor":
    """Inverse of _mbx_pack -> [n, D] uint8 codes."""
    n = pk.shape[0]
    if bits == 2:
        q = torch.empty((n, D // 4, 4), dtype=torch.uint8)
        q[..., 0] = pk & 3
        q[..., 1] = (pk >> 2) & 3
        q[..., 2] = (pk >> 4) & 3
        q[..., 3] = (pk >> 6) & 3
        return q.view(n, D)
    if bits == 4:
        q = torch.empty((n, D // 2, 2), dtype=torch.uint8)
        q[..., 0] = pk & 15
        q[..., 1] = (pk >> 4) & 15
        return q.view(n, D)
    b = pk.view(n, D // 8, 3)
    q = torch.empty((n, D // 8, 8), dtype=torch.uint8)
    q[..., 0] = b[..., 0] & 7
    q[..., 1] = (b[..., 0] >> 3) & 7
    q[..., 2] = ((b[..., 0] >> 6) & 3) | ((b[..., 1] & 1) << 2)
    q[..., 3] = (b[..., 1] >> 1) & 7
    q[..., 4] = (b[..., 1] >> 4) & 7
    q[..., 5] = ((b[..., 1] >> 7) & 1) | ((b[..., 2] & 3) << 1)
    q[..., 6] = (b[..., 2] >> 2) & 7
    q[..., 7] = (b[..., 2] >> 5) & 7
    return q.view(n, D)


def _mbx_q_build(self) -> None:
    """One-time EVER per (tag,bits,group): quantize the bf16 bank and PERSIST the packed table to
    the cache (write-once + marker, same pattern as the bank itself); later boots mmap the packed
    file directly -- no bf16 skim, no rebuild, PLE ready in seconds and page-cache-friendly."""
    import time as _t

    _log = __import__("logging").getLogger("mbx.ple.qbits")
    bits = _mbx_ple_qbits()
    w = self.ngram_embedding.weight
    V, D = int(w.shape[0]), int(w.shape[1])
    group = int(_mbx_os.environ.get("MBX_PLE_QBITS_GROUP", "64"))
    align = 8 if bits == 3 else (4 if bits == 2 else 2)
    if D % group != 0:
        group = D
    assert D % align == 0, f"PLE head_dim {D} not divisible by {align} -- packing needs it"
    G = D // group
    levels = (1 << bits) - 1
    bytes_per_row = {2: D // 4, 3: (D // 8) * 3, 4: D // 2}[bits]

    _dir = _mbx_os.environ.get("MBX_PLE_MMAP_DIR", "/cache/ple-mmap")
    _tag = _mbx_os.environ.get("MBX_PLE_MMAP_TAG", "") or "untagged"
    _base = _mbx_os.path.join(_dir, f"{_tag}-qbits{bits}-g{group}-{V}x{D}")
    _pf, _sf, _mf, _ok = _base + ".packed", _base + ".scales", _base + ".mins", _base + ".ok"

    if (
        _mbx_os.path.exists(_ok)
        and _mbx_os.path.getsize(_pf) == V * bytes_per_row
        and _mbx_os.path.getsize(_sf) == V * G * 2
        and _mbx_os.path.getsize(_mf) == V * G * 2
    ):
        packed = torch.from_file(_pf, shared=True, size=V * bytes_per_row, dtype=torch.uint8).view(V, bytes_per_row)
        scales = torch.from_file(_sf, shared=True, size=V * G, dtype=torch.float16).view(V, G)
        mins = torch.from_file(_mf, shared=True, size=V * G, dtype=torch.float16).view(V, G)
        self._mbx_q = (packed, scales, mins, group, bits)
        _log.info(
            "MBX_PLE_QBITS: mmapped persisted int%d table %s (%.1f GiB, no rebuild)",
            bits, _base, (packed.numel() + 2 * scales.numel() + 2 * mins.numel()) / 2**30,
        )
        print(f"MBX_PLE_QBITS: mmapped persisted int{bits} table {_base} (no rebuild)", flush=True)
        return

    _mbx_os.makedirs(_dir, exist_ok=True)
    packed = torch.empty((V, bytes_per_row), dtype=torch.uint8)
    scales = torch.empty((V, G), dtype=torch.float16)
    mins = torch.empty((V, G), dtype=torch.float16)
    rows = max(1, (1 << 27) // D)  # ~512MB fp32 working chunk
    t0 = _t.time()
    i = 0
    while i < V:
        j = min(V, i + rows)
        x = w[i:j].to(torch.float32)
        g = x.view(j - i, G, group)
        mn = g.amin(-1)
        sc = ((g.amax(-1) - mn) / levels).clamp_min(1e-8)
        q = torch.round((g - mn.unsqueeze(-1)) / sc.unsqueeze(-1)).clamp_(0, levels).to(torch.uint8)
        packed[i:j] = _mbx_pack(q.view(j - i, D), bits)
        scales[i:j] = sc.to(torch.float16)
        mins[i:j] = mn.to(torch.float16)
        i = j
    self._mbx_q = (packed, scales, mins, group, bits)
    try:
        packed.numpy().tofile(_pf)
        scales.numpy().tofile(_sf)
        mins.numpy().tofile(_mf)
        _mbx_os.sync()
        open(_ok, "w").write("ok")
        for _f in (_pf, _sf, _mf, _ok):
            _mbx_os.chmod(_f, 0o644)
        _persist = "persisted for future boots"
    except Exception as _e:  # noqa: BLE001
        _persist = f"persist FAILED ({_e}); staying resident this boot"
    _log.info(
        "MBX_PLE_QBITS: %dx%d table quantized to int%d in %.1fs (group %d) -- %.1f GiB; %s",
        V, D, bits, _t.time() - t0, group,
        (packed.numel() + 2 * scales.numel() + 2 * mins.numel()) / 2**30, _persist,
    )
    print(f"MBX_PLE_QBITS: quantized to int{bits} in {_t.time()-t0:.1f}s -- {_persist}", flush=True)


def _mbx_q_gather(self, ids: "torch.Tensor", out: "torch.Tensor") -> None:
    """Gather rows for ids and dequant into out (bf16). Runs on the offload CPU worker."""
    if getattr(self, "_mbx_q", None) is None:
        _mbx_q_build(self)
    packed, scales, mins, group, bits = self._mbx_q
    pk = packed.index_select(0, ids)
    D = out.shape[-1]
    x = _mbx_unpack(pk, bits, D).view(pk.shape[0], D // group, group).to(torch.float32)
    x.mul_(scales.index_select(0, ids).unsqueeze(-1).to(torch.float32))
    x.add_(mins.index_select(0, ids).unsqueeze(-1).to(torch.float32))
    out.copy_(x.view(pk.shape[0], D))
'''
open(p, "w").write(s)
print("ple_layer.py: MBX_PLE_QBITS resident 2/3/4-bit table + dequant-on-gather installed (default off)")
PYEOF
