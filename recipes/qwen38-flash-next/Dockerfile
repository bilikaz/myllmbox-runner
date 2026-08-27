# mbx-qwen38-flash-next — official vLLM qwen38-flash-next image + PLE-from-disk patch (SM121/GB10 UMA).
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
                    __import__("logging").getLogger("mbx.ple").info("MBX_PLE_MMAP: persistent PLE table complete (marker written)")
                loaded.add("ngram_embedding.weight")
                continue"""
assert s.count(la) == 1, "shard-copy anchor not found/unique — upstream changed"
s = s.replace(la, lp, 1)
open(p, "w").write(s)
print("ple_layer.py patched: PLE file-backed + write-once persistence under MBX_PLE_MMAP_TAG")
PYEOF
