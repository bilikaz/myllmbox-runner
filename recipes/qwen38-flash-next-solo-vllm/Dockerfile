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
        # MBX_PLE_MMAP: back the CPU-worker table with an (unlinked) NVMe file so shard copies write
        # through to disk and pages stay evictable — UMA boxes can't afford 95G resident (see Dockerfile).
        import os as _os
        if (
            _os.environ.get("MBX_PLE_MMAP", "0") == "1"
            and envs.VLLM_PLE_CPU_OFFLOAD
            and is_offload_process()
        ):
            import tempfile as _tf
            _w = self.ngram_embedding.weight
            _dir = _os.environ.get("MBX_PLE_MMAP_DIR", "/cache/ple-mmap")
            _os.makedirs(_dir, exist_ok=True)
            _f = _tf.NamedTemporaryFile(dir=_dir, delete=False); _p2 = _f.name; _f.close()
            _numel = 1
            for _s in _w.shape:
                _numel *= int(_s)
            _t = torch.from_file(
                _p2, shared=True, size=_numel, dtype=_w.dtype
            ).view(tuple(_w.shape))
            _os.unlink(_p2)
            _new = torch.nn.Parameter(_t, requires_grad=False)
            for _n, _v in vars(_w).items():
                setattr(_new, _n, _v)
            self.ngram_embedding.weight = _new"""
assert s.count(anchor) == 1, "ngram_embedding constructor anchor not found/unique — upstream changed"
assert "MBX_PLE_MMAP" not in s, "already patched"
s = s.replace(anchor, patch, 1)
assert "is_offload_process" in s and "envs." in s, "expected imports missing from ple_layer.py"
open(p, "w").write(s)
print("ple_layer.py patched: PLE table file-backed under MBX_PLE_MMAP=1")
PYEOF
