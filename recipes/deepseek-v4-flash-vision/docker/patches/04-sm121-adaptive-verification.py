"""SM121 (GB10) enablers for DSpark adaptive verification on the FlashInfer DSV4 SM120 path.

Source: r0b0tlab/dsv4-flash-vision-exp-vllm-sm121 `docker/overlay-sm121-adaptive/` (whole-file
overlays, validated on 2×GB10) — reviewed and re-expressed here as the three minimal diffs they
contain (their two `get_cudagraph_support` overrides are covered by the upstream PR #52724 port in 03).

(a) indexer.py — DSA indexer decode. Upstream allows device-decided (flattened) query lengths only on
    Hopper+DeepGEMM. SM121 already runs the same per-token flattened decode (decode_len=1) for
    uniform k; the adaptive varlen flatten is torch.repeat_interleave, no sm90 headers involved →
    extend the gate to capability family 120. Hopper/SM100 predicates untouched.
(b) sparse_mla.py — the C128A global decode top-k indices were a `[:N, :W]` view of a wider buffer,
    which is NOT contiguous; the SM120 TVM kernel rejects it ("eidx must be contiguous") once
    adaptive verification produces non-uniform widths. Slice the first dim only (stays contiguous);
    extra_topk_length still clips the logical width.
(c) flashinfer_sparse.py — squeeze 3-D `[tokens, 1, topk]` index tensors to 2-D and force contiguity
    (indices + lengths, SWA and compressed-cache) right before the SM120 paged-attention call.
"""
import pathlib, vllm

root = pathlib.Path(vllm.__file__).parent

# ---- (a) indexer flatten gate ---------------------------------------------------------------------------
p = root / "v1/attention/backends/mla/indexer.py"
s = p.read_text()
assert "is_device_capability_family(120)" not in s.split("def _supports_flattened_device_query_lens")[1].split("def ")[0], "already patched"
old = '''def _supports_flattened_device_query_lens() -> bool:
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(90)
        and has_deep_gemm()
    )
'''
new = '''def _supports_flattened_device_query_lens() -> bool:
    # MBX: SM121 adaptive — Hopper (90) is the original flatten path. Family 120
    # (GB10/SM121) already runs the same per-token flattened decode_len=1 indexer
    # logits for uniform k; the adaptive varlen flatten is torch.repeat_interleave,
    # not sm90 headers, so admit it too.
    return (
        current_platform.is_cuda()
        and has_deep_gemm()
        and (
            current_platform.is_device_capability_family(90)
            or current_platform.is_device_capability_family(120)
        )
    )
'''
assert s.count(old) == 1, "indexer.py flatten-gate anchor not found/unique — upstream changed"
p.write_text(s.replace(old, new, 1))
print("indexer.py: flattened device query lens admitted on capability family 120")

# ---- (b) C128A decode indices contiguous ---------------------------------------------------------------
p = root / "models/deepseek_v4/sparse_mla.py"
s = p.read_text()
assert "MBX: SM121 adaptive" not in s, "already patched"
old = '''        if num_decode_tokens > 0:
            result["c128a_global_decode_topk_indices"] = global_decode.view(
                num_decode_tokens, 1, -1
            )
            result["c128a_decode_topk_lens"] = decode_lens
'''
new = '''        if num_decode_tokens > 0:
            # MBX: SM121 adaptive — a first-dim slice of the full-width buffer stays
            # contiguous; [:N, :W] of a wider buffer does not, and the SM120 TVM
            # kernel rejects it ("eidx must be contiguous"). extra_topk_length still
            # clips the logical width.
            contig = self.c128a_global_decode_buffer[:num_decode_tokens]
            result["c128a_global_decode_topk_indices"] = contig.view(
                num_decode_tokens, 1, contig.shape[-1]
            )
            result["c128a_decode_topk_lens"] = decode_lens
'''
assert s.count(old) == 1, "sparse_mla.py c128a anchor not found/unique — upstream changed"
p.write_text(s.replace(old, new, 1))
print("sparse_mla.py: C128A global decode top-k indices kept contiguous")

# ---- (c) squeeze/contiguous before the SM120 kernel ----------------------------------------------------
p = root / "models/deepseek_v4/nvidia/flashinfer_sparse.py"
s = p.read_text()
assert "extra_sparse_indices.contiguous()" not in s, "already patched"
old = '''        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        q = self._prepare_query(q, output)
        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
'''
new = '''        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        # MBX: SM121 adaptive — the SM120 TVM paged-attention entry checks
        # `eidx must be contiguous`; adaptive verification hands it 3-D
        # [tokens, 1, topk] views and non-contiguous slices. Normalise here.
        if extra_sparse_indices is not None:
            if extra_sparse_indices.dim() == 3 and extra_sparse_indices.size(1) == 1:
                extra_sparse_indices = extra_sparse_indices.squeeze(1)
            if not extra_sparse_indices.is_contiguous():
                extra_sparse_indices = extra_sparse_indices.contiguous()
        if extra_sparse_lengths is not None and not extra_sparse_lengths.is_contiguous():
            extra_sparse_lengths = extra_sparse_lengths.contiguous()
        if swa_indices.dim() == 3 and swa_indices.size(1) == 1:
            swa_2d = swa_indices.squeeze(1)
            swa_indices = swa_2d if swa_2d.is_contiguous() else swa_indices
        if not swa_indices.is_contiguous():
            swa_indices = swa_indices.contiguous()
        if not swa_lens.is_contiguous():
            swa_lens = swa_lens.contiguous()
        q = self._prepare_query(q, output)
        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
'''
assert s.count(old) == 1, "flashinfer_sparse.py decode anchor not found/unique — upstream changed"
p.write_text(s.replace(old, new, 1))
print("flashinfer_sparse.py: index tensors squeezed/contiguous before the SM120 kernel")
