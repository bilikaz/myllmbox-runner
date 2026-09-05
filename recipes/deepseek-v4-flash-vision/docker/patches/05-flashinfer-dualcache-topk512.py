"""FlashInfer 0.6.18 csrc — dual-cache sparse-MLA PREFILL dispatch for index_topk=512 (DSV4 Vision-Exp).

Stock 0.6.18 `dispatch_dsv4_dual` only instantiates the dual-cache (SWA + compressed extra cache)
prefill for TOPK=128; the single-cache path already has 512. Vision-Exp's config has index_topk=512
→ the first image request dies with "Unsupported sparse-MLA prefill configuration". This adds the
512 arm for both enumerated extra-cache page layouts (extra_page_block_size 64 and 2), num_heads
8/16/32/64/128. FP8 compute mode, matching the single-cache heuristic in this same file (K≥512 → FP8,
"larger K amortises FP8's higher Tensor-Core throughput"); FlashInfer PR #4850 chose BF16 for the
same arms — either is valid, this one is the r0b0tlab variant validated on 2×GB10. Superseded
upstream by the #4802 SM120 refactor (merged 2026-09-03, not in any release).

The file is JIT source: nvcc compiles it on the first prefill into FLASHINFER_WORKSPACE_BASE
(/cache/flashinfer-workspace) — ~20 s per box (measured, sm_121a), then cached.

THE TRAP (cost us one failed boot): the image also ships `flashinfer_jit_cache` — PREBUILT AOT
modules — and `JitSpec.is_aot` is simply `aot_path.exists()`: when
`flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so` is present, FlashInfer loads
it and never looks at the patched source → the stock TOPK=128 dispatcher runs and the profiling
prefill dies with the exact error above. Robot's published tag is `fi512b` — their undocumented
second build presumably did the same removal. So this patch also DELETES that one AOT module
(952K); every other prebuilt module stays. The JIT then compiles the patched source on first use.
"""
import pathlib, shutil, flashinfer

p = pathlib.Path(flashinfer.__file__).parent / "data/csrc/sparse_mla_sm120_prefill.cu"
s = p.read_text()
assert "DISPATCH_BY_NH_PBSX_512" not in s, "already patched"
old = '''  if (topk != 128) return false;
  if (extra_page_block_size == 64) {
    DISPATCH_BY_NH_PBSX(64);
'''
new = '''  // MBX: Vision-Exp / DSV4 index_topk=512 dual-cache prefill. Stock 0.6.18 only
  // instantiates dual-cache TOPK=128; single-cache already has 512.
  if (topk == 512) {
#define DISPATCH_BY_NH_PBSX_512(PBSX)                   \\
  do {                                                  \\
    switch (num_heads) {                                \\
      case 8:                                           \\
        DISPATCH_DUAL_MG_CM(FP8, 8, 512, PBSX, 1);      \\
        return true;                                    \\
      case 16:                                          \\
        DISPATCH_DUAL_MG_CM(FP8, 16, 512, PBSX, 1);     \\
        return true;                                    \\
      case 32:                                          \\
        DISPATCH_DUAL_MG_CM(FP8, 32, 512, PBSX, 2);     \\
        return true;                                    \\
      case 64:                                          \\
        DISPATCH_DUAL_MG_CM(FP8, 64, 512, PBSX, 2);     \\
        return true;                                    \\
      case 128:                                         \\
        DISPATCH_DUAL_MG_CM(FP8, 128, 512, PBSX, 2);    \\
        return true;                                    \\
      default:                                          \\
        return false;                                   \\
    }                                                   \\
  } while (0)
    if (extra_page_block_size == 64) {
      DISPATCH_BY_NH_PBSX_512(64);
    } else if (extra_page_block_size == 2) {
      DISPATCH_BY_NH_PBSX_512(2);
    }
#undef DISPATCH_BY_NH_PBSX_512
    return false;
  }
  if (topk != 128) return false;
  if (extra_page_block_size == 64) {
    DISPATCH_BY_NH_PBSX(64);
'''
assert s.count(old) == 1, "prefill.cu dual-cache dispatch anchor not found/unique — FlashInfer changed"
# guard: the macro we call must exist in this file with the expected arity
assert "#define DISPATCH_DUAL_MG_CM(CM, NH, TK, PBSX, NHG)" in s, "DISPATCH_DUAL_MG_CM macro signature changed"
p.write_text(s.replace(old, new, 1))
print("sparse_mla_sm120_prefill.cu: dual-cache TOPK=512 prefill dispatch added (FP8 compute)")

# ---- evict the prebuilt AOT module so the patched source is what gets compiled -------------------------
import importlib.util
spec = importlib.util.find_spec("flashinfer_jit_cache")
assert spec is not None and spec.origin, "flashinfer_jit_cache not installed — AOT shadowing impossible; fine"
aot = pathlib.Path(spec.origin).parent / "jit_cache" / "sparse_mla_sm120"
assert aot.is_dir(), f"expected prebuilt AOT module at {aot} — layout changed, re-check is_aot()"
shutil.rmtree(aot)
assert not aot.exists()
print(f"removed prebuilt AOT module {aot} → sparse_mla_sm120 will JIT from the patched csrc")
