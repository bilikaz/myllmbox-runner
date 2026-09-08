#!/usr/bin/env python3
"""MBX: fp8_e4m3 / nvfp4 KV cache on the QSA path = upstream vLLM PR #54846 (andreasgru, open), ported onto this build.

Base of the PR: vLLM nightly 8a728663 (2026-09-04), model dir `qwen4_exp`. Our build (0.1.dev20073+g8e685d198) has the
same three files under `qwen3_8_flash_next` with (a) the model renamed, (b) formatting differences, (c) EXTRA kernels in
ops/qsa.py (the QSA indexer's compressed-key scoring path) that the PR never touches. The PR was applied as a diff with
the rename substituted: 18 of 21 hunks mechanically, 3 by hand (multi-line raise statements). Files ship whole in
docker/overlays/ and are copied over the originals here; anchors below assert we are overlaying the file we ported from.
Reference port: tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark (their overlay = PR head 1e11f40bbc on the nightly;
their ledger: fp8 KV = 1.59× the bf16 pool at the same utilization, quality clean). Apache-2.0 headers kept.
Activation: `--kv-cache-dtype fp8` (or fp8_e4m3; nvfp4 = separate branch, untested here). `--check` = anchors only.
"""
import shutil, sys

CHECK = "--check" in sys.argv
V = "/usr/local/lib/python3.12/dist-packages/vllm/"
targets = {
    "qsa.py": V + "models/qwen3_8_flash_next/nvidia/qsa.py",
    "ops_qsa.py": V + "models/qwen3_8_flash_next/nvidia/ops/qsa.py",
    "platforms_interface.py": V + "platforms/interface.py",
}
anchors = {  # things that must be in the ORIGINAL (proves same base) and in the OVERLAY (proves the port is complete)
    "qsa.py": (['"Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"', "class Qwen3_8FlashNextQSAFlashAttentionImpl"],
               ["_QSA_KV_CACHE_DTYPES", "kv_cache_fp8", "_nvfp4_views_for", "class Qwen3_8FlashNextQSAFlashAttentionImpl"]),
    "ops_qsa.py": (["def _qsa_mqa_paged_kernel(", "def qsa_select_paged_tokens(", "def _qsa_sparse_paged_gqa_splitk_kernel("],
                   ["def _qsa_mqa_paged_kernel(", "def qsa_select_paged_tokens(", "KV_QUANT_FP8", "KV_QUANT_NVFP4", "_nvfp4_decode_e2m1"]),
    "platforms_interface.py": (["class Platform"], ["class Platform", "nvfp4"]),
}
for name, dst in targets.items():
    orig = open(dst).read()
    assert "_QSA_KV_CACHE_DTYPES" not in orig or name != "qsa.py", "already patched"
    for a in anchors[name][0]:
        assert a in orig, f"{name}: original lacks anchor {a!r} — not the base this port was made from"
    ov = open(f"/opt/mbx/overlays/{name}").read()
    for a in anchors[name][1]:
        assert a in ov, f"{name}: overlay lacks {a!r}"
    if not CHECK:
        shutil.copyfile(f"/opt/mbx/overlays/{name}", dst)
print("QSA fp8/nvfp4 KV patch (PR #54846 port):", "anchors OK" if CHECK else "applied — kv-cache-dtype fp8 / fp8_e4m3 / nvfp4 accepted by the QSA path")
