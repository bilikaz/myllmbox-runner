#!/usr/bin/env python3
"""MBX: fp8_e4m3 / nvfp4 KV cache on the QSA path = upstream vLLM PR #54846 (andreasgru, open), ported onto this build.

Base of the PR: vLLM nightly 8a728663 (2026-09-04), model dir `qwen4_exp`. Our build (0.1.dev20073+g8e685d198) has the
same three files under `qwen3_8_flash_next` with (a) the model renamed, (b) formatting differences, (c) EXTRA kernels in
ops/qsa.py (the QSA indexer's compressed-key scoring path) that the PR never touches. The PR was applied as a diff with
the rename substituted: 18 of 21 hunks mechanically, 3 by hand (multi-line raise statements). Files ship whole in
docker/overlays/ and are copied over the originals here; anchors below assert we are overlaying the file we ported from.
Reference port: tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark (their overlay = PR head 1e11f40bbc on upstream's qwen4_exp dir
on nightly 8a728663; no bytes shared with ours — different base dir and version; their ledger: fp8 KV = 1.59× the bf16 pool,
quality clean). Credits: andreasgru (the change), tonyd2wild (the reference port), myllmbox (this port, this patch, the GB10
test run). All three overlays keep vLLM's Apache-2.0 SPDX header; vLLM is Copyright contributors to the vLLM project.

PROVENANCE IS CHECKED, NOT DOCUMENTED: BASE_SHA256 = the vendor files this port was made from (image myllmbox/qwen38-flash-next-vllm:v1,
vLLM 0.1.dev20073+g8e685d198); OVERLAY_SHA256 = the files shipped in docker/overlays/. Both are asserted at build time, so an
edited overlay or a changed base fails the image build here instead of outdating a note. Re-derive and update the two tables
when you intentionally change either (sha256sum <file>).
Activation: `--kv-cache-dtype fp8` (or fp8_e4m3; nvfp4 = separate branch, untested here). `--check` = anchors + hashes only.
"""
import hashlib, shutil, sys

BASE_SHA256 = {       # vendor originals, before overlay (myllmbox/qwen38-flash-next-vllm:v1)
    "qsa.py": "748addc85efaa8f7df940d1245bc900192f92e1f17af8fa774625758600751cb",
    "ops_qsa.py": "c4ffe3674cafa0ce2dabc39a39f0ddbb4b594bc358ad210ffce9d04383350c7f",
    "platforms_interface.py": "7109cdf97649c1b7a3e471fc98df06be2eacb77501df94016a44f1327b01f55d",
}
OVERLAY_SHA256 = {    # the files in docker/overlays/ = base + PR #54846 with the model-dir rename substituted
    "qsa.py": "965c97657bbdff5a7118ce2385d540afb8a3257a98225a0d7bda2bd00dc1e33c",
    "ops_qsa.py": "5c21448b29976bb7d7b8909bd980e28494af367e2270a3d5b87c2c90ffe9dda0",
    "platforms_interface.py": "840886e777c171da77fa846d58e8b240ab282aa8f60a30e8a8b1af6034936b98",
}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

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
    assert sha(dst) == BASE_SHA256[name], f"{name}: vendor base file differs from the one this port was made from ({sha(dst)[:12]} vs {BASE_SHA256[name][:12]}) — re-port PR #54846 onto it and update BASE_SHA256/OVERLAY_SHA256"
    ov = open(f"/opt/mbx/overlays/{name}").read()
    for a in anchors[name][1]:
        assert a in ov, f"{name}: overlay lacks {a!r}"
    assert sha(f"/opt/mbx/overlays/{name}") == OVERLAY_SHA256[name], f"{name}: overlay file changed ({sha(f'/opt/mbx/overlays/{name}')[:12]} vs {OVERLAY_SHA256[name][:12]}) — if intentional, update OVERLAY_SHA256 here"
    if not CHECK:
        shutil.copyfile(f"/opt/mbx/overlays/{name}", dst)
print("QSA fp8/nvfp4 KV patch (PR #54846 port):", "anchors + base/overlay sha256 OK" if CHECK else "applied — kv-cache-dtype fp8 / fp8_e4m3 / nvfp4 accepted by the QSA path")
