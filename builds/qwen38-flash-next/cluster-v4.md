# docker.io/myllmbox/qwen38-flash-next-cluster-vllm:v4

**Published:** 2026-09-08 (pushed from the head DGX Spark; base layers already on the Hub, only the patch layer uploaded).
**Rebuild:** `docker build -t myllmbox/qwen38-flash-next-cluster-vllm:v4 builds/qwen38-flash-next/cluster/`
**Digest:** `sha256:91423fc292d527935b2f0363cc614305b1c1a00dc56981953a723abe1b50ed2e`
**Image id (local, = mbx-qwen38-flash-next-cluster):** `sha256:9b638749b5bc61bf5df039927a90b1297d03e9a5fc879c41d7a044208a3361b2`
**Reason for a new tag:** adds patch 04 — fp8_e4m3 (and nvfp4, untested) KV cache on the QSA attention path, upstream
PR #54846 ported (the same port as the solo image v2's patch 06). v3 stays as published (bf16 KV).

## What changed vs v3
`cluster/docker/patches/04-qsa-fp8-nvfp4-kv.py`: whole-file overlays of `models/qwen3_8_flash_next/nvidia/qsa.py`,
`.../ops/qsa.py` and `platforms/interface.py` (anchor-asserted against the v1 base; provenance in `docker/overlays/`),
the PR's 13 numerical tests in `docker/tests/` (pass on the GB10). Patches 01–03 byte-identical to v3. The recipe image
`mbx-qwen38-flash-next-cluster` (id 9b638749) IS this image: same base digest, same four patch files, same converter —
tagged, not rebuilt.

## Measured with it (recipes/qwen38-flash-next-cluster, hibrid47, 2× Spark, K=4, kv-cache-dtype fp8, 28 G pin)
Boot 2026-09-08 05:04: GPU KV cache 2,846,834 tokens (bf16 same pin: 1.71M; 1.66×), concurrency 10.86× a 262k request.
c=1 thinking on, one 39,487-token request in 670 s: steps 17.3/s (16.8–17.7), code phase 74 tok/s (68–81), thinking 54,
whole request 59. KV 1.9 % per running request at admission (GDN state blocks; unchanged from bf16) → ~52 short seats.
c=64 hold, c=32 ladder: to be measured. Card: experiments/data/qwen38-cluster-hibrid47-fp8-card.png.

## Consumers
`solo/qwen38-flash-next-cluster-recipe` v2.1 (recipe.yaml `image:`), `recipes/qwen38-flash-next-cluster` (as the local
`mbx-qwen38-flash-next-cluster` build of the same Dockerfile).
