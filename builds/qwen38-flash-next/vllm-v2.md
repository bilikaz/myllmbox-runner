# docker.io/myllmbox/qwen38-flash-next-vllm:v2

**Published:** 2026-09-07 (pushed from ai1; base layers already on the Hub, only the patch layer uploaded; built 13:24 on ai1 as `mbx-qwen38-flash-next-solo-test`, image id
`sha256:d2d370a5786db47ddf49d6857be3f6e151609cc6385c37c950270f19e70194f2`; served boot 10 the same day).
**Digest:** `sha256:b2f35cd81998f4d58ef4266792282f48309e0d1bbe19f4816dc2f98684e0a3ec`
**Reason for a new tag:** the single-Spark kit moves to the hibrid47 checkpoint. v1's int3-in-worker path stays published.

## What it is
`solo/Dockerfile` here (= `recipes/qwen38-flash-next/Dockerfile`, byte-identical `docker/`): v1 base
(`myllmbox/qwen38-flash-next-vllm:v1@sha256:92ccd7de…`) + six anchor-asserted patches:
01 multi-node offload gate (fallback, inert) · 02 GPU-resident NVFP4 PLE method · 03 loader page-cache drop ·
04 demand-paged table (`MBX_PLE_MMAP`: mmap + DLPack over the mapping, GB10 ATS gather, populate + watcher) ·
05 dual gather path (`vllm::mbx_ple_gather` splitting op, O_DIRECT during boot, mmap after; `MBX_PLE_MMAP_MODE`,
`MBX_PLE_MMAP_PREWARM`, flag files) · 06 fp8/nvfp4 KV on QSA (PR #54846 port as whole-file overlays; the PR's tests in
`docker/tests/`). Build-time sanity: nvfp4 round-trip, mmap+DLPack CPU path on a synthetic 2-shard checkpoint, nvme path +
op in both modes, splitting op registered, fp8 anchors + py_compile.

## Validation before push (2026-09-07, single DGX Spark, recipes/qwen38-flash-next-solo-test boot 10)
- Weights 73.3 GiB (686 s); `GPU KV cache size: 391,943 tokens` (fp8, 7 GB pin, 262,144 ctx, attention block 3200);
  graph capture 0.50 GiB; flip after 38 eager steps; health 200; free 15 G / avail 26 G, swap 0 before populate.
- populate: 27 GB mapped, 5.7 GB cold heaps swapped once, swap-ins ≈ 0 after; NVMe 0.4 reads/s decoding.
- c=1 pasture thinking off ×4: 49.0–51.0 tok/s avg, 52.7–54.2 peak, steps 14.40–14.46, finish=stop.
- c=1 pasture thinking on ×4 (30k tokens, 12–14 min each): 39.1–42.3 avg, peaks 52–56, steps 14.34–14.42.
- Not yet on this image: the gauntlet renders (tests/), max-res image at 8192 batched tokens, K=4 A/B.

## Shipping serve config (at release)
`--kv-cache-memory 7000000000 --kv-cache-dtype fp8 --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 8192
--gpu-memory-utilization 0.70 --speculative-config mtp K=3 --async-scheduling`; env `MBX_PLE_MMAP=1 MBX_PLE_MMAP_MODE=auto
MBX_PLE_MMAP_PREWARM=auto MALLOC_MMAP_THRESHOLD_=65536 MALLOC_TRIM_THRESHOLD_=131072`, cpuset 5-9,15-19.

## Consumers
`solo/qwen38-flash-next-recipe` v2 (recipe.yaml `image:`), `recipes/qwen38-flash-next` (as the local `mbx-qwen38-flash-next`
build of the same Dockerfile).

## Publish (from ai1, where the image is)
```
docker tag mbx-qwen38-flash-next-solo-test myllmbox/qwen38-flash-next-vllm:v2
docker push myllmbox/qwen38-flash-next-vllm:v2          # base layers already on the Hub; only the patch layer uploads
docker image inspect myllmbox/qwen38-flash-next-vllm:v2 --format '{{index .RepoDigests}}'
```
