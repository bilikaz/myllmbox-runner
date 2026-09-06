# docker.io/myllmbox/qwen38-flash-next-cluster-vllm:v3

**Published:** 2026-09-06 (pushed from ai1; base layers already on the Hub, only the patch layer uploaded).
**Digest:** `sha256:ba28f473c766919afac75a898014d1dbe18a0929a7f55c0588512f27ee365513`
**Image id (local, = mbx-qwen38-flash-next-cluster):** `sha256:935f15138bc6f162e2d9975536082cce4002523c8816dd819c8ae188e04eee0c`
**Reason for a new tag:** adds patch 03 — the NVFP4 PLE table (hibrid47) as a resident GPU parameter — and ships the
converter (`/opt/mbx/make-hibrid47.py`). v1/v2 stay as published (int3 / CPU-offload stack).

## What changed vs v2
`cluster/docker/patches/03-ple-gpu-nvfp4.py` on the vendor's `ple_layer.py`: when `config.json` declares
`ple_quantization.format == "nvfp4"` and `VLLM_PLE_CPU_OFFLOAD` is off, `_MbxNvfp4EmbeddingMethod` allocates the packed
table for the rank's vocab partition (uint8 codes + fp8 block scales + fp32 global), the loader routes the 8
`ple-nvfp4-*` shards by row range, and `embedding()` gathers + dequantizes in-forward (index_select → nibble unpack →
e2m1 LUT → × block scale × global). `MBX_PLE_REPLICATE=1` = `VocabParallelEmbedding(disable_tp=True)`: the full table
on every rank, no all-reduce. Sanity in the Dockerfile round-trips a synthetic row through the converter's `quantize()`
and the method's gather (rel err 0.095) and checks the [tokens,16]→[tokens,16,D] contract.
Patches 01 (multi-node offload gate) and 02 (loader page-cache drop) unchanged from v2.

## Measured with it (recipes/qwen38-flash-next-cluster v2, hibrid47, 2× Spark, K=4, vm.compaction_proactiveness=0)
c=1 17.7 steps/s, 73–76 tok/s avg, 80 peak · c=32 4.0 steps/s, 533 avg (pasture) / 517 (fish), 579 peak · boss-animals
gauntlet 26/32 (int3: ~16/32). Full ladder + conditions: recipes/qwen38-flash-next-cluster/README.md, reports.md.

## Consumers
`solo/qwen38-flash-next-cluster-recipe` v2 (recipe.yaml `image:`), `recipes/qwen38-flash-next-cluster` (as the local
`mbx-qwen38-flash-next-cluster` build of the same Dockerfile).
