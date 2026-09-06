# hf.co/myllmbox/Qwen3.8-Flash-Next-hibrid47 — the NVFP4-table checkpoint

**Built:** 2026-09-06 (as `…-hibrid46-ple4`, renamed hibrid47 on release). **Size:** 98.9 GiB, 27 shard files.
**Source:** hibrid46 (body tensors, hardlinked = byte-identical) + the bf16 PLE table from hibrid46-off (128 shards of
[2,500,012 × 160] = 320,001,536 rows), re-quantized to NVFP4 by `docker/make-hibrid47.py` (two streaming passes:
global amax → per-16 block fp8 scale, codes nearest-to-EFFECTIVE scale; ~200 MB working set; ran next to a serve).
**Table on disk:** 8 shards `ple-nvfp4-0000k-of-00008.safetensors`, each `…ngram_embedding.nvfp4_shard_k.packed`
U8 [40,000,192 × 80] + `.scales` F8_E4M3 [40,000,192 × 10]; `nvfp4_global` F32 (3.3242e-05) in shard 1.
23.8 GiB packed + 4.8 GiB scales = 28.6 GiB (int3 was 17.9; bf16 95.4). **config.json:**
`ple_quantization = {"format":"nvfp4","block":16,"rows":320001536,"dim":160,"shards":8,"shard_rows":40000192}`.
Drafter (MTP) experts stay NVFP4 W4A4 (Inferact's) — the bf16-drafter A/B was null (recipes/qwen38-flash-next-mtp-test/reports.md).

## Why a new table format
The int3 table needed the CPU-offload gather (a per-step ZeroMQ → CPU gather → pinned buffer → DMA detour). NVFP4 is a
standard layout the GPU can dequantize in-forward, so the table becomes a resident GPU parameter (patch
`cluster/docker/patches/03-ple-gpu-nvfp4.py`: `_MbxNvfp4EmbeddingMethod`; sharded over TP by default, or full copy per
rank with `MBX_PLE_REPLICATE=1`). No CPU worker, no IPC, inside the CUDA graphs.

## Measured (2 × DGX Spark, TP=2 RDMA, replicated table, kv 25 G, K=4, vm.compaction_proactiveness=0, pasture, thinking off)
| c | steps/s (published int3 → hibrid47) | gen tok/s avg (published → now) | steady peak |
|---|---|---|---|
| 1 | 16.4 → 17.7 | 68.4 → 73–76 (7 runs) | **80.3** |
| 2 | 13.8 → 15.1 | 118 → 126 | 133 |
| 4 | 10.7 → 11.8 | 185 → 198 | 209 |
| 8 | 8.0 → 8.8 | 280 → 294 | 309 |
| 16 | 5.8 → 6.2 | 396 → 417 | 451 |
| 24 | 4.4 → 4.9 | 450 → 488 | 514 |
| 32 | 3.7 → 4.0 | 502 → 533 (pasture) · 517 (fish, 7 runs) | 579 |
Acceptance unchanged by the table on pasture (4.2–4.3) and fish (4.0). Thinking c=32 ≈ 322–340 tok/s at acc 2.5.
Half of the step gain is the resident table, half is the compaction fix that let it show (kcompactd memory).

## Quality (the deciding result)
User's visual gauntlet, 32 × boss-animals at c=32, thinking on, 2026-09-06 13:05–16:33: **26 good/super, 3 partial,
3 broken**. The int3 table scored ≈ 16/16 on the same set. logprob (pasture+fish texts): hibrid47 pooled −2.1138 (ppl 8.28).

## Fit
2 Sparks: 64.9 GiB/box replicated (kv 25 G → 49 seats) or ~50.5 GiB/box sharded (kv 40–46 G → 80+ seats; same steps at c=1).
1 Spark: ~102 GiB → ~9 GB for KV → c≤8 box; the int3 kit stays the single-Spark release until the int3-on-GPU path lands.

## Files
`docker/make-hibrid47.py` (converter) · `cluster/docker/patches/03-ple-gpu-nvfp4.py` (engine patch, next image) ·
`hibrid47-README.md` (the HF model card as uploaded).
