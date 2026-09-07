# qwen38-flash-next — runs (v2: hibrid47 on ONE Spark, 2026-09-07)

Lab history from recipes/qwen38-flash-next-solo-test (gitignored lane); boot 10 is the image that shipped as v2 (mbx-qwen38-flash-next-solo-test d2d370a5, built 13:24).

## Boots
| # | table | kv pin | outcome |
|---|---|---|---|
| 1 | resident (patch 03) | 8 GB | 99 GiB weights; OOM in FlashInfer autotune before KV — autotune+profile peak ≈ +34 GB over weights at TP=1 |
| 2 | mmap (patch 04, GPU faults) | 8 GB | 73.3 GiB weights, 112 GiB free at profile; vLLM refused: one 262k request needs 7.57 GiB > 7.45 pool |
| 3 | mmap | 8.5 GB | killed by hand during capture (box thrashed: autotune workspaces + 4.8 GB swap from boot 1) |
| 4 | mmap+dual (patch 05, nvme during boot) | 8.5 GB | KeyError in the eager op during capture: the ids piece is recorded, not run → junk ids. Fixed: clamp |
| 5 | same | 8.5 GB | "Cannot copy CPU↔CUDA during capture" — this runner RECORDS splitting ops into the piecewise graph. Fixed: mmap kernels under capture |
| 6 | same, MODE=nvme | 8.5 GB | UP. c=1: run 1 39 (21–49, cold rows), runs 2–4 48/48/49 tok/s, steps 13.4–13.9, acc 3.5 (K=3). Flip by flag → 50 tok/s |
| 7 | MODE=auto PREWARM=auto + heap trim | 7.0 GB / 200k | UP. auto flip ✓, trim 3.57→3.57 GiB (no glibc heap to give), populate 26.82 GiB in 25.6 s → 19.9 GB stayed; 2nd populate → 27.0 GB resident but 6 GB of heaps swapped out |
| 8 | + limit-mm-per-prompt image=0 (text-only) | 7.0 GB / 200k | text-only loads 72.45 GiB (vision tower ≈ 0.85 GiB) but the AOT compile cache was built with mm inputs → `NoneType.size` in the compiled forward. Vision is a must anyway (user) — dropped |
| 9 | vision on, MODE=auto, PREWARM=0 | 7.0 GB / 200k | UP (launched by a command the user had rejected — it had already run). KV 217,808 tok. Populate by trigger: pass 1 kept 20.7 of 26.8 GB (kswapd stole 17 GiB in 25 s, free floor ~7 GB); pass 2 never ran (watcher one-shot bug → fixed, image 40dc2573). Table then refilled itself on demand to ~25 GB cached, free 2 GB, swap 0. c=1: 50/51 tok/s, steps 14.1–14.4, acc 3.5 |
| 10 | + patch 06 fp8 KV (PR #54846 port), max-model-len back to 262144 | 7.0 GB fp8 | UP 13:41. KV **391,943 tokens** (1.80× bf16's 217,808 on the same pin; attention block 1600 → 3200), 1.50× max-length requests at 262k; weights 73.3 GiB, free 15 GB, swap 0, health 200. Output/quality/c=1 checks pending (user) |

## Memory on the solo box (sampler /tmp/boot-sampler.log, 5 s)
- Steady serve, hot-set policy: GPU process 85 GB (73.3 weights + 7.9 pool + 0.5 graphs + misc), host anon 7.0 GB
  (API 2.8, worker 2.8, engine 1.2), table pages mapped 19 GB after 4×c=1, free 8 GB, swap 0. Steps 14.3–14.5/s flat.
- The free floor: ~6–8 GB the kernel keeps FREE but fragmented (buddyinfo: everything ≤128 KB blocks, one 2 MB block on
  the box). 31k direct-compaction stalls this uptime; kswapd woken at the low watermark 953× (16 clean). High-order
  (2 MB) allocations — GPU driver / hugepage-hinted CUDA host buffers — make the kernel reclaim file+anon pages to form
  blocks it then fragments again. Populate-all into that = 7 GB of table evicted (pass 1) or 6 GB of heaps swapped (pass 2).
- Arithmetic: 84 GPU + 27 table + 7 anon + 2 kernel + 6 floor = 126 > 119.7. Full residency needs ~6 GB from:
  vm.watermark_boost_factor=0 (root), fp8 KV (−3 GB, PR 54846 port), text-only serve (vision tower + processor out).
- NVMe during decode with the hot set warm: ~10 major faults/s ≈ 1 % of lookups, 3 % drive busy — not a factor.

## Speed vs prediction
Solo hibrid47 c=1 = 48–50 tok/s at 72–75 ms/step (K=3), below the int3 solo kit's 64 ms/step (55 tok/s) and the
cluster's 56 ms. Suspects: the splitting-op graph break (A/B: same image with the op out of _attention_ops), then physics.
Not disk, not faults (measured).

## int3 solo kit vs hibrid47 solo — the like-for-like (c=1, pasture, K=3)
| | int3 kit (README) | hibrid47 solo boot 9 |
|---|---|---|
| sustained tok/s | 44 | 50–51 |
| peak window | 55 @ acc 4.0 | 54 @ acc 3.85 |
| steps/s | ~13.75 | 14.1–14.4 (70 ms/step, flat ±1 %) |
The "55" is the int3 kit's best window at the K=3 acceptance ceiling; sustained it was 44. The single-GPU step is ~70 ms
for this model either way (one GPU reads all the dense bytes; the cluster's 56 ms halves them per GPU).

## fp8 KV on QSA — PR #54846 port (patch 06, 2026-09-07)
Tony's overlay = upstream PR #54846 (andreasgru) on nightly 8a728663. Applied to our build as a rename-substituted diff:
18/21 hunks mechanical, 3 by hand (multi-line raises). Our ops/qsa.py keeps its extra indexer kernels (untouched by the PR).
One missing symbol (fa_utils.reshape_and_cache_flash, nvfp4 write branch only) → fallback to _custom_ops. The PR's 13
tests pass on the GB10 (fp8 vs dequantized reference, tile profiles, nvfp4 layout+reference). Expect ~1.59× pool tokens.

## Boot 10 — the v2 serve (image d2d370a5, fp8 KV, PREWARM=0 + manual populate)
- 13:41 UP: weights 73.3 GiB (686 s), KV 391,943 tokens fp8 (attention block 3200), graphs 0.50 GiB, flip after 38 eager
  steps (492 ms of nvme reads), heap trim 3.57 → 3.56 GiB. free 15 G / avail 26 G, swap 0 before populate.
- populate (trigger): 27 GB mapped after the second pass; 5.7 GB of cold heaps swapped out, swap-ins ≈ 0 afterwards;
  NVMe 4 reads / 10 s while decoding.
- c=1 pasture thinking off (13:57–14:05, 4 runs): 49.0–51.0 avg, 52.7–54.2 peak, steps 14.40–14.46.
- c=1 pasture thinking on (14:05–15:12, 4 full runs, 30k tokens each, 12–14 min): 42.3 / 39.2 / 39.9 / 39.1 avg,
  peaks 52–56, steps 14.34–14.42; the 14:05 run (71 windows) = experiments/data/qwen38-solo-hibrid47-card.{html,png}:
  thinking phase 39 avg (acc 2.5), code phase 51 avg (acc 3.5–3.9), steps flat 14.0–14.6.
- c=4 pasture thinking off (17:03, 130 s steady, 13 samples): 128.7 avg (123.2–133.2), per-stream 32.2, steps 9.3 (9.0–9.6,
  108 ms/step), acc 3.47, P(pos) 0.91/0.82/0.74, kv 48 % with 4 running (~12 % of the pool per request = the GDN floor).
- earlier the same day (boot 9, bf16 KV): c=1 warm runs 48.4–51.1 avg / 52.5–54.6 peak, steps 13.9–14.5;
  c=4 124.0 avg (103–137), steps 8.86.

## Open
- MADV_RANDOM on the table mappings (patch 04): expect one populate pass with no compaction storm and no heap swap.
- No-split A/B (op out of `_attention_ops`) for the 70 ms step; K=4 vs K=3; max-res image at 8192 batched tokens.
- Gauntlet tests/ for v2 (user-generated).
