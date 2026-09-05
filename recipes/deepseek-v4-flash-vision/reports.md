# deepseek-v4-flash-vision — benchmark report

Image `mbx-deepseek-v4-flash-vision` (official vision image + 5 patches, see Dockerfile). 2× DGX Spark GB10,
TP=2 over ConnectX RoCE. DSpark k=5 probabilistic + adaptive verification, fp8 KV, block 256, 300k ctx,
util 0.85 + `kv-cache-memory` 18.5 GiB, FULL_DECODE_ONLY graphs [1..48], batched 8192, thinking ON.

## Config facts (boot 3, 2026-09-04)
- KV pool **757,715 tokens** (2.53× 300k). Weights + non-torch 77.7 GiB/rank, graphs 3.35 GiB, engine init 99 s warm.
- Host `free` after boot: 117 G used / 1–2 G available (UMA; watch item).

## Runs (user-driven, read off engine logs; thinking ON, real prompts)

| date | boot | c | content | gen tok/s | steps/s (ms) | mean acc len | draft accept % | notes |
|------|------|---|---------|-----------|--------------|--------------|----------------|-------|
| 2026-09-04 | 2 (TCP) | 1 | reasoning prose | 25–28 | 8.5 (118) | 3.0–3.4 | 40–47 | container had NO /dev/infiniband → NCCL on TCP sockets; RDMA port counter 0 |
| 2026-09-04 | 3 (RDMA) | 1 | reasoning prose | **46–49** | 14 (71) | 3.3–3.5 | 45–50 | runner fix: --device /dev/infiniband + IPC_LOCK + memlock; RDMA 125 MB/s during decode |
| 2026-09-04 | 3 (RDMA) | 1 | mixed incl. image requests (MM cache hits 50–75%) | 35–45 (195 windows avg 39, peak 58) | 14 | 2.4–3.7 | 28–53 | acceptance is content-driven; image/agent prompts sit at the low end |
| 2026-09-04 | 3 (RDMA) | 2 | mixed | **61 agg** (peak 70) / 30 per req | | | | 36 windows |
| 2026-09-04 | 3 (RDMA) | 4 | structured (accept 4.4–4.8!) | **75–86 agg** / 19 per req | ~4.5 (220) | 4.4–4.8 | 67–76 | 78 windows, KV 4–5%; step cost grows with batch (more distinct experts per step) |
| 2026-09-04 | 3 (RDMA) | 5 | mixed | 91 agg (peak 107) / 18 per req | | | | 25 windows |
| 2026-09-04 | 3 (RDMA) | 7 | mixed | 113 agg (peak 128) / 16 per req | | | | 48 windows |
| 2026-09-04 | 3 (RDMA) | 8 | mixed | **118 agg (peak 140)** / 15 per req | ~3.3 (300) | | | 61 windows; robot's c8 (thinking off) = 93; our 0731 B12X lane on TCP = ~95–105 |

## Notes
- The whole 2× came from the transport: acceptance unchanged, per-step time 118 → 71 ms. Every earlier 2-Spark
  number in this repo (0731 lane "~50 ceiling", c-ladders) predates the runner fix = TCP numbers; re-measure.
- Per-position acceptance 0.93 / 0.75 / 0.55 / 0.08 / 0.06 → positions 4–5 nearly dead on prose. k=3 vs k=5
  (adaptive) is the next A/B; k must divide dspark_block_size=5, so the candidates are 5 and 1 unless the
  n_predict normalisation is relaxed — check before assuming k=3 boots.
- Robot's published (thinking OFF, same image family): SHORT 43 / PROSE 35 / c8 93. We are above at every rung
  with thinking ON: c1 46–49 · c2 61 · c4 75–86 · c8 118 (peak 140). Ladder is monotonic, no wall through c8.
- Scaling shape: per-step time grows ~71 → ~220 → ~300 ms from c1 → c4 → c8 (MoE: more tokens per step touch
  more distinct experts), so per-request speed halves by c4 (19) and settles ~15 at c8 while aggregate keeps rising.
- Gauntlet (`tests/pasture.html`, `tests/fish.html`) pending.
