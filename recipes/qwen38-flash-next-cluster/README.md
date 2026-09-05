# qwen38-flash-next-cluster — the champion, TP=2 over RDMA

The published `qwen38-flash-next` stack (hibrid46 checkpoint + `myllmbox/qwen38-flash-next-vllm:v1`)
served across both Sparks instead of one. Everything the world gets, plus:

- **`cluster:`** — TP=2 over the ConnectX link. The runner opens `/dev/infiniband` + IPC_LOCK + memlock
  (2026-09-04 fix) so NCCL runs RDMA, not TCP. Prove it on boot: `port_xmit_data` under
  `/sys/class/infiniband/rocep1s0f1/ports/1/counters/` must move during decode.
- **`docker/patches/01-ple-offload-multinode.py`** — upstream fences the PLE CPU-offload worker to one node;
  the patch runs one full-table worker per node under `MBX_PLE_MULTINODE=1`. Ported from the 2026-08-29
  `experiments/qwen38-flash-next-cluster-vllm` boot (NVFP4 + bf16 table, TCP), which proved the path.

## Why
Solo champion = 44 tok/s c=1 at 12.9 steps/s (K=3). The per-step floor is the dense weight traffic;
TP=2 halves it per node, and RDMA (not TCP) is what made the vision lane's TP=2 step go 118 → 71 ms.
The question this lane answers: how many steps/s does the champion do on two boxes with a real interconnect.

## Run
1. Both boxes free (takes the cluster; the vision serve must be down — user decides).
2. `./run.sh qwen38-flash-next-cluster` — builds the image from this folder, ships it to box2, rsyncs the
   95G checkpoint to box2 (one-time), launches head + headless worker.
3. Verify a real completion via the proxy, then read steps/s + accept len off the engine logs.
4. Gauntlet: `tests/pasture.html` + `tests/fish.html` (best of 3, user-generated) before quoting numbers.

## Measured (boot 2: K=4, max-num-seqs 64, kv 46G — 65 runs, 1,994 steady windows, pasture, both bands)

| concurrent requests | AVERAGE tok/s (min–max) | AVERAGE per-stream | code tok/s | code per-stream | thinking tok/s | thinking per-stream | engine steps/s | acceptance |
|---|---|---|---|---|---|---|---|---|
| 1 | **59** (34–77) | **59.2** | 68.4 | 68.4 | 50.1 | 50.1 | 16.4 (15.3–17.2) | 3.60 (2.19–4.62) |
| 2 | **99** (61–123) | **49.7** | 118.3 | 59.1 | 80.4 | 40.2 | 13.8 (12.6–14.4) | 3.59 (2.33–4.44) |
| 4 | **153** (103–195) | **38.2** | 184.7 | 46.2 | 121.3 | 30.3 | 10.7 (9.7–11.3) | 3.56 (2.53–4.48) |
| 8 | **224** (157–290) | **28.1** | 279.8 | 35.0 | 169.1 | 21.1 | 8.0 (7.3–8.7) | 3.47 (2.55–4.36) |
| 16 | **322** (229–416) | **20.1** | 395.7 | 24.7 | 247.8 | 15.5 | 5.8 (5.3–6.2) | 3.47 (2.55–4.48) |
| 24 | **366** (265–475) | **15.3** | 450.2 | 18.8 | 282.3 | 11.8 | 4.4 (4.2–4.6) | 3.47 (2.54–4.50) |
| 32 | **407** (293–522) | **12.7** | 501.7 | 15.7 | 312.7 | 9.8 | 3.7 (3.4–3.9) | 3.45 (2.56–4.37) |
| 48 | **493** (364–645) | **10.3** | 600.8 | 12.5 | 384.3 | 8.0 | 3.0 (2.7–3.2) | 3.45 (2.49–4.41) |
| 52 | **509** (375–661) | **9.8** | 620.2 | 11.9 | 398.3 | 7.7 | 2.8 (2.6–3.0) | 3.46 (2.58–4.41) |
| 64 | **540** (346–721) | **8.4** | 666.5 | 10.4 | 413.2 | 6.5 | 2.4 (2.1–2.7) | 3.45 (2.55–4.42) |

AVERAGE = (code avg + thinking avg) / 2, range = extremes of either band; steps/s and acceptance pooled the same way.
Engine speed is band-independent; acceptance is the only difference (4.2 code vs 2.7–3.1 thinking) and it holds at
3.45–3.60 average on every rung. Batch-size step between 52 and 56 sequences (56/60 pay a 64-sized step). Full
ladder, all 21 rungs, and the run log: `reports.md`. Regenerate: `./bench/summary.py --model Qwen/Qwen3.8-Flash-Next --prompt pasture`.

## Knobs left for after the first boot
`gpu-memory-utilization` 0.70 is the cold-boot profile; `kv-cache-memory` is unpinned until the TP=2 fit is
read off a real boot. Both are config decisions, recorded in `reports.md` when measured.
