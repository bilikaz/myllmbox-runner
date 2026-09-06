# qwen38-flash-next-cluster — the champion, TP=2 over RDMA (v2: hibrid47, table on the GPU)

The published stack served across both Sparks. **v2 (2026-09-06)** = the
[hibrid47](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid47) checkpoint: hibrid46's body plus the
95 GB n-gram (PLE) table re-quantized to NVFP4 and held **resident on the GPU** (patch 03 in `docker/`), gathered
inside the model's forward pass. No CPU offload worker, no per-step IPC detour. Plus:

- **`cluster:`** — TP=2 over the ConnectX link; the runner opens `/dev/infiniband` + IPC_LOCK + memlock so NCCL runs
  RDMA. Prove it on boot: `port_xmit_data` under `/sys/class/infiniband/rocep1s0f1/ports/1/counters/` moves during decode.
- **`docker/patches/03-ple-gpu-nvfp4.py`** — the table as an ordinary GPU parameter, row-sharded over TP (default) or a
  full copy per box (`MBX_PLE_REPLICATE=1`). `docker/make-hibrid47.py` is the converter that made the checkpoint.
- **`docker/patches/01`, `02`** — the int3/offload fallback path (inert without `MBX_PLE_MULTINODE`) and loader
  page-cache hygiene.

## Run
1. Both boxes free. `vm.compaction_proactiveness=0` on both (see below).
2. `./run.sh qwen38-flash-next-cluster` — builds the image from this folder, ships it to box2, rsyncs the checkpoint
   (98.9 GiB; 19 of 27 shards are hibrid46's — hardlink them if hibrid46 is already on the box), launches head + worker.
3. Verify a real completion via the proxy, then read steps/s + acceptance off the engine log
   (`./bench/accept.py --since <HH:MM> --host <box1>` gives the per-position picture).
4. Gauntlet before quoting numbers: `tests/pasture.html` + `tests/fish.html`, best of 3, user-generated.

## Measured — v2 (hibrid47, replicated table, kv 25 G · c=48 on the 28 G boot, K=4, `vm.compaction_proactiveness=0`, pasture, thinking off)
Boot 2026-09-06 10:46 (fresh reboot). Each row = 3–7 independent runs of steady 10 s windows; peak = best steady window.

| concurrent | engine steps/s (v1 → v2) | tok/s avg (v1 → v2) | **v2 peak** | per stream | acceptance |
|---|---|---|---|---|---|
| 1 | 16.4 → **17.7** (17.5–18.1) | 68.4 → 73 | **80** | 73 | 4.1 (3.9–4.3) |
| 2 | 13.8 → **15.1** | 118.3 → 126 | **133** | 63 | 4.15 |
| 4 | 10.7 → **11.8** | 184.7 → 198 | **209** | 50 | 4.2 |
| 8 | 8.0 → **8.8** | 279.8 → 294 | **309** | 37 | 4.16 |
| 16 | 5.8 → **6.2** | 395.7 → 417 | **451** | 26 | 4.2 |
| 24 | 4.4 → **4.9** | 450.2 → 488 | **514** | 20 | 4.18 |
| 32 | 3.7 → **4.0** | 501.7 → 533 | **579** | 17 | 4.19 |
| 48 | 3.0 → **3.2** | 493 → 635 | **674** | 13.2 | 4.18 |

Steps +7–11 % on every rung; the old averages are the new floors. Fish prompt at c=32: 517 avg over 7 runs
(acceptance 4.0). Thinking at c=32: 320–340 tok/s at acceptance 2.5 — same steps, the text decides the rest.
Per-position acceptance on pasture, c=1: P(draft 1..4 accepted) = 0.91 / 0.85 / 0.80 / 0.71.
c=48 (this yaml: replicated, 28 G, 19:52, 400 s hold): steps flat 3.1–3.3 over 38 bins, no dip; the pool stood at
98.9 % when the hold ended — 48 is the seat ceiling of the 28 G pin, not a graph bucket below it.

**Quality (the deciding result).** 32 boss-animals renders at c=32, thinking on, user's visual gauntlet:
**26 good/super · 3 partial · 3 broken.** The int3 table (v1) scored about half/half on the same scenes.
logprob (pasture + fish texts): pooled −2.114, ppl 8.28. Drafter experts stay NVFP4: a bf16-drafter A/B was null
(acceptance +0.01, −3 % steps, +3.4 GB — `recipes/qwen38-flash-next-mtp-test/reports.md`).

**Where the gain comes from.** Half is the resident table (the offload detour gone), half is the compaction fix
below that let it show: the same design with the compactor running measured 3.7 steps at c=32, identical to v1.

## v1 reference — the int3 / CPU-offload stack (hibrid46), kept for comparison
Tagged and reproducible: kit repo `qwen38-flash-next-cluster-recipe` **tag `v1`** (`git checkout v1`), image
`myllmbox/qwen38-flash-next-cluster-vllm:v2` (digest in `builds/qwen38-flash-next/cluster-v2.md`), checkpoint
[hf.co/myllmbox/Qwen3.8-Flash-Next-hibrid46](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid46).
Boot 2026-09-05 (K=4, max-num-seqs 64, kv 46 G, 20 GB free — the compactor never woke), 65 runs, 1,994 steady windows:

| concurrent requests | **PEAK tok/s** | average tok/s | average per-stream | code tok/s (min–max) | thinking tok/s (min–max) | engine steps/s | acceptance |
|---|---|---|---|---|---|---|---|
| 1 | **77** | 59 | 59.2 | 68.4 (54.7–77.1) | 50.1 (34.1–71.7) | 16.4 (15.3–17.2) | 3.60 (2.19–4.62) |
| 2 | **123** | 99 | 49.7 | 118.3 (109.9–123.4) | 80.4 (61.4–107.7) | 13.8 (12.6–14.4) | 3.59 (2.33–4.44) |
| 4 | **195** | 153 | 38.2 | 184.7 (172.3–195.2) | 121.3 (103.4–156.6) | 10.7 (9.7–11.3) | 3.56 (2.53–4.48) |
| 8 | **290** | 224 | 28.1 | 279.8 (258.9–289.9) | 169.1 (157.1–194.5) | 8.0 (7.3–8.7) | 3.47 (2.55–4.36) |
| 16 | **416** | 322 | 20.1 | 395.7 (377.1–415.8) | 247.8 (229.2–267.1) | 5.8 (5.3–6.2) | 3.47 (2.55–4.48) |
| 24 | **475** | 366 | 15.3 | 450.2 (403.1–474.9) | 282.3 (265.1–296.2) | 4.4 (4.2–4.6) | 3.47 (2.54–4.50) |
| 32 | **522** | 407 | 12.7 | 501.7 (473.9–522.4) | 312.7 (292.6–333.6) | 3.7 (3.4–3.9) | 3.45 (2.56–4.37) |
| 48 | **645** | 493 | 10.3 | 600.8 (560.3–644.8) | 384.3 (364.1–423.1) | 3.0 (2.7–3.2) | 3.45 (2.49–4.41) |
| 52 | **661** | 509 | 9.8 | 620.2 (567.2–660.6) | 398.3 (375.1–466.4) | 2.8 (2.6–3.0) | 3.46 (2.58–4.41) |
| 64 | **721** | 540 | 8.4 | 666.5 (598.4–721.3) | 413.2 (346.5–494.9) | 2.4 (2.1–2.7) | 3.45 (2.55–4.42) |

AVERAGE = (code avg + thinking avg) / 2, range = extremes of either band. Full v1 ladder (21 rungs) and run log:
`reports.md`. Regenerate any lane: `./bench/summary.py --model <folder> --thinking both`.

## Knobs (v2)
- `MBX_PLE_REPLICATE: "1"` + `kv-cache-memory` **28 G** (1.71M pooled tokens, 48 seats fill it, `max-num-seqs` 48): the full
  table on each box, the layout every v2 number was measured on (rungs 1–32 at 25 G, c=48 at 28 G). First boot at 28 G:
  ~5 GB available per box after capture, box2 holds 3 GB in swap — but swap-ins stayed flat through the c=1 and c=48
  holds (load leftovers, never touched). 25 G is the fallback if swap-ins ever climb.
- Unset `MBX_PLE_REPLICATE` for half the table per box: 14 GiB freed → a 40 G pin (~80 seats) at c=1 steps within 1 %
  of replicated (measured with the compactor on). Not yet measured at c=32 with the compactor off — do that before
  publishing numbers from it.
- `gpu-memory-utilization` **0.70**: with the pin set it does not size KV and the start-up check does not apply it
  to weights + pin (65 GiB + 28 G booted at 0.70).
- `OMP_NUM_THREADS: "8"` and the X925 cpuset are inherited from the offload era; the CPU gather is gone, so an
  unpinned / 1-thread A/B at c=1 is the cheap follow-up.

## Box tuning that is not in the yaml
`vm.compaction_proactiveness=0` on every box (`cluster/server-profile.sh` sets it; persisted in
`/etc/sysctl.d/99-myllmbox-compaction.conf`). Found 2026-09-06: at 5–7 GB free the kernel's proactive compactor
migrated GPU-mapped pages every ~37 s → 4–5 s at −30 % step rate, both ranks idling, no swap, clocks flat. Same
boot, same test, sysctl applied live: c=32 steps 3.7 → 4.1–4.3, floor 397 → 488. The v1 boot (20 GB free) never
triggered it, which is why its ladder is clean; any tighter pin on the same boxes would not have been.
