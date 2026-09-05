# qwen38-flash-next-cluster — runs

Baseline to beat: solo champion (`../qwen38-flash-next`), single Spark, K=3, thinking-off code band:
c=1 44 tok/s @ 12.9 steps/s · c=2 72 · c=4 103 · c=8 148-158 (peak 167.5 aggregate).

## Runs (user-driven, read off engine logs)

| date | boot | c | content | gen tok/s | steps/s (ms) | mean acc len | RDMA proven? | notes |
|------|------|---|---------|-----------|--------------|--------------|--------------|-------|

## Boot 1 — 2026-09-05 09:18 (first TP=2 boot, util 0.70, KV unpinned)
Per node (identical on box1/box2): body shard **38.05 GiB** · int3 PLE table 17.9 GiB in the per-node offload
worker (MBX_PLE_MULTINODE worker came up on BOTH boxes) · available KV **40.2 GiB** → **2,700,791 tokens**,
10.3 full 262k seats. torch.compile 18 s + 3 s (fresh TP=2 shapes), FlashInfer autotune 42 configs saved.
Host mem after load: box1 97G used / 22G available, box2 93G / 25G. Weights: 306 s head (rsync'd ckpt, cold
page cache) / 166 s worker. RDMA port_xmit_data baselines before first request: box1 484428305892 ·
box2 491024377791 (×4 = bytes; must move during decode).
**RDMA PROVEN 09:28:** healthy (API 200 on head). 5 s sample during traffic: box1 RDMA xmit +482 MB / rcv +482 MB,
box2 xmit +503 MB / rcv +503 MB (≈100 MB/s each way), ConnectX netdev tx **0 MB** → NCCL is on verbs, not TCP.
Container has /dev/infiniband, CAP_IPC_LOCK, memlock=-1 on both nodes.

| date | boot | c | content | gen tok/s | steps/s (ms) | mean acc len | RDMA proven? | notes |
|------|------|---|---------|-----------|--------------|--------------|--------------|-------|
| 2026-09-05 09:27-09:29 | 1 | 1 | thinking prose, FIRST request (Triton JIT firing on box2) | 34-42 | 14.5-16.9 (59-69) | 2.2-2.6 | yes | cold window |
| 2026-09-05 09:32-09:33 | 1 | 1 | thinking prose, warm (0 JIT warnings) | 39-50 | **17-18 (56-59)** | 2.3-2.8 | yes | drafted/3 = 16.9-18.0 every window; solo = 12.9 → **+35-40% steps** |
| 2026-09-05 09:34:02 | 1 | 1 | prose, high-accept window (per-pos 0.88/0.79/0.69) | **58.1** | 17.3 (58) | 3.36 | yes | user-reported window; K=3 ceiling = acc 4.0 → 18 steps caps at ~72 tok/s |

## Next (user decision 2026-09-05): K=4 once boot 1 proves steady
Engine speed is flat at 17-18 steps/s regardless of content; tok/s is the acceptance dial and K=3 caps acceptance
at 4.0 (≈72 tok/s ceiling). K=4 → cap 5.0. Geometry OK (ring K+4 = 8 divides block 880). Judge by steps/s =
drafted ÷ 4 (expect a small dip) and code-band acceptance > 4.0. Config change: speculative-config
num_speculative_tokens 3 → 4, restart. Also optional A/B after: PLE offload OFF (int3 table GPU-resident, ~9G/node).
| 2026-09-05 09:37:32 | 1 | 1 | STILL THINKING (not code yet) — high-accept reasoning (per-pos 0.94/0.87/0.77, draft accept 85.9%) | **64.4** | 18.0 (56) | 3.58 | yes | user-reported; position 3 still 0.77 → K=4 has headroom; code band pending |
| 2026-09-05 09:30-09:41 | 1 | 1 | **AVERAGE, 64 windows, first 3 min excluded** (thinking/prose, no code yet) | **46.2** (34-64) | **17.6 (57)** min 16.1 max 18.3 | 2.61 | yes | all windows c=1; steps/s spread ±6% while tok/s spread ±33% → acceptance is the only variable |
| 2026-09-05 09:41:32-52 | 1 | 1 | **CODE band** (started 09:41; per-pos 0.99/0.99/0.97 in the top window) | **62.6-71.8** | 17.9-18.2 (55) | 3.50-3.95 | yes | first code windows; 71.8 @ acc 3.95 = the K=3 ceiling (4.0) reached → K=4 is the only way past ~73 |

### Boot 1 phase averages (c=1, thinking ON, K=3; 10-s engine windows)
| phase | windows | gen tok/s avg (min–max) | steps/s avg (min–max) | ms/step | acc len avg (min–max) |
|---|---|---|---|---|---|
| THINKING (boot+1 min → 09:41:12) | 76 | **44.8** (31.2–64.4) | 17.4 (15.3–18.3) | 57 | 2.57 (1.92–3.58) |
| CODE (09:41:12 → 09:43:22) | 14 | **65.5** (59.0–71.8) | 18.1 (17.7–18.5) | 55 | 3.62 (3.28–3.95) |
Solo reference: 12.9 steps/s · code c=1 44 tok/s → cluster = +35-40% steps, **+49% code tok/s**.

## Boot 2 — 2026-09-05 09:49 (K=4, max-num-seqs 64, kv-cache-memory 46G, util 0.70)
| date | boot | c | content | gen tok/s | steps/s (ms) | mean acc len | RDMA proven? | notes |
|------|------|---|---------|-----------|--------------|--------------|--------------|-------|
| 2026-09-05 13:03-13:06 | 2 | 8 | pasture, thinking OFF, test.py 190 s steady / 19 samples | **253.7** (222.7–276.2) agg · 31.7 (27.8–34.5) per stream | **7.5 (133)** engine | 4.23 (4.05–4.48) | boot-1 proof | solo c=8 = 148-158 → **+65%**; engine steps 7.5 vs solo ~5.5 (+36%, same as c=1) × acceptance 4.23 vs ~3.4 (K=4, +25%); kv 9.6%; 8/8 finished, avg 6781 tok / 224 s |
| 2026-09-05 ~13:10 | 2 | 32 | pasture, thinking OFF, test.py (15 steady samples 30–170 s, user-pasted) | **464** (436–492) agg · 14.5 (13.6–15.4) per stream | **3.4 (290)** engine (per-seq drafts 106–118 ÷ 32) | 4.25 (3.90–4.45) | boot-1 proof | solo c=32 max 305 → **+52%**; kv 35–38% at 32 reqs ≈ 1.15%/request (SSM state blocks, not context) → ~85 seats max |
| 2026-09-05 ~13:25 | 2 | 64 | pasture, thinking OFF, test.py (13 steady samples 40–160 s, user-pasted; fixed script → engine steps) | **626** (587–693) agg · 9.7 (9.2–10.8) per stream | **2.35 (425)** engine | 4.19 (3.86–4.35) | boot-1 proof | kv 70–75% at 64 reqs (1.14%/req → pool = ~87 seats); solo c=32 = 305 agg @ 9.5/stream → cluster c=64 = 626 @ 9.7/stream = **2× capacity at equal per-stream speed** |

### Boot 2 ladder (pasture, thinking OFF, K=4, test.py)
| c | aggregate tok/s | per stream | engine steps/s | ms/step | acc | vs solo |
|---|---|---|---|---|---|---|
| 1 | ~65 (code, boot 1 K=3) | 65 | 18.0 | 56 | 3.6 | 44 → +49% |
| 8 | 254 | 31.7 | 7.5 | 133 | 4.23 | 148–158 → +65% |
| 32 | 464 | 14.5 | 3.4 | 290 | 4.25 | 305 max → +52% |
| 64 | 626 | 9.7 | 2.35 | 425 | 4.19 | = solo c=32 per-stream at 2× the seats |
| 2026-09-05 14:25-14:28 | 2 | 16 | pasture, thinking OFF, test.py (18 steady samples / 180 s) | **395.7** (377–416) agg · 24.7 (23.6–26.0) per stream | **5.9 (168)** engine | 4.17 (3.94–4.48) | boot-1 proof | kv 19.2% (1.2%/req); 16/16 finished, avg 6982 tok / 281 s; steps spread ±3% |

### Boot 2 full ladder — bench/summary.py, 31 runs / 634 steady samples, 13:33–15:51 (pasture, thinking OFF, K=4)
Rungs 56/60/64 = single runs, reruns pending. Reproduce: `./bench/summary.py --thinking off --prompt pasture`
| c | runs | samples | gen tok/s avg (min–max) | per-stream | engine steps/s | ms/step | acc len | kv % | finished/aborted |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 30 | 68.4 (54.7–77.1) | 68.4 (54.7–77.1) | 16.7 (15.9–17.2) | 60 | 4.10 (3.24–4.62) | 1.3 (1.2–1.4) | 4/0 |
| 2 | 3 | 25 | 118.3 (109.9–123.4) | 59.1 (54.9–61.7) | 14.1 (13.7–14.4) | 71 | 4.19 (3.84–4.44) | 2.5 (2.4–2.7) | 6/0 |
| 3 | 2 | 26 | 127.5 (117.5–133.6) | 42.5 (39.2–44.5) | 10.2 (9.9–10.5) | 98 | 4.17 (3.77–4.44) | 3.7 (3.4–4.0) | 6/0 |
| 4 | 2 | 22 | 184.7 (172.3–195.2) | 46.2 (43.1–48.8) | 11.0 (10.6–11.3) | 91 | 4.19 (3.81–4.48) | 4.9 (4.6–5.4) | 8/0 |
| 5 | 2 | 23 | 214.0 (202.7–222.8) | 42.8 (40.5–44.6) | 10.2 (9.9–10.6) | 98 | 4.20 (3.85–4.47) | 6.2 (5.7–6.5) | 10/0 |
| 6 | 2 | 28 | 200.1 (186.1–209.2) | 33.4 (31.0–34.9) | 8.0 (7.6–8.4) | 125 | 4.17 (3.93–4.46) | 7.3 (6.9–7.8) | 12/0 |
| 8 | 2 | 28 | 279.8 (258.9–289.9) | 35.0 (32.4–36.2) | 8.4 (8.1–8.7) | 119 | 4.18 (3.76–4.36) | 9.7 (9.1–10.3) | 16/0 |
| 12 | 1 | 16 | 351.1 (329.8–370.8) | 29.3 (27.5–30.9) | 6.9 (6.5–7.2) | 144 | 4.22 (4.02–4.47) | 14.5 (13.7–15.5) | 12/0 |
| 16 | 1 | 18 | 395.7 (377.1–415.8) | 24.7 (23.6–26.0) | 5.9 (5.8–6.2) | 168 | 4.17 (3.94–4.48) | 19.1 (18.3–19.9) | 16/0 |
| 20 | 1 | 22 | 436.9 (421.8–457.0) | 21.8 (21.1–22.8) | 5.2 (5.0–5.4) | 193 | 4.21 (3.98–4.50) | 24.0 (22.9–25.6) | 20/0 |
| 24 | 1 | 30 | 450.2 (403.1–474.9) | 18.8 (16.8–19.8) | 4.5 (4.2–4.6) | 224 | 4.20 (3.95–4.50) | 29.0 (27.4–31.0) | 1/23 |
| 28 | 1 | 31 | 481.9 (436.8–509.9) | 17.2 (15.6–18.2) | 4.1 (3.9–4.3) | 244 | 4.20 (3.94–4.43) | 33.7 (32.0–36.2) | 1/27 |
| 32 | 1 | 34 | 501.7 (473.9–522.4) | 15.7 (14.8–16.3) | 3.7 (3.6–3.9) | 267 | 4.19 (3.97–4.37) | 38.5 (36.6–41.5) | 1/31 |
| 36 | 1 | 35 | 528.2 (503.6–565.8) | 14.7 (14.0–15.7) | 3.5 (3.3–3.7) | 287 | 4.21 (3.99–4.41) | 43.0 (39.4–46.4) | 1/35 |
| 40 | 1 | 32 | 535.1 (500.3–563.3) | 13.4 (12.5–14.1) | 3.2 (3.0–3.4) | 313 | 4.19 (4.00–4.35) | 47.3 (43.8–49.9) | 1/39 |
| 44 | 1 | 35 | 586.9 (512.7–630.5) | 13.3 (11.7–14.3) | 3.2 (2.8–3.4) | 314 | 4.18 (4.01–4.48) | 52.3 (48.1–54.7) | 1/43 |
| 48 | 1 | 40 | 600.8 (560.3–644.8) | 12.5 (11.7–13.4) | 3.0 (2.7–3.2) | 335 | 4.20 (4.03–4.41) | 57.1 (52.5–59.9) | 0/48 |
| 52 | 1 | 39 | 620.2 (567.2–660.6) | 11.9 (10.9–12.7) | 2.8 (2.6–3.0) | 352 | 4.20 (4.01–4.41) | 61.6 (56.9–64.6) | 1/51 |
| 56 | 1 | 40 | 578.9 (523.4–627.2) | 10.3 (9.3–11.2) | 2.5 (2.3–2.6) | 407 | 4.21 (3.79–4.48) | 65.8 (61.3–70.0) | 0/56 |
| 60 | 1 | 40 | 591.0 (545.5–643.0) | 9.9 (9.1–10.7) | 2.3 (2.2–2.6) | 426 | 4.20 (3.82–4.47) | 70.3 (65.6–75.0) | 0/60 |
| 64 | 1 | 40 | 665.4 (598.4–705.0) | 10.4 (9.3–11.0) | 2.5 (2.3–2.7) | 404 | 4.21 (3.81–4.42) | 75.2 (70.0–79.9) | 0/64 |

### Boot 2 — FULL LADDER, both bands (bench/summary.py, 65 runs / 1994 steady samples, 13:33–20:45; pasture; K=4)
Polluted c56/c60 runs (20:07–20:22) deleted; rungs 56/60 = 2 clean runs each. AVERAGE = (code avg + thinking avg)/2, MIN/MAX = extremes of either band. Regenerate: `./bench/summary.py --model Qwen/Qwen3.8-Flash-Next --prompt pasture`

| c | MAX tok/s | AVERAGE tok/s | MIN tok/s | AVERAGE /stream | code tok/s avg (min–max) | code /stream | thinking tok/s avg (min–max) | thinking /stream | steps/s avg (min–max) | acc avg (min–max) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 77 | **59** | 34 | **59.2** | 68.4 (54.7–77.1) | 68.4 | 50.1 (34.1–71.7) | 50.1 | 16.4 (15.3–17.2) | 3.60 (2.19–4.62) |
| 2 | 123 | **99** | 61 | **49.7** | 118.3 (109.9–123.4) | 59.1 | 80.4 (61.4–107.7) | 40.2 | 13.8 (12.6–14.4) | 3.59 (2.33–4.44) |
| 3 | 134 | **105** | 69 | **35.0** | 127.5 (117.5–133.6) | 42.5 | 82.7 (69.4–104.2) | 27.6 | 9.9 (8.9–10.5) | 3.53 (2.38–4.44) |
| 4 | 195 | **153** | 103 | **38.2** | 184.7 (172.3–195.2) | 46.2 | 121.3 (103.4–156.6) | 30.3 | 10.7 (9.7–11.3) | 3.56 (2.53–4.48) |
| 5 | 223 | **171** | 109 | **34.3** | 214.0 (202.7–222.8) | 42.8 | 128.6 (109.3–151.9) | 25.7 | 9.6 (8.4–10.6) | 3.52 (2.38–4.47) |
| 6 | 209 | **163** | 108 | **27.2** | 200.1 (186.1–209.2) | 33.4 | 125.8 (107.7–152.2) | 21.0 | 7.7 (7.0–8.4) | 3.49 (2.49–4.46) |
| 8 | 290 | **224** | 157 | **28.1** | 279.8 (258.9–289.9) | 35.0 | 169.1 (157.1–194.5) | 21.1 | 8.0 (7.3–8.7) | 3.47 (2.55–4.36) |
| 12 | 371 | **285** | 205 | **23.7** | 351.1 (329.8–370.8) | 29.3 | 218.3 (205.1–240.5) | 18.2 | 6.7 (6.2–7.2) | 3.52 (2.60–4.47) |
| 16 | 416 | **322** | 229 | **20.1** | 395.7 (377.1–415.8) | 24.7 | 247.8 (229.2–267.1) | 15.5 | 5.8 (5.3–6.2) | 3.47 (2.55–4.48) |
| 20 | 457 | **354** | 257 | **17.7** | 436.9 (421.8–457.0) | 21.8 | 271.3 (257.2–292.7) | 13.6 | 5.0 (4.7–5.4) | 3.49 (2.52–4.50) |
| 24 | 475 | **366** | 265 | **15.3** | 450.2 (403.1–474.9) | 18.8 | 282.3 (265.1–296.2) | 11.8 | 4.4 (4.2–4.6) | 3.47 (2.54–4.50) |
| 28 | 510 | **391** | 288 | **14.0** | 481.9 (436.8–509.9) | 17.2 | 301.1 (287.8–317.3) | 10.8 | 4.0 (3.8–4.3) | 3.46 (2.55–4.43) |
| 32 | 522 | **407** | 293 | **12.7** | 501.7 (473.9–522.4) | 15.7 | 312.7 (292.6–333.6) | 9.8 | 3.7 (3.4–3.9) | 3.45 (2.56–4.37) |
| 36 | 566 | **429** | 312 | **11.9** | 528.2 (503.6–565.8) | 14.7 | 329.4 (312.1–350.5) | 9.1 | 3.4 (3.2–3.7) | 3.47 (2.55–4.41) |
| 40 | 563 | **438** | 313 | **11.0** | 535.1 (500.3–563.3) | 13.4 | 341.6 (312.7–361.6) | 8.5 | 3.2 (3.0–3.4) | 3.45 (2.56–4.35) |
| 44 | 630 | **479** | 352 | **10.9** | 586.9 (512.7–630.5) | 13.3 | 371.2 (352.0–407.2) | 8.4 | 3.2 (2.8–3.4) | 3.44 (2.59–4.48) |
| 48 | 645 | **493** | 364 | **10.3** | 600.8 (560.3–644.8) | 12.5 | 384.3 (364.1–423.1) | 8.0 | 3.0 (2.7–3.2) | 3.45 (2.49–4.41) |
| 52 | 661 | **509** | 375 | **9.8** | 620.2 (567.2–660.6) | 11.9 | 398.3 (375.1–466.4) | 7.7 | 2.8 (2.6–3.0) | 3.46 (2.58–4.41) |
| 56 | 627 | **469** | 322 | **8.4** | 576.4 (523.4–627.2) | 10.3 | 362.5 (322.5–412.1) | 6.5 | 2.4 (2.2–2.6) | 3.45 (2.55–4.48) |
| 60 | 643 | **483** | 347 | **8.0** | 588.8 (538.3–643.0) | 9.8 | 376.8 (347.3–490.2) | 6.3 | 2.3 (2.1–2.6) | 3.46 (2.58–4.47) |
| 64 | 721 | **540** | 346 | **8.4** | 666.5 (598.4–721.3) | 10.4 | 413.2 (346.5–494.9) | 6.5 | 2.4 (2.1–2.7) | 3.45 (2.55–4.42) |
