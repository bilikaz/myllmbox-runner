# qwen38-flash-next — one Spark (v2: hibrid47, demand-paged NVFP4 table, fp8 KV)

The single-box serve of the published stack. **v2 (2026-09-07)** = the same
[hibrid47](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid47) checkpoint the 2-Spark cluster serves — hibrid46's
body plus the 95 GB n-gram (PLE) table re-quantized to NVFP4 (26.9 GiB) — on ONE GB10, no CPU offload worker. Three
things make it fit where a resident table OOMed:

- **`docker/patches/04-ple-mmap.py`** — the table is never allocated. The 8 `ple-nvfp4-*` shards are mmapped read-only
  and a DLPack capsule over the mapping hands the GPU a tensor it gathers from directly (GB10 ATS: `pageableMemoryAccess`,
  `usesHostPageTables`). Boot = 73.3 GiB of weights instead of 99; FlashInfer autotune gets its ~34 GB of transient room.
- **`05-ple-mmap-dual.py`** — `vllm::mbx_ple_gather`, a splitting op with two bodies: O_DIRECT preads during the eager
  boot steps (no page-cache growth, no GPU faults while autotune runs), the mmap gather once warm-up + capture finish
  (`MBX_PLE_MMAP_MODE=auto`). `MBX_PLE_MMAP_PREWARM=auto` then populates the whole table into memory (MADV_POPULATE_READ,
  ~25 s). Flags in `/cache`: `mbx-ple-mmap`, `mbx-ple-nvme`, `mbx-ple-prewarm` (repeatable) — a watcher thread polls them.
- **`06-qsa-fp8-nvfp4-kv.py`** — upstream PR #54846 ported (whole-file overlays in `docker/overlays/`, the PR's 13 tests
  in `docker/tests/` pass on the GB10): `--kv-cache-dtype fp8` on the QSA path. 391,943 KV tokens on a 7 GB pin, 1.80× bf16.
- `01`–`03`: the int3/offload fallback gate (inert), the GPU-resident NVFP4 method (patch 04 zero-sizes its params), the
  loader's page-cache drop per shard.

## Run
1. Box free, `vm.compaction_proactiveness=0` (`cluster/server-profile.sh`), ≥100 G available (`free -g`).
2. `./run.sh qwen38-flash-next` — builds `mbx-qwen38-flash-next` from this folder, launches. Boot log milestones:
   `PLE MMAP: gather path NVME (O_DIRECT) during boot`, `Model loading took 73.3 GiB`, `GPU KV cache size: 391,943 tokens`,
   `PLE MMAP: gather path → MMAP (auto: …)`, `PLE MMAP: populated …`. ~12 min cold (weights 11 min), faster warm.
3. Verify a real completion; `free -g` should show ~27 GB of the table in `buff/cache` (mapped), swap ≤ 6 GB, then flat.
4. Numbers: `bench/test.py --c 1 --thinking off --prompt pasture`; expect 14.4 steps/s, 50–51 tok/s, acceptance ~3.5.
5. Gauntlet before quoting anything: `tests/pasture.html` + `tests/fish.html`, best of 3, user-generated (pending for v2).

## Measured — v2 (K=3, pasture, `vm.compaction_proactiveness=0`, boot 10 of 2026-09-07, fp8 KV)
| | v1 (int3 table, offload worker, kv 18 G bf16) | **v2** |
|---|---|---|
| c=1 sustained, thinking off | 44 tok/s | **50–51** (12 runs: 49.0–51.1 avg) |
| c=1 peak window | 55 @ acceptance 4.0 | 54.6 |
| engine steps/s c=1 | ~13.75 | **14.4** (14.1–14.5, ±1 %) |
| c=4 | 103 avg | **129** avg (123–133), 133 peak, 9.3 steps/s, acc 3.47 (fp8 KV boot); bf16 boot 9: 124 / 8.9 |
| c=8 (all seats) | 148–158 | **182** avg (158–193, 21 windows), 193 peak, 6.6 steps/s, acc 3.42; kv 94 → 99.3 %, no preemption |
| c=1 thinking on, full 30k-token request | — | 42 avg (39 thinking → 51 code), same 14.4 steps, 4 runs 39–42 |
| KV pool | 579,550 tok bf16 @ 18 G | **391,943 tok fp8 @ 7 G** (1.50× a 262k request) |
| weights at boot | 91 G (table in the worker) | 73.3 GiB GPU + 26.9 GiB table in page cache |
| NVMe while decoding | — | 0.4 reads/s with the table resident |

The step is ~70 ms either way on one GPU (it reads all the dense bytes; the cluster's 56 ms halves them). The gain
over v1 is the table path (no IPC detour) and the NVFP4 table's quality (cluster gauntlet 26/32 vs ~16/32 for int3).

## Memory (why the numbers are what they are)
119.7 GiB box: GPU process 85 GB (73.3 weights + 7.9 pool + 0.5 graphs) + 27 GB table pages + 7 GB host heaps
(API 2.8, worker 2.8, engine 1.2) = 119. The kernel holds ~6–8 GB free but fragmented (every free block ≤128 KB), so the
populate pass parks ~6 GB of cold heaps in swap once; swap-ins ≈ 0 afterwards and steps stay flat. Rows the kernel
reclaims come back on demand (GPU fault ≈ 200 µs, 45 hash-random rows per 4 KiB page). Candidate next: `MADV_RANDOM` on
the mappings (kill 2 MB readahead folios → no compaction storms during populate). Details: `reports.md`,
`builds/qwen38-flash-next/ple-mmap-probe/`.

## Tuning
- **`kv-cache-memory`** 7 G fp8. Bigger pin = fewer table pages resident (they are the same memory); measured free after
  populate ≈ 4–15 GB depending on the pass. `kv-cache-dtype: fp8` → drop the line for bf16 (217,808 tokens).
- **`max-num-seqs`** 8: ~1 GB of pool per running request is GDN state regardless of length.
- **`MBX_PLE_MMAP_PREWARM`**: `auto` (populate after the flip), `0` (hot set only, table fills on demand), `<seconds>`.
- **`speculative-config`** K=3 (K=4 not yet A/B'd on the solo).
- No `dashboard:` — the memory is the constraint.

## v1
hibrid46 + in-checkpoint int3 table, `myllmbox/qwen38-flash-next-vllm:v1`, `VLLM_PLE_CPU_OFFLOAD=1`, kv 18 G: the
git history of `myllmbox.yaml` up to 2026-09-06, and the kit's tag `v1`.
