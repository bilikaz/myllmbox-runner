# PLE table demand-paged from NVMe — the GB10 ATS probe (2026-09-07)

Question: can patch 03's gather kernel read the NVFP4 n-gram table straight out of a memory-mapped safetensors
file, so a single Spark never allocates the 26.9 GiB table at boot and only touched rows ever occupy memory?

`probe.cu` mmaps a real hibrid47 shard and runs a GPU gather kernel on the host pointer. Run on box2 (idle),
driver 580.142, kernel 6.17.0-nvidia, inside `myllmbox/qwen38-flash-next-cluster-vllm:v3` (nvcc 13.0):

    nvcc -O2 -arch=sm_121a -o probe probe.cu && ./probe /m/model-00008-of-00021.safetensors 4096 2048 3

Raw output: `results-2026-09-07-box2.txt`. Findings (4096 random 2 KB rows unless noted):

| path | time | note |
|---|---|---|
| GPU reads an mmapped file | works | `pageableMemoryAccess=1`, `usesHostPageTables=1` (ATS); no cudaMalloc, no copy |
| rows already mapped (CPU touched the pages) | 0.35–0.51 ms | same class as the resident device copy (0.27–0.44 ms) |
| rows in page cache but NOT mapped → GPU takes the fault | 50–530 ms | ~90–200 µs per page, 12–130 µs amortised; the killer |
| single GPU page fault, warm cache | 197 µs | vs 7 µs for a mapped row, 6 µs device |
| CPU pre-touch of a warm-cache page (populates the PTE) | 1.2–1.7 µs | 4096 rows in 4.5 ms |
| cold from NVMe via GPU faults | 79 µs per row | 4096 random rows in 325 ms |
| streaming 3 GiB, PTEs present | 157 GB/s | device copy: 190 GB/s |
| streaming 3 GiB, GPU faulting page by page | 3.3 GB/s warm, 0.66 GB/s cold | |

Design consequence: the GPU must never take the fault. The CPU (which knows the n-gram indices before the layer
runs) populates the needed rows — `MADV_POPULATE_READ` on the row ranges, or a plain touch — and the kernel then
reads at device speed. Cold rows cost one NVMe read (~80 µs, overlappable); warm-but-unmapped rows ~1.5 µs.

The catch is page granularity. Table geometry (config.json `ple_quantization`): 320,001,536 rows × 90 B
(80 B packed NVFP4 + 10 B fp8 scales), 8 shards, one PLE layer (`ple_layer_ids: [2]`), 16 heads → 16 row
lookups per token. A 4 KiB page holds 45 rows and rows are hash-ordered (no locality), so the fraction of PAGES
touched is 1 − (1 − p)^45 for a row fraction p: p = 1 % → 36 % of pages resident, p = 5 % → 90 %. The memory
bet only pays if the distinct-row working set stays under ~1 % (~3 M rows), or the table is re-laid-out by
hotness (needs a corpus frequency profile + an index remap). Measure before building: count distinct rows AND
distinct pages over a real traffic window (patch 03 sees every index).

Per step, c seats, K=4: 16 × 5 × c row lookups (80 per seat) → at c = 8, 640 rows/step ≈ 640 pages if random.
