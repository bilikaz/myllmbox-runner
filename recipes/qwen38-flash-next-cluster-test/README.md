# qwen38-flash-next-cluster-test — the lab lane: NVFP4 PLE table on the GPUs

The K=4 cluster champion (`../qwen38-flash-next-cluster`) with ONE change under test: the n-gram (PLE) table is
re-quantized from int3 to **NVFP4** and lives **on the GPUs**, row-sharded across the two boxes, gathered and
dequantized in-process. Today's serve keeps the int3 table in a separate CPU worker per box and pays a
ZeroMQ → gather → DMA → semaphore round trip every step.

## What changes, exactly

| | reference lane | this lane |
|---|---|---|
| table format | int3, group 160 (one scale+min per row) — 17.9 GiB | NVFP4: e2m1 codes, fp8 scale per 16, fp32 global — 28.6 GiB |
| where | CPU worker per box, full copy (18 GiB each) | GPU process, half per box (14.4 GiB each) |
| per-step path | IPC → CPU gather → pinned buffer → DMA → semaphore | `index_select` + LUT + scale, in the forward |
| processes | vLLM + PleOffloadWorker per box | vLLM only |
| patches | multi-node offload gate | (kept, inert) + GPU NVFP4 gather |
| box memory | 38 + 46 + 18 ≈ 102 GiB | 38 + 14.4 + 46 ≈ 98 GiB (util 0.86 covers it, see yaml) |

## Files

- `docker/make-ple-nvfp4.py` — the converter. Hardlinks the 19 body shards of hibrid46 (files 20/21 hold only
  the int3 table and are left behind), streams the 128 bf16 shards from hibrid46-off in two passes (global amax,
  quantize), writes 8 NVFP4 shard files + index + `config.json ple_quantization = {format: nvfp4, …}`.
  Resumable per shard. Output: `models/myllmbox/Qwen3.8-Flash-Next-hibrid46-ple4` (~110 GiB incl. links).
- `docker/patches/02-ple-gpu-nvfp4.py` — the engine patch on `ple_layer.py`: a quant method for
  `VocabParallelEmbedding` that allocates packed rows instead of a bf16 weight, a loader that routes the shard
  tensors by row range into this rank's partition, and the gather. Anchor-asserted, `--check` mode.
- `Dockerfile` — v1 image + both patches + a build-time encode/decode round-trip test (the converter's encoder
  against the patch's decoder on synthetic rows; fails the build if they disagree).

## Run order

1. **Build the checkpoint** (one-time, on a box, ~30–60 min; user runs it — the command is in `myllmbox.yaml`).
2. **Both boxes free** → `./run.sh qwen38-flash-next-cluster-test` (builds + ships the image, rsyncs the new dir).
3. **Proof the table loaded on the GPU**: the log must show `PLE: NVFP4 table resident on GPU (… rows this rank …)`
   on BOTH ranks and NO `PleOffloadWorker` lines. `Model loading took` should read ~52 GiB (38 + 14.4).
4. **A/B, same prompts as the reference lane**: `./bench/test.py --c 1,1,8,16,32,64 --thinking off --prompt pasture`
   → compare engine steps/s per rung with `../qwen38-flash-next-cluster/reports.md` (that is the detour's cost).
5. **Accuracy**: `./bench/logprob.py --tag ple4` here, `--tag int3` on the reference lane, and `--tag bf16` on the
   bf16-table lane (`../qwen38-flash-next-off`) — three pooled averages, same texts. NVFP4 should sit between
   int3 and bf16, or at bf16.
6. Gauntlet (`tests/pasture.html`, `tests/fish.html`) before anything graduates.

## If it graduates

The cluster recipe drops `VLLM_PLE_CPU_OFFLOAD` and the multi-node patch, points at the `-ple4` checkpoint, and the
kit follows. The single-box lane keeps int3 (29 GiB unsharded does not fit beside a 19 GiB KV pool there).
