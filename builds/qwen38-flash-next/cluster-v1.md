# docker.io/myllmbox/qwen38-flash-next-cluster-vllm:v1

**Published:** 2026-09-05 (pushed from ai1; only the patch layers uploaded, base layers mounted from
`myllmbox/qwen38-flash-next-vllm`).
**Digest:** `sha256:c93988a847674d742fc4cc87ec2bd386c753727d65e8bbc4b739de9a9da68618` (from the `docker push` output)
**Image id (local, = mbx-qwen38-flash-next-cluster):** `sha256:f17baedb1802028398480580aca3163530e56d9a9b097030bda621b4191abf1c`

## What it is
`myllmbox/qwen38-flash-next-vllm:v1` (see vllm-v1.md) + one patch, `cluster/docker/patches/01-ple-offload-multinode.py`:
upstream vLLM refuses the PLE CPU-offload worker at `nnodes != 1`; under env `MBX_PLE_MULTINODE=1` the patch lifts four
gates (validation, spawn gate → node-local rank 0, worker topology collapsed to the local view, connector request
source per rank) so each box runs one full int3-table worker fed by its own rank. Without the env: byte-identical
behaviour to v1. Six anchored edits, each asserting the anchor is unique and refusing to apply twice.

## Source
- Dockerfile: `builds/qwen38-flash-next/cluster/Dockerfile` (+ `docker/patches/`) — same bytes as
  `recipes/qwen38-flash-next-cluster/`
- Build: `docker build -t myllmbox/qwen38-flash-next-cluster-vllm:v1 builds/qwen38-flash-next/cluster/`
- Base: `myllmbox/qwen38-flash-next-vllm:v1@sha256:92ccd7de6d80b9597844d7a5bffb9ded4067dd323129e18bc24192c00dac09ad`

## Validation (2026-09-05, 2× DGX Spark, TP=2, NCCL on RDMA — proven by HCA port counters, TCP path flat)
- Boot 1 (K=3, util 0.70, KV auto): body 38.05 GiB/node, int3 table 17.9 GiB in each node's offload worker
  (worker came up on BOTH boxes), KV 40.2 GiB → 2.70M tokens. c=1: 17–18 engine steps/s (56 ms), code 62–72 tok/s.
- Boot 2 (K=4, max-num-seqs 64, kv-cache-memory 46G): pasture/thinking-off ladder via bench/test.py —
  c=1 68 (55–77) · c=8 254 · c=32 464 · c=64 626 tok/s aggregate; engine 16.6 / 7.5 / 3.4 / 2.35 steps/s;
  acceptance 4.2–4.6. Solo reference: 44 / ~153 / 305 max. Rows: recipes/qwen38-flash-next-cluster/reports.md.
- Pasture/fish gauntlet: pending.

## Shipping serve config (at export) — see solo/qwen38-flash-next-cluster-recipe/recipe.yaml
`--tensor-parallel-size 2 --nnodes 2` (kit-added) · util 0.70 · kv-cache-memory 46000000000 · max-model-len 262144 ·
max-num-seqs 64 · max-num-batched-tokens 8192 · MTP K=4 · async-scheduling on · env VLLM_PLE_CPU_OFFLOAD=1
MBX_PLE_MULTINODE=1 + RoCE v2 NCCL pins; container: --network host --device /dev/infiniband --cap-add IPC_LOCK
--ulimit memlock=-1:-1.

## Pairs with
- Weights: `hf.co/myllmbox/Qwen3.8-Flash-Next-hibrid46` (unchanged)
- Kit: `github.com/bilikaz/qwen38-flash-next-cluster-recipe` (solo/qwen38-flash-next-cluster-recipe; first push 2026-09-05, commit 66fb106)
