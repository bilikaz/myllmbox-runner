# docker.io/myllmbox/qwen38-flash-next-vllm:v1

**Published:** 2026-09-02
**Digest:** `sha256:92ccd7de6d80b9597844d7a5bffb9ded4067dd323129e18bc24192c00dac09ad`

## What it is
The serving image for `hf.co/myllmbox/Qwen3.8-Flash-Next-hibrid46`: the vendor's SM121 vLLM base
plus the myllmbox `ple_layer.py` patch set — config-declared quantized-PLE loading
(`ple_quantization` in config.json → the int3 table loads from ordinary checkpoint tensors,
resident in the offload worker), MBX_PLE_MMAP bank persistence (lab path), MBX_PLE_QBITS ladder
(lab path), MBX_INDEX_STRICT, MBX_VOCAB_GEMV.

## Source
- Dockerfile: `builds/qwen38-flash-next/Dockerfile` (this folder — the full reproducibility
  bundle: Dockerfile + `docker/` payloads + the hibrid46 converter the image bakes)
- Rebuild: `docker build -t myllmbox/qwen38-flash-next-vllm:v1 builds/qwen38-flash-next/`
- Base image: `vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`
- Built on a DGX Spark (GB10), 2026-09-02; tagged `myllmbox/qwen38-flash-next-vllm:v1`

## Validation before push (2026-09-02, single DGX Spark)
- Fresh-cache boot of the exported HF repo (self-describing checkpoint, no MBX env):
  health 200; `PLE: int3 quantized table loaded from checkpoint (17.9 GiB packed, standard load)`;
  KV pool 579,550 tokens @ 18G bf16; memory 102G used / 17G available.
- Completion check: pasture one-shot, thinking off — 8,210 tokens @ **41.4 tok/s c=1**,
  `finish=stop`, valid HTML head-to-tail, full animal roster. In-band (39.9–46.1).
- Config-driven allocation verified: all 4 `_mbx_ple_std_cfg` patch sites present in image.

## Shipping serve config (at release)
`--kv-cache-memory 18000000000 --max-model-len 262144 --max-num-seqs 8
--max-num-batched-tokens 8192 --gpu-memory-utilization 0.70
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--enable-prefix-caching --async-scheduling=OFF` + env `VLLM_PLE_CPU_OFFLOAD=1`.
- async OFF is load-bearing: async+MTP corrupted ~2 of 8 concurrent outputs (2026-09-01).
- KV must stay bf16 (vendor QSA guard refuses fp8).
- Prefix caching: watch item — peer recipe reports a mamba/GDN split-alignment bug; under
  investigation.

## Pairs with
- Weights: `hf.co/myllmbox/Qwen3.8-Flash-Next-hibrid46` (91.11 GiB, 21 shards + in-shard int3
  PLE tensors + `ple_quantization` config declaration)
- Kit: `github.com/bilikaz/qwen38-flash-next-recipe` (pending)
