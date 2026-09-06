---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - nvfp4
  - dgx-spark
  - gb10
  - vllm
  - quantized
  - multi-node
---

# Qwen3.8-Flash-Next — hibrid47 (4.7-bit): the n-gram table lives on the GPU

**Two DGX Sparks. 262k context. 80 tok/s single-stream peak (76 average), 533 tok/s at 32 concurrent
streams, 26 of 32 boss-level render tests passed. Same body as hibrid46 — a different table.**

hibrid46 made Qwen3.8-Flash-Next fit one Spark by shipping its 95 GB n-gram (PLE) table as int3 and gathering it on
the CPU side of unified memory every step. That gather is a detour: a message-queue hop, a CPU gather, a pinned
buffer, a DMA, per step. hibrid47 ships the same table as **NVFP4 (28.6 GiB)** — a layout the GPU dequantizes
in-kernel — so the serving image holds it as an ordinary resident GPU parameter and gathers it inside the model's
forward pass, inside the CUDA graphs. No CPU worker, no IPC, no per-step detour. Everything else is byte-identical
to hibrid46.

Two things came out of that, measured on the same boxes, same tests, same day:

- **speed**: engine steps +8–11 % on every concurrency rung (17.7 vs 16.4 steps/s at c=1; 4.0 vs 3.7 at c=32);
- **quality**: on the 32-scene "boss-animals" render gauntlet, thinking on, **26 good, 3 partial, 3 broken** —
  the int3 table scored about half/half on the same scenes. The 4-bit block-scaled table is simply a better table
  than 3-bit per-row.

## Precision map

| tier | precision | notes |
|---|---|---|
| routed experts | NVFP4 W4A4 (from [Inferact/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4)) | 39 % of the checkpoint |
| GDN linear-attention projections | NVFP4 W4A16, Marlin (myllmbox) | A/B-validated tolerant |
| QSA attention q/k/v/o, shared experts, lm_head, embeddings, norms, hyper-connection mixers | bf16 | the quality tier |
| MTP drafter: routed experts | NVFP4 W4A4 (Inferact) | bf16 drafter A/B'd 2026-09-06: acceptance +0.01, −3 % steps → kept NVFP4 |
| MTP drafter: everything else | bf16 | |
| **PLE n-gram table (320,001,536 × 160)** | **NVFP4: e2m1 codes, fp8 block scale per 16, one fp32 global** | 28.6 GiB; 8 shards `ple-nvfp4-*.safetensors`; declared in `config.json` → `ple_quantization` |

## Measured (2× DGX Spark GB10, TP=2 over ConnectX RDMA, vLLM, MTP K=4, temperature 1.0 as shipped)

Structured-output prompt, thinking off; every row is the average of 3–7 independent runs of steady 10 s windows,
warm-up and prefill windows excluded; "peak" = best steady window.

| concurrent | engine steps/s | aggregate tok/s avg | peak | per stream |
|---|---|---|---|---|
| 1 | 17.7 | 73–76 | **80** | 76 |
| 2 | 15.1 | 126 | 133 | 63 |
| 4 | 11.8 | 198 | 209 | 50 |
| 8 | 8.8 | 294 | 309 | 37 |
| 16 | 6.2 | 417 | 451 | 26 |
| 24 | 4.9 | 488 | 514 | 20 |
| 32 | 4.0 | 533 | 579 | 17 |

Thinking on at c=32: 320–340 tok/s (acceptance 2.5 on reasoning prose; the step rate is the same). Acceptance is a
property of the text and the sampling, not of this table: pasture-type prose 4.2–4.3, code 4.2, reasoning 2.5.

One host setting mattered as much as the table: `vm.compaction_proactiveness=0`. On a unified-memory box the
kernel's background page compactor migrates pages the GPU has mapped; on a tightly-pinned serve that showed up
as a 4–5 s slowdown every ~37 s. The kit README explains it; it needs root, so the kit only recommends it.

## Fit

- **2 Sparks (the target):** ~50 GiB per box with the table sharded across the pair (KV 40–46 G, 80+ seats), or
  ~65 GiB per box with the full table on each (KV 25 G, ~49 seats; no cross-box table exchange — measured equal steps).
- **1 Spark:** it loads (~102 GiB) but leaves ~9 GB for KV — a c≤8 box. For a single Spark, hibrid46 remains the
  release until the int3-on-GPU load path lands.

## How to run

The checkpoint is standard safetensors; the GPU-resident NVFP4 table is an image-side load path (a small patch on the
vendor's `ple_layer.py`, in the myllmbox repo). The two-Spark kit:

```
git clone https://github.com/bilikaz/qwen38-flash-next-cluster-recipe.git
cd qwen38-flash-next-cluster-recipe && ./run.sh     # discovers the pair, pulls image + weights, serves :8000
```

(Kit v2 with the hibrid47 image is the release that accompanies this checkpoint; v1 = the hibrid46 kit, tagged.)

## Repo layout

19 body shards (`model-*.safetensors`, hibrid46's) + 8 table shards (`ple-nvfp4-0000k-of-00008.safetensors`) + index;
`config.json` = base release + `ple_quantization`; tokenizer files unchanged. Every tensor exists exactly once.

## Reproducibility

`builds/qwen38-flash-next/docker/make-hibrid47.py` (the table converter: two streaming passes over the bf16 table,
global amax then per-block fp8 scales with codes rounded against the *effective* scale), the engine patch, the bench
tools and every number's conditions are in the myllmbox repo. Run the kit on a pair of Sparks and count the tokens.

## Attribution & license

- Base model: Qwen/Qwen3.8-Flash-Next (Alibaba) — **Qwen Community License 1.0** (included as `LICENSE`): modification,
  derivative works, distribution, hosting and commercial use permitted; products over 100M MAU / $20M monthly revenue
  must display the model name; a Model-as-a-Service or AI-assistant *business* on it needs a separate Qwen license.
- Routed-expert (and drafter-expert) NVFP4 tensors derived from Inferact/Qwen3.8-Flash-Next-NVFP4.
- GDN-tier quantization, the NVFP4 table pipeline, the GPU-resident load path and the serving stack: myllmbox.
