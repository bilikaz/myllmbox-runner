---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
  - nvfp4
  - fp8
  - compressed-tensors
  - abliterated
  - uncensored
  - dgx-spark
  - gb10
  - vllm
  - multi-node
extra_gated_prompt: >-
  This checkpoint has had its safety alignment removed (abliteration, by OrcaRouter). It complies with requests the
  original Qwen3.8-Flash-Next refuses and has no guardrails of its own. It is published for research, red-teaming,
  interpretability and private use behind your own moderation. By requesting access you confirm that you will use it
  lawfully, that you take full responsibility for what you do with it and what it generates, and that you accept the
  Qwen Community License 1.0 that governs these weights.
extra_gated_fields:
  I will use this model lawfully and take full responsibility for its use and outputs: checkbox
  I will put my own safety and moderation layer in front of any deployment reachable by others: checkbox
  Intended use:
    type: select
    options:
      - Research / interpretability
      - Red-teaming / safety evaluation
      - Private use
      - Other
---

# Qwen3.8-Flash-Next — hibrid47-uncensored: the abliterated body on the hibrid47 stack

**Two DGX Sparks, 262k context, 75 tok/s single-stream (17.9 engine steps/s), the same 2.85M-token fp8 KV pool as hibrid47 —
and no refusals.** OrcaRouter's refusal-direction-removed Qwen3.8-Flash-Next, put on the myllmbox layout: the NVFP4 n-gram
table resident on the GPU, the drafter quantized like the body, 99.0 GiB on disk.

## What this is

[OrcaRouter](https://huggingface.co/orcarouter) abliterated Qwen3.8-Flash-Next (Arditi et al. 2024: one refusal direction
estimated at layer 24, orthogonalized out of every residual-writing matrix — 149 tensors) and released an NVFP4 build of it,
[orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4).
That release keeps the 95 GB n-gram (PLE) table in bf16 — 189 GB, two boxes minimum, table gathered from disk. This
checkpoint is the same abliterated body with the three changes the myllmbox stack needs:

| part | orcarouter's release | here |
|---|---|---|
| routed experts (512 × 48 layers) | NVFP4, compressed-tensors, weight-only | **unchanged** — byte-identical shards |
| QSA attention, GDN projections, shared expert, lm_head | FP8 per-channel | **unchanged** |
| PLE n-gram table (320,001,536 × 160) | bf16, 95.4 GiB in one file | **NVFP4, 26.8 GiB, 8 shards** — hibrid47's table, verified identical to this release's bf16 rows on sampled ranges (the abliteration never touched it) |
| MTP drafter experts | bf16, fused 3-D layout | **NVFP4 per expert**, same compressed-tensors recipe as the body (4.7 → 1.5 GiB) |
| tokenizer / generation config | transformers-5.16 re-serialization | the base release's files (the ones the serving image is validated with) |

Size 99.0 GiB (hibrid47: 98.9). Speculative decoding verifies every draft against the full model, so the drafter's
quantization cannot change outputs — only speed.

## Measured (2 × DGX Spark GB10, TP=2 over ConnectX RDMA, vLLM, MTP K=4, fp8 KV, 28 G pin, pasture prompt, thinking off)

| | hibrid47-uncensored | hibrid47 |
|---|---|---|
| engine steps/s, c=1 | **17.9** (17.7–18.1, 3 runs) | 17.7 |
| tok/s, c=1 | 73–75 avg, 80 peak | 73–76, 80 peak |
| draft acceptance | 4.07–4.21 (per position 0.91 / 0.84 / 0.77 / 0.67) | 4.1–4.3 |
| time to first token, 1.2k-token prompt | 0.57 s | — |
| KV pool | 2,846,834 tokens fp8 | 2,846,834 |
| weights per box | 64.95 GiB (full table on each box) | 64.9 |

The body's FP8 linears where hibrid47 has bf16 make a step a hair cheaper; everything else is the hibrid47 stack.
Refusal behaviour: OrcaRouter measured 64–100 % → ~0–3.3 % on harmful prompts with capability within ±2 points of the base;
we have not re-run that evaluation. The body is OrcaRouter's data-free quantization; hibrid47's body is calibrated
(GPTQ / SmoothQuant) — a quality difference this card does not quantify.

## How to run

The two-Spark kit serves it with one changed line (`model:` in `recipe.yaml`); the myllmbox repo carries the lane
`recipes/qwen38-flash-next-uncensored-cluster`. The serving image is the hibrid47 one (patches: GPU-resident NVFP4 table,
fp8 KV on QSA). This repo is gated — accept the agreement above, then `hf auth login` (or `export HF_TOKEN=…`) before the
kit downloads; the kit checks and tells you if either is missing.

```
git clone https://github.com/bilikaz/qwen38-flash-next-cluster-recipe.git
cd qwen38-flash-next-cluster-recipe
# recipe.yaml → model: myllmbox/Qwen3.8-Flash-Next-hibrid47-uncensored
./run.sh
```

## Repo layout

16 body shards (`model-000NN-of-00017.safetensors`, orcarouter's, shard 2 — the bf16 table — omitted) + `model-mtp.safetensors`
(re-quantized drafter) + 8 table shards (`ple-nvfp4-0000k-of-00008.safetensors`, hibrid47's) + index. `config.json` =
orcarouter's with `ple_quantization` added, the QSA layer type in the vendor image's spelling, and one `quantization_config.ignore`
rule (`re:.*mtp\..*`) removed so the drafter's experts load quantized. Every tensor exists exactly once.

## Reproducibility

`builds/qwen38-flash-next/standardize-flash-next.py` (any Flash-Next checkpoint → this layout; verifies or re-quantizes the
table) and `quantize-drafter-experts.py` (fused bf16 drafter → per-expert NVFP4 with the compressed-tensors library's own
functions; checks the gate/up order against a reference) in the myllmbox repo, with every number's conditions.

## Responsible use

No guardrails. Research, red-teaming, interpretability and private use behind your own moderation; anything reachable by
other people needs its own safety layer. You are responsible for lawful use and for the outputs.

## Attribution & license

- Base model: Qwen/Qwen3.8-Flash-Next (Alibaba) — **Qwen Community License 1.0** (included as `LICENSE`). It governs these
  weights and every derivative of them: modification, distribution, hosting and commercial use permitted; products over
  100M MAU / $20M monthly revenue must display the model name; a Model-as-a-Service or AI-assistant *business* on it needs a
  separate Qwen license. (OrcaRouter's cards label their releases Apache-2.0 "inherited from the base model"; the base
  release is under the Qwen Community License, so that is the license carried here.)
- Abliteration and the body's NVFP4/FP8 quantization: OrcaRouter, from orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4
  (their contribution offered under Apache-2.0).
- NVFP4 n-gram table, drafter re-quantization, the standardizer, the GPU-resident load path and the serving stack: myllmbox.
