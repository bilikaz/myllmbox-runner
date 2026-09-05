# deepseek-v4-flash-vision — DeepSeek-V4-Flash-Vision-Exp on 2× DGX Spark

The vision DeepSeek-V4-Flash (305B MoE, FP8 dense / FP4 experts, 32-layer ViT + aligner trained in,
DSpark self-drafting) served TP=2 across both GB10 boxes. Text quality = V4-Flash-0731; adds native
image input (OpenAI `image_url` parts, ≤387 tokens per image, 8 images per request cap here).

```
./run.sh deepseek-v4-flash-vision      # builds mbx-deepseek-v4-flash-vision, ships image+weights to box2, serves
```

Model card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp (MIT). Weights: 48 shards, 157G.

## What the image is (and why not the official one as-is)

`vllm/vllm-openai:deepseekv4-flash-vision-arm64-cu130` (2026-09-01, vLLM 0.28.1rc1.dev137 from the
#54566 branch, FlashInfer 0.6.18, arch list incl. 12.1) is the only vision-capable vLLM for this
checkpoint — upstream main merged vision on 2026-09-02, no release carries it yet. Stock, it fails on
GB10 in three places; our `Dockerfile` layers five small anchor-asserted patches on the pinned digest:

| # | patch | fixes | origin |
|---|---|---|---|
| 01 | streaming Vision loader + `n_predict := dspark_block_size` | load materialises 157G in host RAM → dies; DSpark depth 3 vs trained 5 | vLLM PR #54631 (open) |
| 02 | plain RoPE on sparse-SWA layers | YaRN wrongly applied to SWA layers (correctness) | vLLM PR #54815 (merged 09-02) |
| 03 | `ALWAYS` cudagraph builders for the FlashInfer DSV4 backend | adaptive verification refused at runtime | vLLM PR #52724 (merged 09-01) |
| 04 | SM121 adaptive enablers: indexer flatten gate, contiguous C128A indices, squeeze/contiguous before the SM120 kernel | `eidx must be contiguous` crash once adaptive runs | r0b0tlab overlay, re-expressed as minimal diffs |
| 05 | FlashInfer dual-cache prefill dispatch for `index_topk=512` | first image request: "Unsupported sparse-MLA prefill configuration" | FlashInfer PR #4850 / r0b0tlab FP8 variant |

Every patch script asserts its anchor and refuses to run twice — if upstream moves code, the build
fails loudly instead of shipping a half-patched image. Reference material (robot's + tony's repos)
is cloned under `.data/` (gitignored) for review; nothing from it is copied wholesale.

Not taken: tony's lane (v0.21 B12X fork + bind-mounted `ds4v_*.py`, `nvfp4_ds_mla` KV) — a different
runtime family; anemll's image (same idea, their patches). Both are references, not our build.

## Serve profile (myllmbox.yaml)

r0b0tlab's 2×GB10 production profile — k=5 DSpark with adaptive verification, fp8 KV, block 256,
`FULL_DECODE_ONLY` cudagraphs sized `[1,2,4,8,16,32,48]`, 8192 batched tokens, long-prefill threshold
1024, FlashInfer autotune on except the fp4-block-scale MoE ops — plus house rules: thinking ON by
default, context capped at 300k (robot validated 512k and 1M at util 0.875 on this same image+model;
raising `max-model-len`/util is a config call), X925 cpuset pin + `OMP_NUM_THREADS=8`.

Robot's measured numbers on the same stack (512k profile, thinking off, k=5 adaptive): SHORT c1
43 tok/s · PROSE c1 35 tok/s · c8 93 tok/s aggregate · accept length ~2.7 · Q200 91% · vision gate
8/8. Ours are TBD (`reports.md` once measured).

## First boot

- `run.sh` rsyncs 157G to box2 over the interconnect (one-time, ~15–25 min) and ships the image.
- Cold caches: FlashInfer JITs the patched prefill kernel + SM120 sparse decode (~2–3 min), TileLang
  compiles the mHC kernels, cudagraph capture. All land in `.data/cache` → `/cache` on both boxes.
  Warm boots skip them. Budget ~15 min to healthy either way (157G load).
- `wait_healthy` proves the proxy; verify with a real text completion AND an image request.

Image request shape (base64 data URI in a `user` message):
```json
{"model":"deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
 "messages":[{"role":"user","content":[
   {"type":"text","text":"What is in this image?"},
   {"type":"image_url","image_url":{"url":"data:image/png;base64,...."}}]}],
 "chat_template_kwargs":{"thinking":false}}
```

## Gauntlet

`tests/pasture.html` + `tests/fish.html` (best of 3, user-generated) — pending first serve.
