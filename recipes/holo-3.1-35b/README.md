# Holo-3.1-35B-A3B — the vision model: fast, flat, one box

Hcompany's 35B-A3B vision-language model (NVFP4) on a single DGX Spark. It reads screenshots,
documents, UI states — anything with pixels — and answers through plain `/v1/chat/completions`.

```
./run.sh holo-3.1-35b
```

## Measured (2026-08-30, GB10, clean single-stream decode windows)

- **~75 tok/s steady** — median 74.7, p10–p90 within 73.9–75.4. No speculative decoding, so the
  speed is *flat*: code, prose, vision answers all decode at the same rate (unlike the MTP-spec
  serves whose tok/s swings with content). A c=2 overlap window showed ~100 aggregate.
- **KV pool: 3,407,872 tokens** (3.4M) at 131k max context, fp8 KV, `gpu-memory-utilization 0.55`
  — the model is small enough that the box barely notices it; enormous concurrency headroom
  (`max-num-seqs 16` configured).

## Quant

NVFP4 ModelOpt checkpoint end-to-end (attention included), run as **W4A16 via Marlin** — 4-bit
weights dequantized to bf16 on the fly; GB10/SM121 can't run the true W4A4 FlashInfer path
(`VLLM_USE_FLASHINFER_MOE_FP4=0` pins that off). KV cache fp8.

Note the contrast with the Qwen3.8 hibrid45 lesson (attention quantization = quality tax): this
checkpoint quantizes attention to 4-bit too, and it shows in generation quality — Holo draws a
rough pasture where Qwen paints one. That's fine: this box's job is *seeing*, not composing. Use it
as the eyes next to a bigger brain.

## Role in the circle

- Vision requests: base64 or URL images in standard OpenAI `image_url` content parts.
- Tool calling + reasoning parser configured (`qwen3_xml` / `qwen3`), so agent harnesses can point
  a vision sub-task here while the main model lives on another box.
- At 0.55 memory utilization it leaves ~half the box free — a natural co-host candidate (mind the
  port/name collisions if you try — see the co-hosting notes in the runner docs).
