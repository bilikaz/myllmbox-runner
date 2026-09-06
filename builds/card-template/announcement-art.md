# scripts/ — post & announcement art

Two asset types per model announcement, made two different ways. The rule that makes both work:

> **The model paints, the code types.** Diffusion models compose beautiful scenes and do clean
> targeted edits, but they cannot reliably spell ("Qwen3.8-Flash-Next" came back as
> `-I-`, `-)-` and `Quen3 8` in three tries). Any text that must be EXACT is rendered
> deterministically with PIL. Never ask the model to one-shot a poster with correct copy.

## 1. Benchmark banner — `benchmark-banner.py` (pure code)

Spark-Arena-style dark stat card in the house palette. Fully deterministic, run anywhere with PIL:

```bash
python3 scripts/benchmark-banner.py \
  --org Qwen --title QWEN3.8-FLASH-NEXT \
  --chip vLLM --chip "hibrid45 · NVFP4+bf16 attn" --chip "1 node" \
  --stat "55:tok/s:DECODE · CODE C1:green" \
  --stat "32.6:tok/s:DECODE · MIXED AVG:white" \
  --stat "3.0K:tok/s:INGEST · WARM PREFILL:orange" \
  --stat "800K:tok:KV POOL · 262K CTX:white" \
  --hardware "NVIDIA DGX Spark · GB10 × 1" \
  --foot "one command: ./run.sh qwen38-flash-next · clean 10s decode windows, single stream · code band 50–57 · quality-gated" \
  --out qwen38-benchmark-banner.png
```

House rules for the numbers:
- **Only measured values** — clean 10-second decode windows, single stream, no prefill in-window.
- Lead with the honest pair: the code band AND the mixed-diet average (never a bare peak).
- KV pool is a standing comparison stat; warm-ingest ≠ cold prefill (label which one it is).
- The footnote carries the methodology so the card can't be read as cherry-picked.

## 2. Illustrated poster — the qwen-image-edit serve, step by step

Needs `./run.sh qwen-image-edit` up (any box). Work like an editor, in ROUNDS — never one-shot:

```bash
# step 1 — creative scene from the logo (let the model live; describe mood, not layout minutiae)
curl -s -X POST http://127.0.0.1:8000/v1/images/edits \
  -F "image=@logo-512.png" -F "model=Qwen/Qwen-Image-Edit-2511" \
  -F "prompt=Create a beautiful poster background from this logo: warm cozy basement at dusk,
      a small glowing computer box on a desk, circuit-like light trails, cream/purple/orange
      palette, the red roundel logo at the top center, flat-illustration style" \
  | python3 -c "import json,sys,base64;open('step1.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"

# step 2 — targeted edits on the result (fix/remove elements; feed the previous output back in)
#   e.g. "Remove the large headline text completely, keep everything else exactly the same."

# step 3 — typeset the exact copy in code (PIL over the clean plate):
#   headline + stat chips — see the inline PIL block pattern in benchmark-banner.py
#   (fonts: DejaVuSans-Bold; house colors in that script's constants)
```

Operational notes for the edit serve:
- First request at each NEW image size pays a ~1–2 min Triton compile stall (`COMPILE_BLOCKS=1`);
  subsequent same-size requests are ~75–80s at 2048².
- `size=auto` targets ~4.2MP at the input's aspect — safely above the **1024 edit floor trap**
  (never request output at the input's exact size when a dimension is exactly 1024).
- The response is OpenAI Images JSON (`data[0].b64_json`).
- Text-removal and element-level edits are far more reliable than text *insertion* — hence the
  typeset step.

Provenance: this workflow produced `qwen38-poster-final.png` and `qwen38-benchmark-banner.png`
for the hibrid45 champion announcement (2026-08-30) — the first real jobs through the ported
qwen-image-edit recipe, which double-served as its validation.
