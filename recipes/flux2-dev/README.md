# flux2-dev — FLUX.2-dev image generation (NVFP4, single Spark)

The first **non-vLLM** recipe (image generation) **and** the first user of the **quantizer**. FLUX.2-dev is
quantized to NVFP4 **once, offline**, into a shareable checkpoint; then served **instantly** (no on-the-fly
quant), fronted by the same keepalive proxy + cloudflared tunnel as every other box.

## Two steps
```bash
./download.sh black-forest-labs/FLUX.2-dev   # BF16 base → models/black-forest-labs/FLUX.2-dev
./build-and-copy.sh flux2-dev                # build the serve image → mbx-flux2-dev  (torchao stack)
./quantize.sh flux2-dev                      # NVFP4 → models/myllmbox/FLUX.2-dev-nvfp4  (also: --to mxfp8)
./run.sh flux2-dev                           # serve the pre-quant (generic mode) → proxy → tunnel
```
`quantize.sh` builds the shared `mbx-quantizer` image the first time (slow, once), then reuses it. The
output `models/myllmbox/FLUX.2-dev-nvfp4` is **self-contained + shareable** — publish it and others just
`./download.sh myllmbox/FLUX.2-dev-nvfp4` and `./run.sh` on their own Spark.

## What makes it "generic" (not vLLM)
The recipe sets **`command:`** — the runner flips to generic mode: `docker run … mbx-flux2-dev uvicorn
server:app …` with **no** vLLM flags. Everything else (env, mounts, `-p`, proxy `:8011`, tunnel) is shared.
`QUANT_PRELOADED=1` tells `server.py` to load `MODEL_PATH` (the pre-quant) as-is.

## Why offline quant (not on-the-fly)
NVFP4 gives ~4× on GB10 (Triton FP4 kernels, sm_121a) and ~16–20 GB vs 80 GB BF16. Doing it **once** means
every boot is a fast load — no ~2 min quantize, no per-resolution JIT stall — and the result is a portable
artifact. The `quantize:` block in `myllmbox.yaml` holds all the params (source, skip-modules, formats).

## API (OpenAI Images)
- `POST /v1/images/generations` — text→image · `POST /v1/images/edits` — edit/multi-reference (multipart)
- `GET /health`, `GET /v1/models` — health/model-card (public through the proxy)

Files: `Dockerfile` (serve image), `server.py` (the app), `myllmbox.yaml` (serve + `quantize:` blocks).
**Solo only**, FLUX.2-dev Non-Commercial.
