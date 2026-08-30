# qwen-image-edit — Qwen-Image-Edit-2511 (NVFP4) image-edit server

Image **edit** server (OpenAI Images `/v1/images/edits`) for `Qwen/Qwen-Image-Edit-2511` on a single DGX
Spark (GB10/SM121), served as a **pre-quantized NVFP4** checkpoint with the 4-step **Lightning LoRA** on top.
Ported from the old eugr on-the-fly recipe (`x/qwen-image-edit-2511.yaml`) into our `recipes/` structure with
a pre-quant step (like `flux2-dev`) so it boots instantly — no BF16 peak, no JIT stall.

## Measured (2026-08-30, GB10, 8 steps + Lightning, avg of the 3 canonical seeds)

| tier | size | seconds / edit |
|---|---|---|
| HD | 1280×720 | **~21s** |
| FHD | 1920×1080 | **~37s** |
| QHD | 2560×1440 | **~67s** |

First request at each NEW size pays a one-time Triton compile stall (~1–2 min, `COMPILE_BLOCKS=1`);
the numbers above are warm. Timing/quality protocol: `bench/README.md` (gauntlet, canonical seeds
123123123/456456456/789789789, QHD = the showcase tier — winners in `tests/`). The one trap size:
avoid exactly 1024×1024 (breaks only when output==input size AND a dim is exactly 1024).

## Run it

```bash
./quantize.sh qwen-image-edit     # downloads the BF16 base if missing, NVFP4-quants the transformer
                                  #   → models/myllmbox/Qwen-Image-Edit-2511-nvfp4
./build-and-copy.sh qwen-image-edit   # build the serve image (mbx-qwen-image-edit) — solo box, no copy needed
./run.sh qwen-image-edit          # auto-downloads the Lightning LoRA (extra_models), serves pre-quant + LoRA
```

Everything self-provisions: `quantize.sh` fetches the base, `run.sh` fetches the LoRA. No manual `./download.sh`.

## What it is

- **Base:** `Qwen/Qwen-Image-Edit-2511` — Qwen-Image MMDiT multi-image edit model. **Edit-only**: every request
  needs ≥1 input image → `POST /v1/images/edits` (multipart). `/v1/images/generations` returns 400.
- **Quant:** transformer → NVFP4 (torchao, Triton sm_121a), VAE/text-encoder kept BF16. ~10–12 GB vs ~40 GB BF16.
- **LoRA:** `lightx2v/Qwen-Image-Edit-2511-Lightning` (4-step) applied **unfused** at serve time (BF16 adapter
  over the quantized transformer). Declared in `extra_models:`, pointed at by `LORA_PATH`. Requires the
  few-step + CFG-off profile: `DEFAULT_STEPS=8`, `DEFAULT_TRUE_CFG=1.0`. For full quality, drop the LoRA and set
  steps 40 / true_cfg 4.0.
- **Placement:** SOLO — one Spark, takes the media slot. Does **not** co-host a TP LLM serve.

## ⚠ Edit resolution floor

Edits **collapse to a neon-posterized identity copy** (prompt ignored) when **either output dimension ≤ 1024** —
it is *not* total area (2048×1024 breaks; 1328×1328 is fine). It is NOT a LoRA / scheduler / steps / CFG issue.
From ~1328² (Qwen native) up, edits are excellent. `server.py`'s `size=auto` therefore targets `EDIT_AUTO_PIXELS`
(default here 4194304 ≈ 2048²) at the input's aspect. Pass `size=input` for raw pipeline behavior, or an explicit
`WIDTHxHEIGHT` (client owns the floor: short side ≥ 1088).

Timings (NVFP4 + 4-step LoRA): 1328² @8st ≈ 21–27s, 2048² @8st ≈ 75–81s.

## Version lockstep

`recipes/qwen-image-edit/Dockerfile` pins **the same** torch / torchao / mslk / diffusers as
`quantizer/Dockerfile` (currently `torchao 0.18.0`, `diffusers 0.39.0`). torchao serializes FP4 as versioned
tensor subclasses, so a mismatch fails on load — `run.sh` pre-flights the quant manifest against the serve image.
If a future build needs a newer diffusers, bump **both** Dockerfiles together and re-quantize.

## Test

```bash
curl -F prompt="make it snowy" -F image=@in.png -F size=auto \
  http://192.168.1.66:8000/v1/images/edits
# via the tunnel/proxy: add  -H "Authorization: Bearer $BINDING_TOKEN"
# watch the boot log for:  "LoRA loaded (unfused adapter)"
```
