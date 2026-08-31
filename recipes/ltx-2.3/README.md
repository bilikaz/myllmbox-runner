# ltx-2.3 — LTX-2.3 22B audio+video server

Video **with its own synchronized sound** in one pass: Lightricks LTX-2.3, a 22B DiT audio-video
foundation model, on a single DGX Spark (GB10/SM121). Served as the **distilled-1.1** checkpoint —
two-stage generation (stage 1 at half resolution, 8 sigmas → 2× spatial upsample → stage 2 refine,
4 sigmas), CFG-free — with **fp8-cast** weights (~22GB transformer instead of ~44GB).

Ported from the old eugr recipe (`old-eugr/x/ltx-2.3-video.yaml` + `servers/ltx-video`) into the
`recipes/` structure: same server.py (verbatim), Dockerfile rebased on the house serve stack with
the LTX-2 monorepo pinned to v1.3.0.

## Measured (2026-07-20, OLD build — re-measure on this port before any model card)

- 5s clip, 1280×704 @ 24fps (121 frames): **~91s steady**, ~117s first request (warm-up).
- Output: h264 + **real synthesized AAC audio** (waves/ambience actually sound right), ~2–4MB.
- Memory: the pipeline trims aggressively between requests — ~1GB resident idle; spikes only
  during generation (peak unmeasured — don't co-host until it is).
- **Keyframe i2v verified**: two `image` files → start+end keyframes, smooth in-betweening,
  identity+style locked (cartoon fox test; keyframes came from the image server on the other box).

## Run it

```bash
./build-and-copy.sh ltx-2.3     # build mbx-ltx-2.3 — solo box, no copy needed
./run.sh ltx-2.3                # downloads Lightricks/LTX-2.3 + the gemma mirror if missing, serves
```

**Placement: the MEDIA slot.** Video (this) and image (qwen-image-edit / flux2-dev) are
alternatives — swap, never co-host two media serves.

## API

`POST /v1/videos` (multipart form) → JSON `{"data":[{"b64_json": <mp4 base64>}], "_meta": {...}}`

| field | meaning |
|---|---|
| `prompt` | the scene (required) |
| `image` (repeatable) | conditioning keyframes → image-to-video. Defaults: 1st → frame 0, 2nd → LAST frame (start+end in-betweening) |
| `image_frame_idx` / `image_strength` | comma-separated placement overrides (`-1` = last frame) |
| `width` / `height` | snapped to /64 (stage 1 runs at half res). Default 1280×704 |
| `num_frames` | snapped to 8k+1. Default 121 (~5s @ 24) |
| `frame_rate` | default 24 |
| `seed` | gauntlet uses the canonical 123123123 / 456456456 / 789789789 |

```bash
curl -F prompt="a golden retriever running on a beach at sunset, waves" \
  http://192.168.1.66:8000/v1/videos | python3 -c \
  'import json,sys,base64; open("out.mp4","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
# via the tunnel/proxy: add  -H "Authorization: Bearer $BINDING_TOKEN"
```

## The gauntlet

`./bench/gauntlet-video.py <ip:port>` — the two standing scenes (`bench/pasture-video.txt`,
`fish-video.txt`, the video adaptations where MOTION is literal) × the canonical seeds, at the
canonical clip spec (1280×704, 121 frames @ 24). Winners → `tests/{pasture,fish}.mp4`.

## Knobs & TODO (carried from July)

- `LTX_QUANTIZATION`: `fp8-cast` (verified) · `none` = BF16 · `fp8-scaled-mm` = fp8 matmuls,
  **untested on GB10** — A/B before trusting.
- Untested backlog: temporal upscaler (2× fps), IC-LoRA (`lora-384-1.1`), generation-peak memory
  measurement (gates any LLM co-hosting idea).
- Dockerfile torch trio is nightly-unpinned (no torchaudio pin was recorded in July) — freeze the
  versions into the Dockerfile after the first green build+serve.
