"""OpenAI-style video generation server for Lightricks LTX-2.3 (distilled).

LTX-2.3 = 22B DiT audio+video foundation model. This serves the DISTILLED-1.1
checkpoint via ltx_pipelines.DistilledPipeline: two-stage generation (stage 1
at half resolution, 8 sigmas; stage 2 refines after 2x spatial upsample,
4 sigmas), CFG-free, synchronized audio in the same pass. Text-to-video and
image-to-video (input image(s) become conditioning frames).

Endpoints (loosely OpenAI-shaped, mirrors recipes/qwen-image-edit):
  POST /v1/videos            multipart form: prompt, [image ...], width, height,
                             num_frames, frame_rate, seed, enhance_prompt
                             → JSON {"data":[{"b64_json": <mp4 b64>}], ...}
                             (leading-whitespace keepalive stream, CF-safe)
  GET  /health , GET /v1/models

Ported verbatim from the July-verified old-eugr build (servers/ltx-video); only
this header moved. Bearer gating happens at the keepalive-proxy hop, not here.
"""
import base64
import io
import json
import os
import tempfile
import threading
import time

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

KEEPALIVE_SECS = int(os.environ.get("KEEPALIVE_SECS", "15"))


def _streaming_json(work):
    """Same CF-keepalive trick as the image server: emit a space every
    KEEPALIVE_SECS while work() runs (video gen takes minutes), then the JSON
    body. Leading whitespace is valid JSON."""
    box = {}

    def run():
        try:
            box["result"] = work()
        except HTTPException as e:
            box["error"] = {"message": str(e.detail), "code": e.status_code}
        except Exception as e:  # noqa: BLE001
            box["error"] = {"message": f"{type(e).__name__}: {e}", "code": 500}

    t = threading.Thread(target=run, daemon=True)
    t.start()

    def gen():
        while True:
            t.join(timeout=KEEPALIVE_SECS)
            if not t.is_alive():
                break
            yield b" "
        yield json.dumps(box.get("result") or {"error": box["error"]}).encode()

    return StreamingResponse(gen(), media_type="application/json")


MODEL_ROOT = os.environ.get("MODEL_ROOT", "/models/Lightricks/LTX-2.3")
CKPT = os.environ.get(
    "LTX_CHECKPOINT", f"{MODEL_ROOT}/ltx-2.3-22b-distilled-1.1.safetensors")
UPSCALER = os.environ.get(
    "LTX_SPATIAL_UPSCALER", f"{MODEL_ROOT}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
GEMMA_ROOT = os.environ.get("GEMMA_ROOT", "/models/Lightricks/gemma-3-12b")
MODEL_ID = os.environ.get("SERVED_MODEL_NAME", "Lightricks/LTX-2.3")
# fp8-cast: store weights fp8, compute bf16 (~halves the 44GB transformer).
# "none"/"" = plain BF16. fp8-scaled-mm = fp8 matmuls (Hopper+/Blackwell).
QUANT = os.environ.get("LTX_QUANTIZATION", "fp8-cast").lower()
OFFLOAD = os.environ.get("LTX_OFFLOAD", "none").lower()

# Defaults: 1280x704 @ 24fps, 121 frames (~5s). Constraints: H/W divisible by
# 32 (stage 1 runs at half res → keep divisible by 64), frames = 8k+1.
DEFAULT_WIDTH = int(os.environ.get("DEFAULT_WIDTH", "1280"))
DEFAULT_HEIGHT = int(os.environ.get("DEFAULT_HEIGHT", "704"))
DEFAULT_FRAMES = int(os.environ.get("DEFAULT_FRAMES", "121"))
DEFAULT_FPS = float(os.environ.get("DEFAULT_FPS", "24"))
_CREATED = int(time.time())

app = FastAPI(title="LTX-2.3 video (OpenAI-compatible-ish)")

print(f"loading LTX-2.3 distilled: {CKPT}", flush=True)
print(f"  upscaler: {UPSCALER}", flush=True)
print(f"  gemma:    {GEMMA_ROOT}", flush=True)
print(f"  quantization={QUANT or 'none'} offload={OFFLOAD}", flush=True)

from ltx_pipelines.distilled import DistilledPipeline  # noqa: E402
from ltx_pipelines.utils.args import ImageConditioningInput  # noqa: E402
from ltx_pipelines.utils.types import OffloadMode  # noqa: E402
from ltx_pipelines.utils.media_io import encode_video  # noqa: E402
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number  # noqa: E402

quant_policy = None
if QUANT and QUANT not in ("none", "off", "bf16"):
    try:
        from ltx_pipelines.utils.quantization_factory import QuantizationKind
        quant_policy = QuantizationKind(QUANT).to_policy(checkpoint_path=CKPT)
        print(f"  quantization policy: {QUANT}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  quantization '{QUANT}' unavailable ({e}); running BF16", flush=True)

offload_mode = OffloadMode.NONE
try:
    offload_mode = OffloadMode(OFFLOAD)
except ValueError:
    print(f"  unknown offload '{OFFLOAD}', using none", flush=True)

pipe = DistilledPipeline(
    distilled_checkpoint_path=CKPT,
    spatial_upsampler_path=UPSCALER,
    gemma_root=GEMMA_ROOT,
    loras=(),
    quantization=quant_policy,
    offload_mode=offload_mode,
)
print("pipeline loaded: DistilledPipeline (LTX-2.3)", flush=True)

# One GPU, one denoise loop at a time — serialize requests.
_gpu_lock = threading.Lock()


def _snap(v, mult):
    return max(mult, int(round(v / mult)) * mult)


def _run(prompt, image_paths, width, height, num_frames, frame_rate, seed,
         enhance_prompt, frame_idxs=None, strengths=None):
    # H/W divisible by 64 (stage 1 = half res, needs /32), frames = 8k+1
    width, height = _snap(width, 64), _snap(height, 64)
    num_frames = max(9, ((num_frames - 1) // 8) * 8 + 1)
    # Conditioning frame placement: default = first image at frame 0; a second
    # image defaults to the LAST frame (start+end keyframing). Explicit
    # image_frame_idx overrides; -1 = last frame.
    images = []
    for i, p in enumerate(image_paths):
        if frame_idxs and i < len(frame_idxs):
            fi = frame_idxs[i]
        else:
            fi = 0 if i == 0 else (num_frames - 1)
        if fi < 0:
            fi = num_frames + fi  # -1 → last frame
        fi = max(0, min(fi, num_frames - 1))
        st = strengths[i] if strengths and i < len(strengths) else 1.0
        images.append(ImageConditioningInput(path=p, frame_idx=fi, strength=st))
    if images:
        print(f"  conditioning: {[(im.frame_idx, im.strength) for im in images]}",
              flush=True)
    tiling_config = TilingConfig.default()
    chunks = get_video_chunks_number(num_frames, tiling_config)
    t0 = time.time()
    with _gpu_lock, torch.inference_mode():
        video, audio = pipe(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = f.name
        try:
            encode_video(
                video=video,
                fps=frame_rate,
                audio=audio,
                output_path=out_path,
                video_chunks_number=chunks,
            )
            with open(out_path, "rb") as f:
                mp4 = f.read()
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass
    return {
        "created": int(time.time()),
        "data": [{"b64_json": base64.b64encode(mp4).decode()}],
        "_elapsed_s": round(time.time() - t0, 2),
        "_meta": {"width": width, "height": height, "num_frames": num_frames,
                  "fps": frame_rate, "bytes": len(mp4)},
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


def _model_obj():
    return {"id": MODEL_ID, "object": "model", "created": _CREATED,
            "owned_by": "Lightricks"}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [_model_obj()]}


@app.get("/v1/models/{model_id:path}")
def get_model(model_id: str):
    return _model_obj()


@app.post("/v1/videos")
async def videos(
    prompt: str = Form(...),
    # Optional conditioning image(s) → image-to-video. Both spellings.
    image: list[UploadFile] = File(default=[]),
    image_arr: list[UploadFile] = File(default=[], alias="image[]"),
    width: int = Form(DEFAULT_WIDTH),
    height: int = Form(DEFAULT_HEIGHT),
    num_frames: int = Form(DEFAULT_FRAMES),
    frame_rate: float = Form(DEFAULT_FPS),
    seed: int = Form(0),
    enhance_prompt: bool = Form(False),
    # comma-separated, matching image order: e.g. "0,-1" (-1 = last frame),
    # "1.0,0.8". Defaults: 1st image → frame 0, 2nd → last frame, strength 1.0.
    image_frame_idx: str = Form(""),
    image_strength: str = Form(""),
):
    def _ints(s):
        return [int(x) for x in s.split(",") if x.strip() != ""] if s else None

    def _floats(s):
        return [float(x) for x in s.split(",") if x.strip() != ""] if s else None

    try:
        frame_idxs, strengths = _ints(image_frame_idx), _floats(image_strength)
    except ValueError:
        raise HTTPException(400, "image_frame_idx/image_strength must be comma-separated numbers")

    files = [*image, *image_arr]
    image_paths = []
    for f in files:
        data = await f.read()
        suffix = os.path.splitext(f.filename or "")[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(data)
            image_paths.append(tf.name)

    def work():
        try:
            return _run(prompt, image_paths, width, height, num_frames,
                        frame_rate, seed, enhance_prompt, frame_idxs, strengths)
        finally:
            for p in image_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    return _streaming_json(work)
