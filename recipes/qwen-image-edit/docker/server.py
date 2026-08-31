"""OpenAI-compatible image-EDIT server for Qwen/Qwen-Image-Edit-2511.

Qwen-Image-Edit-2511: Qwen-Image MMDiT edit model (multi-image reference edit /
fusion). Served via diffusers QwenImageEditPlusPipeline (offline from /models).
Loads a PRE-QUANTIZED NVFP4 transformer made by ./quantize.sh (QUANT_PRELOADED=1)
— or, unset, quantizes on the fly / runs BF16 (same machinery as flux2-dev).
Modeled on the FLUX.2 server; the deltas are Qwen's call signature
(true_cfg_scale + negative_prompt + guidance_scale=1.0, multi-image `image=[...]`)
and that this is an EDIT model — the pipeline always needs an input image (generations
synthesizes a blank canvas so text→image works anyway).

Endpoints mirror the OpenAI Images API so standard OpenAI clients work:
  POST /v1/images/edits        (multipart form)   image edit / multi-reference  [primary]
  POST /v1/images/generations  (JSON)             text->image: the server synthesizes a blank
                                                  canvas and routes through the edit path
  GET  /health , GET /v1/models

Bearer gating happens at the keepalive-proxy hop, not here.
"""
import base64
import io
import json
import os
import threading
import time

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

KEEPALIVE_SECS = int(os.environ.get("KEEPALIVE_SECS", "15"))


def _streaming_json(work):
    """Run work() in a thread; emit a space every KEEPALIVE_SECS while it runs so
    Cloudflare's ~100s idle timeout never fires on long (minutes) generation, then
    emit the JSON body. Leading whitespace is valid JSON — clients ignore it.
    Status is 200 as soon as streaming starts, so errors go in the body as
    {"error": ...}."""
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


MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen/Qwen-Image-Edit-2511")
MODEL_ID = os.environ.get("SERVED_MODEL_NAME", "Qwen/Qwen-Image-Edit-2511")
# Qwen-Image-Edit defaults (NOT distilled): ~40 steps, true_cfg_scale ~4.0,
# guidance_scale 1.0 (the model's distilled guidance is off; the real classifier-
# free guidance is true_cfg_scale), negative_prompt " ". The recipe overrides
# these for the Lightning LoRA (steps 8, true_cfg 1.0).
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "40"))
DEFAULT_TRUE_CFG = float(os.environ.get("DEFAULT_TRUE_CFG", "4.0"))
DEFAULT_GUIDANCE = float(os.environ.get("DEFAULT_GUIDANCE", "1.0"))
DEFAULT_NEGATIVE = os.environ.get("DEFAULT_NEGATIVE_PROMPT", " ")
_CREATED = int(time.time())

app = FastAPI(title="Qwen-Image-Edit (OpenAI-compatible)")

print(f"loading {MODEL_PATH} (default steps={DEFAULT_STEPS}) ...", flush=True)
# DiffusionPipeline auto-detects the concrete class (QwenImageEditPlusPipeline)
# from the checkpoint's model_index.json.
from diffusers import DiffusionPipeline

# TORCHAO_QUANT selects on-the-fly torchao quant of the transformer:
#   nvfp4 → W4A4 fp4, fastest (~4x on GB10), Triton kernels (needs mslk)
#   mxfp8 → W8A8 fp8, higher accuracy, a bit less speed
#   (unset/off/bf16) → no quant, plain BF16
# Loads BF16 then quantizes, so MODEL_PATH is the normal BF16 repo — UNLESS
# QUANT_PRELOADED is set (below), in which case MODEL_PATH is a pre-quant.
TORCHAO_QUANT = (os.environ.get("TORCHAO_QUANT")
                 or ("nvfp4" if os.environ.get("TORCHAO_NVFP4") else "")).lower()

# AUTODETECTED: our quantizer stamps every pre-quant with myllmbox-quant.json — its presence,
# not an env knob, decides the loading path. (QUANT_PRELOADED env kept as a manual override.)
if os.environ.get("QUANT_PRELOADED") or os.path.exists(os.path.join(MODEL_PATH, "myllmbox-quant.json")):
    # Already quantized offline by ./quantize.sh (MODEL_PATH = models/myllmbox/<name>-<fmt>). Load the
    # quantized transformer with use_safetensors=False (torchao is pickled .bin) but let the rest of the
    # pipeline (text_encoder/vae — safetensors) load normally. import torchao first so the pickled FP4
    # tensor subclasses register, else the weights won't deserialize.
    import torchao  # noqa: F401
    from torchao.prototype.mx_formats import inference_workflow  # noqa: F401
    from diffusers import QwenImageTransformer2DModel
    print(f"loading PRE-QUANTIZED transformer from {MODEL_PATH}/transformer (instant, no on-the-fly quant)", flush=True)
    tf = QwenImageTransformer2DModel.from_pretrained(
        os.path.join(MODEL_PATH, "transformer"), torch_dtype=torch.bfloat16, use_safetensors=False)
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH, transformer=tf, torch_dtype=torch.bfloat16).to("cuda")
elif TORCHAO_QUANT in ("nvfp4", "mxfp8"):
    from diffusers import TorchAoConfig, PipelineQuantizationConfig
    if TORCHAO_QUANT == "nvfp4":
        from torchao.prototype.mx_formats.inference_workflow import (
            NVFP4DynamicActivationNVFP4WeightConfig,
        )
        ao_cfg = NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True, use_triton_kernel=True)
    else:  # mxfp8
        from torchao.prototype.mx_formats.inference_workflow import (
            MXDynamicActivationMXWeightConfig,
        )
        from torchao.prototype.mx_formats.constants import KernelPreference
        ao_cfg = MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float8_e4m3fn, weight_dtype=torch.float8_e4m3fn,
            kernel_preference=KernelPreference.AUTO)
    # MIXED / selective quant: keep accuracy-critical linears (embeddings, final
    # projection, norms-with-linear) in BF16; NVFP4 the heavy blocks. Substring
    # match; override via TORCHAO_SKIP_MODULES ("" = full quant).
    skip = [s.strip() for s in os.environ.get(
        "TORCHAO_SKIP_MODULES",
        "img_in,txt_in,time_text_embed,norm_out,proj_out").split(",") if s.strip()]

    # CONTENT-ADDRESSED cache: key by (model, quant, skip) → each config gets its
    # own subdir. First boot of a key: BF16 load + quant + save; later: fast load.
    QUANT_CACHE = os.environ.get("QUANT_CACHE_DIR")
    cache_path = None
    if QUANT_CACHE:
        import hashlib
        key = hashlib.md5(
            f"{MODEL_PATH}|{TORCHAO_QUANT}|{','.join(skip)}".encode()).hexdigest()[:12]
        cache_path = os.path.join(QUANT_CACHE, key)
        print(f"  quant cache key {key} ({TORCHAO_QUANT}, skip={skip or 'none'})", flush=True)

    loaded = False
    if cache_path and os.path.isdir(cache_path) and os.listdir(cache_path):
        try:
            from diffusers import QwenImageTransformer2DModel
            print(f"  loading cached transformer from {cache_path}", flush=True)
            # use_safetensors=False: quant is saved as pickled (sharded) .bin.
            tf = QwenImageTransformer2DModel.from_pretrained(
                cache_path, torch_dtype=torch.bfloat16, use_safetensors=False)
            pipe = DiffusionPipeline.from_pretrained(
                MODEL_PATH, transformer=tf, torch_dtype=torch.bfloat16).to("cuda")
            loaded = True
        except Exception as e:  # noqa: BLE001
            print(f"  cache load failed ({e}); quantizing fresh", flush=True)
    if not loaded:
        try:
            tao = TorchAoConfig(ao_cfg, modules_to_not_convert=skip) if skip else TorchAoConfig(ao_cfg)
        except TypeError:
            print("  (modules_to_not_convert unsupported here; full quant)", flush=True)
            tao = TorchAoConfig(ao_cfg)
        print(f"  quantizing on-the-fly with torchao {TORCHAO_QUANT.upper()} (triton); "
              f"BF16-kept: {skip or 'none (full)'}", flush=True)
        pqc = PipelineQuantizationConfig(quant_mapping={"transformer": tao})
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, quantization_config=pqc).to("cuda")
        if cache_path:
            try:
                os.makedirs(cache_path, exist_ok=True)
                pipe.transformer.save_pretrained(cache_path, safe_serialization=False)
                print(f"  saved quantized transformer → {cache_path} (fast boot next time)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  cache save failed ({e}); staying on-the-fly", flush=True)
else:
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda")
print(f"pipeline loaded: {type(pipe).__name__}", flush=True)

# Few-step distillation LoRA (e.g. lightx2v/Qwen-Image-Edit-2511-Lightning → 4-step,
# ~10x fewer steps). Loaded AFTER quant as an UNFUSED adapter: fusing into NVFP4
# weights isn't supported, so the low-rank path runs live in BF16 — negligible over
# 4-8 steps. Loaded BEFORE compile so it's captured in the graph.
# ⚠ With a Lightning LoRA you MUST run few steps + CFG OFF: set DEFAULT_STEPS=4 (or 8)
# and DEFAULT_TRUE_CFG=1.0 in the recipe. Leaving CFG on (true_cfg_scale>1) doubles the
# work AND fights the distillation → washed-out/garbled output.
LORA_PATH = os.environ.get("LORA_PATH", "")
if LORA_PATH:
    # LORA_PATH = the exact .safetensors file (a LoRA repo ships several — 4-step / 8-step — so one is named).
    # diffusers wants a dir + weight_name, so split it.
    lora_dir, lora_file = os.path.dirname(LORA_PATH), os.path.basename(LORA_PATH)
    print(f"  loading LoRA {lora_file} (from {lora_dir}) ...", flush=True)
    try:
        pipe.load_lora_weights(lora_dir, weight_name=lora_file)
        print("  LoRA loaded (unfused adapter)", flush=True)
    except Exception as e:  # noqa: BLE001
        # If diffusers can't attach a LoRA to the torchao-quantized transformer,
        # fall back to BF16 (TORCHAO_QUANT unset) + LoRA, or fuse-then-quantize.
        print(f"  LoRA load FAILED ({e}); continuing WITHOUT it (full-step model)", flush=True)
    else:
        # Lightning LoRAs are trained against a FIXED few-step FlowMatch schedule
        # (ModelTC/Qwen-Image-Lightning generate_with_diffusers.py): exponential
        # dynamic shifting with base_shift = max_shift = ln(3). The pipeline's
        # stock scheduler undersamples high-frequency detail at 4-8 steps →
        # crunchy "oil-painting" fur / posterized bokeh. Swap it in whenever a
        # Lightning LoRA is attached. LIGHTNING_SCHEDULER=0 keeps the stock
        # scheduler (A/B knob).
        if os.environ.get("LIGHTNING_SCHEDULER", "1") == "0":
            print("  scheduler swap DISABLED (LIGHTNING_SCHEDULER=0) — stock scheduler", flush=True)
        else:
            import math
            from diffusers import FlowMatchEulerDiscreteScheduler
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config({
                "base_image_seq_len": 256,
                "base_shift": math.log(3),
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            })
            print("  scheduler → FlowMatchEuler (Lightning: exp dyn-shift, ln(3))", flush=True)

# Compile for steady-state speed (both quant and BF16 paths). Each NEW image shape
# pays a one-time Triton JIT stall; COMPILE_BLOCKS=0 avoids stalls (~10-15% slower).
if os.environ.get("COMPILE_BLOCKS", "1") != "0":
    try:
        pipe.transformer.compile_repeated_blocks(fullgraph=True)
        print("  compiled repeated blocks", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  compile_repeated_blocks skipped: {e}", flush=True)
else:
    print("  compile_repeated_blocks disabled (COMPILE_BLOCKS=0)", flush=True)

# A diffusers pipeline is not thread-safe and one GPU can't run concurrent
# denoise loops. Serialize: concurrent callers queue and run one at a time.
_gpu_lock = threading.Lock()


# ⚠ EDIT RESOLUTION FLOOR (refined 2026-07-20 evening): edits COLLAPSE to a
# neon-posterized identity copy when EITHER output dimension is <= 1024 (the
# pipeline's internal condition grid) — it is NOT total area: 2048x1024 (2.1MP)
# breaks while 1328x1328 (1.76MP) is fine. Min dimension >= 1088 verified clean
# (2048x1088, 2176x1088, 1920x1072, 1328², 2048²). Generation (blank canvas)
# is unaffected. size=auto targets EDIT_AUTO_PIXELS at input aspect; explicit
# sizes go to the pipeline VERBATIM (floored to /16 by diffusers) — client owns
# the size rules by policy. size=input restores raw pipeline behavior.
EDIT_AUTO_PIXELS = int(os.environ.get("EDIT_AUTO_PIXELS", str(1328 * 1328)))


def _parse_size(size):
    """Optional output size. 'auto' (default) → EDIT_AUTO_PIXELS at input aspect
    (see note above); 'input' → let the pipeline derive from the input image."""
    if not size or size.lower() in ("auto", "input", ""):
        return None, None
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise HTTPException(400, f"bad size '{size}', want WIDTHxHEIGHT e.g. 1328x1328, 'auto' or 'input'")


def _auto_size(ref, size):
    """For size=auto: scale the first input image's aspect to EDIT_AUTO_PIXELS
    total pixels, snapped to multiples of 16."""
    if EDIT_AUTO_PIXELS <= 0 or (size and size.lower() == "input"):
        return None, None
    import math
    iw, ih = ref.size
    ar = iw / ih
    w = max(16, int(round(math.sqrt(EDIT_AUTO_PIXELS * ar) / 16)) * 16)
    h = max(16, int(round(math.sqrt(EDIT_AUTO_PIXELS / ar) / 16)) * 16)
    return w, h


def _b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# BATCHING: generation is memory-bandwidth bound on GB10 — a batch shares one
# weight-stream per step. BATCH_PIXEL_BUDGET caps batch_size*W*H so big batches
# can't blow the UMA ceiling. MAX_N: GB10 gets no batching speedup (48 SMs
# saturate on one 2MP image) → run MAX_N=1 via recipe env; raise on high-SM cards.
BATCH_PIXEL_BUDGET = int(os.environ.get("BATCH_PIXEL_BUDGET", str(6_000_000)))
MAX_N = int(os.environ.get("MAX_N", "4"))


def _run(prompt, refs, steps, true_cfg, guidance, negative, w, h, seed, n):
    """Generate n images (seed+i each), batching them through the denoise loop
    in chunks that fit BATCH_PIXEL_BUDGET. `refs` = input PIL images."""
    n = min(max(1, n), MAX_N)
    px = (w * h) if (w and h) else EDIT_AUTO_PIXELS or (1024 * 1024)
    max_batch = max(1, BATCH_PIXEL_BUDGET // px)
    out_imgs = []
    i = 0
    while i < n:
        bs = min(max_batch, n - i)
        gens = [torch.Generator(device="cuda").manual_seed(seed + i + j)
                for j in range(bs)]
        kwargs = dict(
            image=refs,                       # multi-image edit: pass the list
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            true_cfg_scale=true_cfg,          # the real CFG for Qwen-Image-Edit
            guidance_scale=guidance,          # distilled guidance (1.0 = off)
            num_images_per_prompt=bs,
            generator=gens if bs > 1 else gens[0],
        )
        if w and h:                           # only override size if requested
            kwargs["width"], kwargs["height"] = w, h
        try:
            with _gpu_lock, torch.inference_mode():
                out_imgs.extend(pipe(**kwargs).images)
        except Exception as e:
            raise HTTPException(500, f"{type(e).__name__}: {e}")
        i += bs
    return out_imgs


def _payload(imgs, t0):
    return {
        "created": int(time.time()),
        "data": [{"b64_json": _b64_png(im)} for im in imgs],
        "_elapsed_s": round(time.time() - t0, 2),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


def _model_obj():
    return {"id": MODEL_ID, "object": "model", "created": _CREATED, "owned_by": "Qwen"}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [_model_obj()]}


@app.get("/v1/models/{model_id:path}")
def get_model(model_id: str):
    return _model_obj()


class GenerationsRequest(BaseModel):
    prompt: str
    model: str | None = None
    n: int = 1
    size: str = "auto"
    response_format: str = "b64_json"
    seed: int = 0
    steps: int = DEFAULT_STEPS
    true_cfg_scale: float = DEFAULT_TRUE_CFG
    guidance: float = DEFAULT_GUIDANCE
    negative_prompt: str = DEFAULT_NEGATIVE


# Text→image on an EDIT model: the checkpoint always wants a conditioning image, so the
# server synthesizes a near-white canvas at the output size and routes through the same
# edit path — the gauntlet-proven trick (bench/gauntlet-image.py --mode edit fed exactly
# this canvas and the renders beat FLUX), now server-side so any stock Images-API client
# just works. A blank canvas still conditions (its tone/aspect seed the composition),
# and generation is immune to the ≤1024 edit floor (see the note above _parse_size).
GEN_CANVAS_RGB = (250, 250, 248)


@app.post("/v1/images/generations")
def images_generations(req: GenerationsRequest):
    t0 = time.time()
    w, h = _parse_size(req.size)
    if w is None:  # no input image to inherit an aspect from → square at EDIT_AUTO_PIXELS
        side = max(16, int(round((EDIT_AUTO_PIXELS or 1328 * 1328) ** 0.5 / 16)) * 16)
        w = h = side
    canvas = Image.new("RGB", (w, h), GEN_CANVAS_RGB)
    return _streaming_json(
        lambda: _payload(
            _run(req.prompt, [canvas], req.steps, req.true_cfg_scale, req.guidance,
                 req.negative_prompt, w, h, req.seed, req.n), t0))


@app.post("/v1/images/edits")
async def images_edits(
    prompt: str = Form(...),
    # Reference images arrive under two spellings: repeated `image` (curl/FastAPI)
    # and `image[]` (OpenAI SDK array serialization). Accept both and merge.
    image: list[UploadFile] = File(default=[]),
    image_arr: list[UploadFile] = File(default=[], alias="image[]"),
    model: str | None = Form(None),
    n: int = Form(1),
    size: str = Form("auto"),
    response_format: str = Form("b64_json"),
    seed: int = Form(0),
    steps: int = Form(DEFAULT_STEPS),
    true_cfg_scale: float = Form(DEFAULT_TRUE_CFG),
    guidance: float = Form(DEFAULT_GUIDANCE),
    negative_prompt: str = Form(DEFAULT_NEGATIVE),
):
    t0 = time.time()
    files = [*image, *image_arr]
    if not files:
        raise HTTPException(422, "at least one input image is required (multipart field `image`, or `image[]`)")
    try:
        refs = [Image.open(io.BytesIO(await f.read())).convert("RGB") for f in files]
    except Exception as e:
        raise HTTPException(400, f"cannot decode image: {e}")
    w, h = _parse_size(size)
    if w is None:
        w, h = _auto_size(refs[0], size)
    return _streaming_json(
        lambda: _payload(
            _run(prompt, refs, steps, true_cfg_scale, guidance, negative_prompt,
                 w, h, seed, n), t0))
