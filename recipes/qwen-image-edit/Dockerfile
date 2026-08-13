# syntax=docker/dockerfile:1.6
#
# Qwen-Image-Edit-2511 image-EDIT server (mbx-qwen-image-edit). Serves a PRE-QUANTIZED NVFP4 checkpoint made
# by ./quantize.sh (the shared mbx-quantizer image), loading it instantly via QUANT_PRELOADED — no on-the-fly
# quant, no BF16 peak. The 4-step Lightning LoRA is applied UNFUSED at serve time (BF16 adapter → needs peft).
#
# ⚠ PINS MUST MATCH quantizer/Dockerfile EXACTLY (torch/torchao/mslk/diffusers). torchao serializes FP4 as
#   versioned tensor subclasses; a mismatch → "Unable to import torchao Tensor objects" on load. The runner
#   pre-flights the quant manifest (myllmbox-quant.json) against this image before serving.
#
# Build:  ./build-and-copy.sh qwen-image-edit     (or: docker build -t mbx-qwen-image-edit recipes/qwen-image-edit)

FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="12.0a;12.1a" CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip git curl ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Nightly torch (cu130, sm_121) + torchao (NVFP4 inference_workflow) + mslk (FP4 Triton kernels) — PINNED to
# mbx-quantizer so the pre-quant's pickled FP4 tensor subclasses deserialize on load.
RUN pip install --pre torch==2.14.0.dev20260801+cu130 torchvision==0.29.0.dev20260803+cu130 --index-url https://download.pytorch.org/whl/nightly/cu130
RUN pip install torchao==0.18.0 triton
RUN pip install --pre mslk==2026.8.3+cu130 --index-url https://download.pytorch.org/whl/nightly/cu130
# diffusers 0.39.0 == the quantizer's (has QwenImageEditPlusPipeline; the transformer class must match what
# quantize saved). peft: required to load the Lightning LoRA over the quantized transformer. + server stack.
# NB: if a future Qwen-Image-Edit build needs a NEWER diffusers than 0.39.0, bump it HERE AND in
# quantizer/Dockerfile together, then re-quantize — the two must stay in lockstep.
RUN pip install diffusers==0.39.0 transformers==5.14.1 accelerate==1.14.0 \
        safetensors sentencepiece protobuf peft \
        Pillow fastapi "uvicorn[standard]" pydantic python-multipart

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 8000
ENTRYPOINT []
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
