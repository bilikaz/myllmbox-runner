# syntax=docker/dockerfile:1.6
# The unified quantizer image (mbx-quantizer) — built ONCE, drives every recipe's `quantize:` block.
# Same nightly torch + torchao (+ Triton/mslk FP4 kernels for sm_121a) stack the FLUX server uses, so it
# can NVFP4/MXFP8 both diffusion (diffusers) and LLM (transformers) models. Output → models/myllmbox/.
#   Build:  ./quantize.sh builds this automatically the first time (or: docker build -t mbx-quantizer quantizer)
#   Run:    handled by ./quantize.sh — docker run --rm ... mbx-quantizer --to <format>
FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="12.0a;12.1a" CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip git curl ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Nightly torch (cu130, sm_121) + torchao (NVFP4 inference_workflow) + mslk (the FP4 Triton kernels
# NVFP4 needs at runtime) — same pins as servers/flux2.
# PINNED — MUST match recipes/*/Dockerfile serve images exactly (torchao serializes FP4 as versioned tensor
# subclasses; a mismatch → "Unable to import torchao Tensor objects" on load). The quant manifest records
# these so the runner can pre-flight a serve image before loading.
RUN pip install --pre torch==2.14.0.dev20260801+cu130 torchvision==0.29.0.dev20260803+cu130 --index-url https://download.pytorch.org/whl/nightly/cu130
RUN pip install torchao==0.18.0 triton
RUN pip install --pre mslk==2026.8.3+cu130 --index-url https://download.pytorch.org/whl/nightly/cu130
# loaders for both model classes
RUN pip install "diffusers==0.39.0" transformers==5.14.1 accelerate==1.14.0 safetensors sentencepiece protobuf Pillow

WORKDIR /app
COPY . /app/quantizer
ENTRYPOINT ["python3", "-m", "quantizer"]
