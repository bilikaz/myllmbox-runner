# syntax=docker/dockerfile:1.6
#
# EXPERIMENTAL: FLUX.2 with ON-THE-FLY torchao NVFP4 (the real-speedup path).
# Separate image from flux2-klein:latest so the nightly torch+torchao stack
# does NOT disturb the working BF16 setup. Same server (flux2-server/server.py);
# the TORCHAO_NVFP4=1 env triggers the torchao quant branch.
#
# Per the PyTorch Blackwell blog (2026-04): NVFP4DynamicActivationNVFP4Weight +
# use_triton_kernel=True → ~1.6x on FLUX. Triton JITs for sm_121a at runtime,
# so it should engage on GB10 despite prebuilt wheels being 120f.
#
# Build:   docker build -f Dockerfile.flux2torchao -t flux2-torchao:latest .
# Run:     ./run-recipe.sh flux2-dev-torchao.yaml --name flux2 --solo -d

FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# 12.1a target so any AOT bits get native NVFP4 MMA on GB10 (sm_121a)
ENV TORCH_CUDA_ARCH_LIST="12.0a;12.1a"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip \
        git curl ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Nightly torch (cu130) for sm_121. torchao is NOT on the cu130 nightly index
# (the Apr-2026 blog command is stale) — it's CUDA-version-independent and
# builds against the installed torch, so pull it from PyPI (latest stable has
# the NVFP4 inference_workflow). If the NVFP4 API is missing/renamed, swap to a
# nightly source: pip install --pre torchao --index-url .../nightly/cu128
# PINNED — these MUST match quantizer/Dockerfile exactly. torchao serializes its FP4 weights as versioned
# tensor subclasses; a torchao mismatch between the quantizer (that wrote the checkpoint) and this serve
# image → "Unable to import torchao Tensor objects" and the load hangs. Pin so a myllmbox/ quant loads on
# any box that uses these same pins. (runner pre-flights the quant's manifest against the image.)
RUN pip install --pre torch==2.14.0.dev20260801+cu130 torchvision==0.29.0.dev20260803+cu130 \
        --index-url https://download.pytorch.org/whl/nightly/cu130
RUN pip install torchao==0.18.0 triton
# MSLK provides the FP4 Triton kernels torchao NVFP4 needs at runtime.
RUN pip install --pre mslk==2026.8.3+cu130 --index-url https://download.pytorch.org/whl/nightly/cu130

RUN pip install \
        "diffusers==0.39.0" \
        transformers==5.14.1 accelerate==1.14.0 safetensors sentencepiece protobuf \
        Pillow fastapi "uvicorn[standard]" pydantic python-multipart

WORKDIR /app
COPY server.py /app/server.py

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8001
ENTRYPOINT []
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
