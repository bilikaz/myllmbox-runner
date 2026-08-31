# syntax=docker/dockerfile:1.6
#
# LTX-2.3 (Lightricks) 22B audio+video DiT — distilled two-stage pipeline, served by docker/server.py
# (mbx-ltx-2.3). GB10/sm_121: torch nightly cu130, same base pattern as recipes/qwen-image-edit.
#
# The ltx-core / ltx-pipelines packages come from the LTX-2 monorepo (github.com/Lightricks/LTX-2,
# uv_build backend — pip installs via PEP 517), PINNED to the v1.3.0 release tag. The July-verified
# build used an unpinned July HEAD; the exact ref was never recorded, so this port pins the nearest
# release and gets re-verified at first run (the recipe README tracks that).
#
# torch: the trio comes from the cu130 nightly channel like the July build (LTX needs torchaudio for
# the audio VAE; the image recipes' pinned torch/vision pair has no matching torchaudio pin recorded).
# ⚠ AFTER the first green build+serve, freeze the trio: docker run --rm mbx-ltx-2.3 pip freeze | grep torch
#   and write the versions into this file — same discipline as the image recipes.
#
# Build:  ./build-and-copy.sh ltx-2.3     (or: docker build -t mbx-ltx-2.3 recipes/ltx-2.3)

FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="12.0a;12.1a" CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip git curl ca-certificates ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Nightly torch (cu130) for sm_121 — installed FIRST so ltx-core's torch compatible-release pin
# resolves against it instead of pulling a cu12x wheel. torchaudio is required (audio VAE).
RUN pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu130

# LTX-2 monorepo packages (pinned tag) + server stack. openimageio/av ship manylinux aarch64 wheels.
RUN pip install \
        "git+https://github.com/Lightricks/LTX-2.git@v1.3.0#subdirectory=packages/ltx-core" \
        "git+https://github.com/Lightricks/LTX-2.git@v1.3.0#subdirectory=packages/ltx-pipelines" \
        "transformers>=4.52" accelerate safetensors sentencepiece protobuf \
        Pillow av fastapi "uvicorn[standard]" pydantic python-multipart

WORKDIR /app
COPY docker/server.py /app/server.py

EXPOSE 8000
ENTRYPOINT []
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
