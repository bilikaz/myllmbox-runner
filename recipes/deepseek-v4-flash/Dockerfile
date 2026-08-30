# syntax=docker/dockerfile:1.6
# ------------------------------------------------------------------------------------------------------
# Our own dspark-capable vLLM for DeepSeek-V4-Flash-0731 on DGX Spark (GB10 / SM121).
# Self-contained: builds vLLM (PR #41834 SM12x branch) + FlashInfer + DeepGEMM from source for sm_121a.
# vLLM = jasl fork tag sm120-pr-41834-stable-preview-20260802 — the SM12x-enable PR (Triton fallbacks for
# DeepGEMM ops, fp8 e8m0 upcast, Marlin cudagraph hardening) VALIDATED on this exact model + 2×GB10.
# Validated dep stack: torch 2.13.0 · triton 3.7.1 · FlashInfer 0.6.15.post1 · nccl 2.30.7 · tilelang 0.1.12.
# `docker build` this folder → mbx-deepseek-v4-flash-0731.
# ------------------------------------------------------------------------------------------------------
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
ARG BUILD_JOBS=16
ARG GPU_ARCH=12.1a
ARG NCCL_NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"

# ============================ STAGE 1: base (CUDA + torch + NCCL) ============================
FROM ${CUDA_IMAGE} AS base
ARG BUILD_JOBS
ARG GPU_ARCH
ARG NCCL_NVCC_GENCODE
ENV MAX_JOBS=${BUILD_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} NINJAFLAGS="-j${BUILD_JOBS}" MAKEFLAGS="-j${BUILD_JOBS}"
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 DG_JIT_USE_NVRTC=0 USE_CUDNN=1
ENV PIP_CACHE_DIR=/root/.cache/pip UV_CACHE_DIR=/root/.cache/uv UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1 UV_LINK_MODE=copy
ENV UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 VLLM_BASE_DIR=/workspace/vllm
ENV TORCH_CUDA_ARCH_LIST=${GPU_ARCH}
ENV PATH=/usr/lib/ccache:$PATH CCACHE_DIR=/root/.ccache CCACHE_MAXSIZE=50G CCACHE_COMPRESS=1
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_CUDA_COMPILER_LAUNCHER=ccache TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

RUN apt update && apt install -y --no-install-recommends \
      curl vim cmake build-essential ninja-build \
      libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
      python3-dev python3-pip git wget \
      libibverbs1 libibverbs-dev rdma-core \
      ccache devscripts debhelper fakeroot \
    && rm -rf /var/lib/apt/lists/* && pip install uv

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install torch==2.13.0 torchvision torchaudio triton==3.7.1 --index-url https://download.pytorch.org/whl/cu130 && \
    uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2" filelock pynvml requests tqdm

WORKDIR $VLLM_BASE_DIR
# NCCL from source for sm_121 (multi-node over the ConnectX/RoCE link), then pin the runtime lib.
RUN git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && make -j ${BUILD_JOBS} src.build NVCC_GENCODE="${NCCL_NVCC_GENCODE}" && \
    make pkg.debian.build && apt install -y --no-install-recommends --allow-downgrades --allow-change-held-packages ./build/pkg/deb/*.deb
RUN apt update && apt install -y --no-install-recommends --allow-downgrades \
      libnccl2=2.30.4-1+cuda13.2 libnccl-dev=2.30.4-1+cuda13.2 && rm -rf /var/lib/apt/lists/*

# ============================ STAGE 2: FlashInfer wheels ============================
FROM base AS flashinfer-builder
ARG GPU_ARCH
ARG FLASHINFER_REF=v0.6.15.post1
ENV FLASHINFER_CUDA_ARCH_LIST=${GPU_ARCH}
WORKDIR /workspace
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv uv pip install packaging
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && git checkout ${FLASHINFER_REF} && git submodule update --init --recursive
WORKDIR /workspace/flashinfer
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    sed -i -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' -e '/license-files/d' pyproject.toml && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    cd flashinfer-cubin && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    cd ../flashinfer-jit-cache && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

# ============================ STAGE 3: vLLM wheels ============================
FROM base AS vllm-builder
ARG GPU_ARCH
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH PROTOC_INCLUDE=/usr/include
RUN apt update && apt install -y --no-install-recommends ca-certificates pkg-config protobuf-compiler libprotobuf-dev && \
    rm -rf /var/lib/apt/lists/* && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path

# vLLM PR #41834 "Add SM12x support for DeepSeek V4 Flash" (jasl fork) — adds Triton fallbacks for the
# DeepGEMM ops + SM12x fp8 e8m0→fp32 upcast + Marlin MoE SM12.0a cudagraph hardening. Validated on
# DeepSeek-V4-Flash-0731 + 2×GB10/sm121. Stock v0.26.0 predates these fixes → the whole DeepGEMM/SM121 wall.
ARG VLLM_REF=sm120-pr-41834-stable-preview-20260802
ARG VLLM_REPO=https://github.com/jasl/vllm.git
ARG DEEPGEMM_REPO=https://github.com/deepseek-ai/DeepGEMM.git
ARG DEEPGEMM_REF=nv_dev
ENV DEEPGEMM_SRC_DIR=/workspace/DeepGEMM
WORKDIR $VLLM_BASE_DIR
RUN git clone --recursive ${VLLM_REPO} vllm && cd vllm && \
    git checkout ${VLLM_REF} && git submodule update --init --recursive
RUN git clone --recursive ${DEEPGEMM_REPO} "$DEEPGEMM_SRC_DIR" && \
    cd "$DEEPGEMM_SRC_DIR" && (git checkout ${DEEPGEMM_REF} || git checkout origin/${DEEPGEMM_REF}) && \
    git submodule update --init --recursive
WORKDIR $VLLM_BASE_DIR/vllm

# --- SM121-relevant carried patch (self-skips if 0.26 already fixed / moved the pattern) ---
# SM120 sparse-attn cooperative_topk: upstream selects it for all SM90+, but it fails to launch on SM12.x.
RUN python3 - <<'PY'
from pathlib import Path
t = Path("vllm/model_executor/layers/sparse_attn_indexer.py")
old = '''        use_cooperative_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and current_platform.has_device_capability(90)
        )'''
new = '''        device_capability = current_platform.get_device_capability()
        use_cooperative_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and device_capability is not None
            and device_capability.to_int() == 90
        )'''
if not t.exists(): print("sparse_attn_indexer.py absent; skip")
else:
    x = t.read_text()
    if "device_capability.to_int() == 90" in x: print("cooperative_topk workaround already present; skip")
    elif old in x: t.write_text(x.replace(old, new, 1)); print("Applied SM120 cooperative_topk workaround")
    else: print("cooperative_topk pattern not found; skip")
PY

# Prepare + compile vLLM wheel (drop the flashinfer pin — we install our own built wheel in the runner).
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i '/^triton\b/d' requirements/test/cuda.txt && \
    sed -i '/^fastsafetensors\b/d' requirements/test/cuda.txt && \
    uv pip install -r requirements/build/cuda.txt "setuptools-rust>=1.9.0"
RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=cargo-registry,target=/opt/cargo/registry \
    --mount=type=cache,id=cargo-git,target=/opt/cargo/git \
    VLLM_REQUIRE_RUST_FRONTEND=1 CARGO_BUILD_JOBS=${MAX_JOBS} \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

# ============================ STAGE 4: runtime (vllm serve) ============================
FROM ${CUDA_IMAGE} AS runner
ARG GPU_ARCH
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 DG_JIT_USE_NVRTC=0 USE_CUDNN=1
ENV UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1 UV_LINK_MODE=copy VLLM_BASE_DIR=/workspace/vllm
ENV TORCH_CUDA_ARCH_LIST=${GPU_ARCH} FLASHINFER_CUDA_ARCH_LIST=${GPU_ARCH} TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings PATH=$VLLM_BASE_DIR:$PATH

RUN --mount=type=bind,from=base,source=/workspace/vllm/nccl/build/pkg/deb,target=/workspace/nccl-pkg \
    apt update && apt install -y --no-install-recommends \
      python3 python3-pip python3-dev vim curl git wget \
      libcudnn9-cuda-13 libibverbs1 libibverbs-dev rdma-core libxcb1 earlyoom && \
    cd /workspace/nccl-pkg && apt install -y --no-install-recommends --allow-downgrades --allow-change-held-packages ./*.deb && \
    rm -rf /var/lib/apt/lists/* && pip install uv

WORKDIR $VLLM_BASE_DIR
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install torch==2.13.0 torchvision torchaudio triton==3.7.1 --index-url https://download.pytorch.org/whl/cu130 && \
    uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

# Install our built FlashInfer + vLLM wheels (pin torch so transitive deps can't swap the CUDA build).
COPY --from=flashinfer-builder /workspace/wheels /workspace/wheels
COPY --from=vllm-builder /workspace/wheels /workspace/wheels
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    PINNED_TORCH=$(python3 -c "import torch; print(torch.__version__)") && \
    echo "torch==${PINNED_TORCH}" > /tmp/ov.txt && \
    echo "fastapi[standard]>=0.115.0,<0.137.0" >> /tmp/ov.txt && \
    uv pip install /workspace/wheels/*.whl --override /tmp/ov.txt && \
    uv pip install ray[default] fastsafetensors instanttensor "nvidia-cutlass-dsl>=4.5.2" tilelang==0.1.12 --override /tmp/ov.txt

# Point vLLM's bundled NCCL at the sm_121 lib we built.
RUN rm -f /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 && \
    ln -s /usr/lib/aarch64-linux-gnu/libnccl.so.2 /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2

ENTRYPOINT ["vllm", "serve"]
