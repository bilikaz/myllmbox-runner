# syntax=docker/dockerfile:1.6
# ======================================================================================================
# Our own B12X-optimized vLLM for DeepSeek-V4-Flash on DGX Spark (GB10 / SM121).
# Self-contained: builds vLLM (the B12X fork) + FlashInfer + DeepGEMM from source for sm_121a, then
# installs the external B12X CUDA-kernel package. `docker build` this folder → mbx-deepseek-v4-flash.
#
# This is a PORT of the eugr fork's `--exp-b12x` build path (build-and-copy.sh + Dockerfile on
# github.com/bilikaz/spark-vllm-docker @ fork/main). `--exp-b12x` is a preset that pins:
#   * vLLM   = https://github.com/local-inference-lab/vllm   ref  dev/gilded-gnosis
#             — the B12X vLLM fork. It carries, IN-TREE, the b12x serving stack this recipe uses:
#               --moe-backend b12x, --linear-backend b12x, --attention-backend B12X_MLA_SPARSE
#               (sparse-MLA attention), the sparse indexer, MHC (multi-head compression),
#               the `dspark` speculative method, and the `instanttensor` load-format.
#   * B12X   = https://github.com/lukealonso/b12x.git        ref  master
#             — the external B12X CUDA-kernel Python package. Installed --no-deps into the runner;
#               its kernels are JIT-compiled on first use (building the wheel does NOT compile CUDA).
#   * torch 2.12.0 / torchvision 0.27.0 / NO torchaudio  (B12X requires torch >= 2.12).
#   * DeepGEMM pinned at a6b593d2… (last-known-good before the SM121 MXFP4 grouped-scale regression).
#
# vs our Marlin reference recipes/deepseek-v4-flash-0731/Dockerfile — see the "DEP PIN DELTAS" note
# at the bottom. Validated dep stack is NOT re-verified here; this Dockerfile is authoring-only.
# ======================================================================================================
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
ARG BUILD_JOBS=16
ARG GPU_ARCH=12.1a
ARG NCCL_NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"

# B12X preset torch stack (build-and-copy.sh: EXP_B12X_TORCH*_VERSION). torchaudio is intentionally absent.
ARG TORCH_VERSION=2.12.0
ARG TORCHVISION_VERSION=0.27.0

# ============================ STAGE 1: base (CUDA + torch + NCCL) ============================
FROM ${CUDA_IMAGE} AS base
ARG BUILD_JOBS
ARG GPU_ARCH
ARG NCCL_NVCC_GENCODE
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
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

# B12X torch stack: torch 2.12.0 + torchvision 0.27.0 (NO torchaudio), triton unpinned, from the cu130 index.
# apache-tvm-ffi pinned to 0.1.12 (upstream B12X pin; our 0731 Marlin uses "<0.2").
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} triton --index-url https://download.pytorch.org/whl/cu130 && \
    uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi==0.1.12" filelock pynvml requests tqdm

WORKDIR $VLLM_BASE_DIR
# NCCL from source for sm_121 (multi-node over the ConnectX/RoCE link). Mirrors the upstream B12X build:
# build NVIDIA/nccl HEAD and install the produced .deb (no secondary libnccl version pin — that pin is a
# 0731-Marlin-specific step; see DEP PIN DELTAS).
RUN git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && make -j ${BUILD_JOBS} src.build NVCC_GENCODE="${NCCL_NVCC_GENCODE}" && \
    make pkg.debian.build && apt install -y --no-install-recommends --allow-downgrades --allow-change-held-packages ./build/pkg/deb/*.deb

# ============================ STAGE 2: FlashInfer wheels ============================
FROM base AS flashinfer-builder
ARG GPU_ARCH
# --exp-b12x does NOT pin FlashInfer (wrapper default = main). PINNED to flashinfer-ai/flashinfer main tip
# resolved 2026-08-13. (Our 0731 Marlin build pins v0.6.15.post1.) To take latest: --build-arg FLASHINFER_REF=main.
ARG FLASHINFER_REF=53a1c3bd7a3f384d3a8fd1261cc99d80bcf9d188
ENV FLASHINFER_CUDA_ARCH_LIST=${GPU_ARCH}
WORKDIR /workspace
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv uv pip install packaging
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git && \
    cd flashinfer && git checkout ${FLASHINFER_REF} && git submodule update --init --recursive
WORKDIR /workspace/flashinfer
# FlashInfer main (0.6.18) ships a `.python-version` pinning 3.10, so `uv build` would run under a bare managed
# 3.10 with no filelock — and flashinfer-cubin's build backend imports flashinfer→jit→cubin_loader→filelock
# ("ModuleNotFoundError: No module named 'filelock'"). Drop the pins and force the SYSTEM python 3.12 (which the
# base stage populated with filelock/requests/tqdm). Wheels are py3-none-any, so the interpreter version is moot.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    sed -i -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' -e '/license-files/d' pyproject.toml && \
    rm -f .python-version flashinfer-cubin/.python-version flashinfer-jit-cache/.python-version && \
    PY="$(command -v python3)" && \
    uv build --no-build-isolation --python "$PY" --wheel . --out-dir=/workspace/wheels -v && \
    cd flashinfer-cubin && uv build --no-build-isolation --python "$PY" --wheel . --out-dir=/workspace/wheels -v && \
    cd ../flashinfer-jit-cache && uv build --no-build-isolation --python "$PY" --wheel . --out-dir=/workspace/wheels -v

# ============================ STAGE 3: vLLM wheels (B12X fork) ============================
FROM base AS vllm-builder
ARG GPU_ARCH
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH PROTOC_INCLUDE=/usr/include
RUN apt update && apt install -y --no-install-recommends ca-certificates pkg-config protobuf-compiler libprotobuf-dev && \
    rm -rf /var/lib/apt/lists/* && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path

# THE B12X vLLM FORK. `--exp-b12x` sets these exact values (build-and-copy.sh EXP_B12X_VLLM_*).
# This fork carries the entire B12X serving stack in-tree (b12x moe/linear backends, B12X_MLA_SPARSE
# sparse-attention backend, sparse indexer, MHC, dspark spec method, instanttensor load-format).
ARG VLLM_REPO=https://github.com/local-inference-lab/vllm.git
# PINNED to the dev/gilded-gnosis tip resolved 2026-08-13 (branch is a moving ref). To take latest: pass
# --build-arg VLLM_REF=dev/gilded-gnosis. Re-pin after a validated build.
ARG VLLM_REF=fa033bd4e1b16d9d729ad94be2d87da5a13210ce
# DeepGEMM pinned SHA (upstream Dockerfile ARG DEEPGEMM_REF) — last known good before the SM121
# DeepSeek-V4 MXFP4 grouped scale-factor regression at nv_dev f8e8fb5 (PR #384).
ARG DEEPGEMM_REPO=https://github.com/deepseek-ai/DeepGEMM.git
ARG DEEPGEMM_REF=a6b593d2826719dcf4892609af7b84ee23aaf32a
ENV DEEPGEMM_SRC_DIR=/workspace/DeepGEMM
WORKDIR $VLLM_BASE_DIR
RUN git clone --recursive ${VLLM_REPO} vllm && cd vllm && \
    (git checkout ${VLLM_REF} || git checkout origin/${VLLM_REF}) && git submodule update --init --recursive
RUN git clone --recursive ${DEEPGEMM_REPO} "$DEEPGEMM_SRC_DIR" && \
    cd "$DEEPGEMM_SRC_DIR" && (git checkout ${DEEPGEMM_REF} || git checkout origin/${DEEPGEMM_REF}) && \
    git submodule update --init --recursive
WORKDIR $VLLM_BASE_DIR/vllm

# --- Build-critical SM12x patch #1: preserve the selected Blackwell subarch target. ---
# Port of docker/patch_vllm_preserve_sm12x_target.py (the upstream `--exp-b12x` build runs this with
# VLLM_PRESERVE_SM12X_TARGET=1). CUDA-13 vLLM CMake otherwise collapses 12.1a→12.0, dropping sm_121a.
# Idempotent + self-checking on the CUDA_SUPPORTED_ARCHS line.
RUN python3 - <<'PY'
from pathlib import Path
t = Path("CMakeLists.txt")
default = 'set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")'
sm121   = 'set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1")'
full    = 'set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;11.0;12.0;12.1")'
x = t.read_text()
if full in x: print("SM12x/SM103 target allow-list already present; skip")
elif x.count(sm121) == 1: t.write_text(x.replace(sm121, full, 1)); print("Enabled SM103+SM12x targets")
elif x.count(default) == 1: t.write_text(x.replace(default, full, 1)); print("Enabled SM103+SM12x targets")
else: raise SystemExit("CUDA_SUPPORTED_ARCHS anchor not found exactly once — verify fork CMakeLists")
PY

# --- SM12x patch #2: restrict sparse-attn cooperative_topk to SM90 (DS-V4 uses sparse MLA). ---
# Port of docker/patch_vllm_sm120_cooperative_topk.py. Upstream selects cooperative_topk for all SM90+,
# but it fails to launch on SM12.x ("cooperative_topk launch failed: invalid argument"). Self-skips if
# the fork already carries the fix or has moved the selector.
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

# NOTE on the OTHER upstream docker/patch_vllm_*.py scripts:
#   The generic eugr Dockerfile also runs patch_vllm_flashinfer_b12x_swigluoai.py (a subset of vLLM
#   PR #47392 for the *flashinfer_b12x* MoE backend — NOT the `b12x` moe-backend this recipe uses),
#   patch_vllm_spark_kv_cache_cleanup.py (UMA warmup allocator cleanup), patch_vllm_topk_softplus_sqrt
#   and several model-specific patches (diffusion-gemma / minimax / gemma4-mtp / qwen autogptq).
#   They are all idempotent + self-skipping and are expected to be either fork-native on
#   dev/gilded-gnosis or off this recipe's code path. They are NOT inlined here — VERIFY before a real
#   build whether spark_kv_cache_cleanup / topk_softplus_sqrt are still needed on this fork.

# Prepare + compile the vLLM wheel (drop the flashinfer pin — we install our own built wheel in the runner).
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
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
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

# Same B12X torch stack as the builders (torch 2.12.0 / torchvision 0.27.0, no torchaudio).
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} triton --index-url https://download.pytorch.org/whl/cu130 && \
    uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi==0.1.12"

# Install our built FlashInfer + vLLM wheels (pin torch so transitive deps can't swap the CUDA build).
COPY --from=flashinfer-builder /workspace/wheels /workspace/wheels
COPY --from=vllm-builder /workspace/wheels /workspace/wheels
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    PINNED_TORCH=$(python3 -c "import torch; print(torch.__version__)") && \
    PINNED_TORCHVISION=$(python3 -c "import importlib.metadata as m; print(m.version('torchvision'))") && \
    echo "torch==${PINNED_TORCH}" > /tmp/ov.txt && \
    echo "torchvision==${PINNED_TORCHVISION}" >> /tmp/ov.txt && \
    echo "fastapi[standard]>=0.115.0,<0.137.0" >> /tmp/ov.txt && \
    uv pip install /workspace/wheels/*.whl --override /tmp/ov.txt && \
    uv pip install ray[default] fastsafetensors instanttensor --override /tmp/ov.txt

# --- The external B12X CUDA-kernel package (lukealonso/b12x) ---
# `--exp-b12x` installs this AFTER vLLM, --no-deps: the vLLM fork already provides/pins the runtime deps
# (incl. API-sensitive nvidia-cutlass-dsl). B12X kernels JIT-compile on first use; this only lands the
# Python package. CACHEBUST arg forces a re-clone when the master branch moves.
ARG B12X_REPO=https://github.com/lukealonso/b12x.git
# PINNED to lukealonso/b12x master tip resolved 2026-08-13 (moving ref). To take latest: --build-arg B12X_REF=master.
ARG B12X_REF=954fd0174c49502b62547a01f09c404029e6035a
ARG B12X_CACHEBUST=0
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    echo "B12X cachebust: ${B12X_CACHEBUST}" && \
    git clone "${B12X_REPO}" /tmp/b12x-source && git -C /tmp/b12x-source checkout "${B12X_REF}" && \
    git -C /tmp/b12x-source rev-parse HEAD > /workspace/b12x-source-commit && \
    uv pip install --reinstall --no-deps /tmp/b12x-source && \
    python3 -c "import importlib.metadata as m; import b12x; print('Verified B12X', m.version('b12x'), 'CUTLASS DSL', m.version('nvidia-cutlass-dsl'))" && \
    rm -rf /tmp/b12x-source

# --- Bake in the recipe's required mod: instanttensor-hybrid-draft-loader ---
# (mods/instanttensor-hybrid-draft-loader, listed in recipes/deepseek-v4-flash-0731.yaml). Normally
# applied at container launch; baked here because it patches the INSTALLED vLLM. It keeps InstantTensor
# for the target model but lets a same-checkpoint speculative draft (the dspark/MTP draft this recipe
# runs) load via lazy safetensors, avoiding a second full InstantTensor GPU-streaming pass.
# Runtime knob preserved: INSTANTTENSOR_DRAFT_LOADER = auto (default) | safetensors | instanttensor.
# Verbatim port of mods/instanttensor-hybrid-draft-loader/patch_model_loader.py (ast-validated, idempotent).
RUN python3 - "$(python3 -c 'import vllm,os;print(os.path.join(os.path.dirname(vllm.__file__),"model_executor/model_loader/__init__.py"))')" <<'PY'
from __future__ import annotations
import ast, sys
from pathlib import Path

MARKER = "# spark-vllm mod: instanttensor-hybrid-draft-loader v1"

GET_MODEL_ANCHOR = """
def get_model(
    *,
    vllm_config: VllmConfig,
    model_config: ModelConfig | None = None,
    prefix: str = "",
    load_config: LoadConfig | None = None,
) -> nn.Module:
    loader = get_model_loader(load_config or vllm_config.load_config)
    if model_config is None:
        model_config = vllm_config.model_config
    return loader.load_model(
        vllm_config=vllm_config, model_config=model_config, prefix=prefix
    )
"""

GET_MODEL_REPLACEMENT = f"""
{MARKER}
def _instanttensor_draft_load_config(
    vllm_config: VllmConfig,
    model_config: ModelConfig,
    load_config: LoadConfig | None,
) -> LoadConfig:
    \"\"\"Resolve the loader for one model without changing the target loader.\"\"\"
    import os

    from vllm.config import replace

    mode = os.environ.get("INSTANTTENSOR_DRAFT_LOADER", "auto").strip().lower()
    allowed_modes = ("auto", "safetensors", "instanttensor")
    if mode not in allowed_modes:
        raise ValueError(
            "INSTANTTENSOR_DRAFT_LOADER must be one of "
            f"{{', '.join(allowed_modes)}}; got {{mode!r}}"
        )

    effective = load_config or vllm_config.load_config
    load_format = getattr(effective.load_format, "value", effective.load_format)
    if mode == "instanttensor" or str(load_format).lower() != "instanttensor":
        return effective

    speculative_config = getattr(vllm_config, "speculative_config", None)
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    if draft_model_config is None or model_config is not draft_model_config:
        return effective

    if mode == "auto":
        target_model_config = (
            getattr(speculative_config, "target_model_config", None)
            or vllm_config.model_config
        )
        draft_source = (
            getattr(draft_model_config, "model", None),
            getattr(draft_model_config, "revision", None),
        )
        target_source = (
            getattr(target_model_config, "model", None),
            getattr(target_model_config, "revision", None),
        )
        if draft_source != target_source:
            return effective

    logger.info_once(
        "Hybrid draft loading: using lazy safetensors for speculative draft "
        "weights while preserving InstantTensor for the target model "
        "(INSTANTTENSOR_DRAFT_LOADER=%s).",
        mode,
    )
    return replace(
        effective,
        load_format="safetensors",
        safetensors_load_strategy="lazy",
    )


def get_model(
    *,
    vllm_config: VllmConfig,
    model_config: ModelConfig | None = None,
    prefix: str = "",
    load_config: LoadConfig | None = None,
) -> nn.Module:
    if model_config is None:
        model_config = vllm_config.model_config
    resolved_load_config = _instanttensor_draft_load_config(
        vllm_config, model_config, load_config
    )
    loader = get_model_loader(resolved_load_config)
    return loader.load_model(
        vllm_config=vllm_config, model_config=model_config, prefix=prefix
    )
"""


def validate_shape(text: str, *, patched: bool) -> None:
    tree = ast.parse(text)
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {"get_model", "get_model_loader"}
    if patched:
        required.add("_instanttensor_draft_load_config")
    missing = sorted(required - functions)
    if missing:
        raise ValueError("unexpected model-loader module; missing " + ", ".join(missing))


def patched_text(text: str) -> str:
    validate_shape(text, patched=MARKER in text)
    if MARKER in text:
        if text.count(MARKER) != 1:
            raise ValueError("hybrid-loader marker occurs more than once")
        if "resolved_load_config = _instanttensor_draft_load_config(" not in text:
            raise ValueError("mod marker exists but get_model does not use the helper")
        compile(text, "<patched model_loader/__init__.py>", "exec")
        return text
    count = text.count(GET_MODEL_ANCHOR)
    if count != 1:
        raise ValueError(f"expected exactly one supported get_model function; found {count}")
    patched = text.replace(GET_MODEL_ANCHOR, GET_MODEL_REPLACEMENT, 1)
    validate_shape(patched, patched=True)
    compile(patched, "<patched model_loader/__init__.py>", "exec")
    return patched


target = Path(sys.argv[1])
if not target.is_file():
    raise SystemExit(f"[instanttensor mod] target not found: {target}")
original = target.read_text()
patched = patched_text(original)
if patched == original:
    print("[instanttensor mod] model loader already patched / no change")
else:
    tmp = target.with_suffix(target.suffix + ".instanttensor-draft-mod.tmp")
    tmp.write_text(patched)
    tmp.replace(target)
    print(f"[instanttensor mod] patched {target}")
PY
RUN find /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Point vLLM's bundled NCCL at the sm_121 lib we built.
RUN rm -f /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 && \
    ln -s /usr/lib/aarch64-linux-gnu/libnccl.so.2 /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2

ENTRYPOINT ["vllm", "serve"]

# ======================================================================================================
# DEP PIN DELTAS vs recipes/deepseek-v4-flash-0731/Dockerfile (Marlin reference):
#   torch          2.12.0            (0731: 2.13.0)          — B12X requires >= 2.12
#   torchvision    0.27.0 pinned     (0731: unpinned)
#   torchaudio     ABSENT            (0731: installed)       — B12X preset sets torchaudio=none
#   triton         unpinned (cu130)  (0731: 3.7.1)
#   apache-tvm-ffi ==0.1.12          (0731: "<0.2")
#   FlashInfer     main @ 53a1c3bd   (0731: v0.6.15.post1)   — pinned to the 2026-08-13 tip
#   NCCL           source HEAD only  (0731: source + apt re-pin libnccl 2.30.4)
#   vLLM           local-inference-lab/vllm @ dev/gilded-gnosis
#                                    (0731: jasl/vllm @ sm120-pr-41834-stable-preview-20260802)
#   DeepGEMM       a6b593d2… (SHA)   (0731: nv_dev branch)
#   + B12X pkg     lukealonso/b12x @ master  (--no-deps)     — 0731 has none
#   NOT installed here (0731 adds them for the Marlin/DeepGEMM path): tilelang==0.1.12,
#     nvidia-cutlass-dsl>=4.5.2. B12X relies on the vLLM fork's own pins for cutlass-dsl.
#
# UNVERIFIED / verify-before-build:
#   * vLLM / B12X / FlashInfer are now PINNED to their 2026-08-13 branch tips (SHAs above) — a stable A/B
#     point. Re-pin (git ls-remote the branches) when you want newer fork fixes.
#   * Whether the other upstream docker/patch_vllm_*.py scripts (spark_kv_cache_cleanup,
#     topk_softplus_sqrt, flashinfer_b12x_swigluoai) are still required on this fork.
#   * Whether B12X needs tilelang / an explicit nvidia-cutlass-dsl pin at runtime.
# ======================================================================================================
