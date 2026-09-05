# syntax=docker/dockerfile:1.6
# ------------------------------------------------------------------------------------------------------
# myllmbox/qwen38-flash-next-cluster-vllm:v1 — the PUBLISHED qwen38-flash-next image + ONE patch: multi-node PLE
# offload. Public build of recipes/qwen38-flash-next-cluster/Dockerfile (byte-identical patch); ledger: cluster-v1.md.
# Build:  docker build -t myllmbox/qwen38-flash-next-cluster-vllm:v1 builds/qwen38-flash-next/cluster/
#
# BASE = docker.io/myllmbox/qwen38-flash-next-vllm:v1 (our own release, 2026-09-02; reproducible from
# builds/qwen38-flash-next/ — vendor SM121 vLLM base + the ple_layer patch set that loads the hibrid46
# checkpoint's in-shard int3 PLE table into the CPU-offload worker). Pinned by digest.
#
# On top: docker/patches/01-ple-offload-multinode.py — upstream fences the PLE offload worker to nnodes=1;
# the patch lifts the four gates under env MBX_PLE_MULTINODE=1 (one full-table worker PER NODE, each rank
# feeds its own). Ported verbatim from experiments/qwen38-flash-next-cluster-vllm (first healthy TP=2 boot
# 2026-08-29). Without the env the image behaves exactly like v1.
#
# Consumed by solo/qwen38-flash-next-cluster-recipe (the 2-Spark kit) and, as mbx-qwen38-flash-next-cluster, by the recipe.
# ------------------------------------------------------------------------------------------------------
FROM myllmbox/qwen38-flash-next-vllm:v1@sha256:92ccd7de6d80b9597844d7a5bffb9ded4067dd323129e18bc24192c00dac09ad
LABEL org.opencontainers.image.source="https://github.com/bilikaz/myllmbox" \
      mbx.base="myllmbox/qwen38-flash-next-vllm:v1@sha256:92ccd7de" \
      mbx.patches="ple-offload-multinode loader-drop-page-cache"

COPY docker/patches/ /opt/mbx/patches/
RUN set -e; for p in /opt/mbx/patches/0*.py; do echo "== $p"; python3 "$p"; done

# Sanity: every gate is in place and the patched modules still import.
RUN python3 - <<'PY'
import pathlib, vllm
root = pathlib.Path(vllm.__file__).parent
# literal-env count per file: gpu_worker has 2 (site 3 reuses the _mbx_mn flag), worker 1, connector 2
for f, n in {"v1/worker/gpu_worker.py": 2, "v1/ple_offload/worker.py": 1, "v1/ple_offload/connector.py": 2}.items():
    got = (root / f).read_text().count("MBX_PLE_MULTINODE")
    assert got >= n, (f, got, n)
assert "_mbx_fadvise_dontneed" in (root / "model_executor/model_loader/weight_utils.py").read_text()
import vllm.v1.ple_offload.worker, vllm.v1.ple_offload.connector, vllm.model_executor.model_loader.weight_utils
print("mbx-qwen38-flash-next-cluster: multi-node PLE offload gates present; vllm", vllm.__version__)
PY
