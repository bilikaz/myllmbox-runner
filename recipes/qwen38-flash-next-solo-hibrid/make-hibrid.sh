#!/usr/bin/env bash
# Build the hibrid checkpoint (CPU-only, ~5-10 min, ~6G of new shard; originals are hardlinked).
# Safe to run WHILE a serve is up: no GPU, modest RAM (largest tensor ~1.8G).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p models/myllmbox
docker run --rm --entrypoint python3 \
  -v "$PWD/models:/models" \
  -v "$PWD/recipes/qwen38-flash-next-solo-hibrid/docker/make-hibrid.py:/make-hibrid.py:ro" \
  mbx-qwen38-flash-next-solo-vllm /make-hibrid.py
