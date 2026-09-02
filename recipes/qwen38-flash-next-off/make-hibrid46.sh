#!/usr/bin/env bash
# Build the hibrid46 checkpoint (CPU-only, ~5-10 min, ~1.1G of new shard; originals are hardlinked).
# Safe to run WHILE a serve is up: no GPU, modest RAM. Uses the recipe's OWN image (converter baked at
# /opt/mbx/make-hibrid46.py) — run.sh builds it before the model pipeline, so this works on a fresh box.
set -euo pipefail
cd "$(dirname "$0")/../.."
IMG=mbx-qwen38-flash-next
docker image inspect "$IMG" >/dev/null 2>&1 \
  || { echo "image $IMG missing — build it first: ./build-and-copy.sh qwen38-flash-next (or just ./run.sh)"; exit 1; }
mkdir -p models/myllmbox
docker run --rm --entrypoint python3 -v "$PWD/models:/models" "$IMG" /opt/mbx/make-hibrid46.py
