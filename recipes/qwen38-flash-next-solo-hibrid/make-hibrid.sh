#!/usr/bin/env bash
# Build the hibrid checkpoint (CPU-only, ~5-10 min, ~3G of new shard; originals are hardlinked).
# Safe to run WHILE a serve is up: no GPU, modest RAM (largest tensor ~1.8G).
# Uses the recipe's OWN image (which bakes the converter at /opt/mbx/make-hibrid.py) — run.sh builds
# it before the model pipeline runs, so this works on a fresh box. The serve entrypoint also runs the
# converter at first boot (fast-skips when complete), so this script is the manual/pre-build path.
set -euo pipefail
cd "$(dirname "$0")/../.."
IMG=mbx-qwen38-flash-next-solo-hibrid
docker image inspect "$IMG" >/dev/null 2>&1 \
  || { echo "image $IMG missing — build it first: ./build-and-copy.sh qwen38-flash-next-solo-hibrid (or just ./run.sh)"; exit 1; }
mkdir -p models/myllmbox
docker run --rm --entrypoint python3 -v "$PWD/models:/models" "$IMG" /opt/mbx/make-hibrid.py
