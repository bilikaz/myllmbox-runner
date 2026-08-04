#!/usr/bin/env bash
# download.sh <hf-id>  —  fetch a model's weights into ./models/<id> via hf, so a recipe can serve it
# offline as  model: /models/<id>. Self-contained: bootstraps its own venv + huggingface_hub.
#
#   ./download.sh Qwen/Qwen3-0.6B
#   ./download.sh deepseek-ai/DeepSeek-V4-Flash-0731
#
set -euo pipefail
cd "$(dirname "$0")"

M="${1:-}"
[ -n "$M" ] || { echo "usage: ./download.sh <hf-id>   e.g. ./download.sh Qwen/Qwen3-0.6B"; exit 1; }

# venv — bootstraps hf here, not on you
V=.venv
[ -x "$V/bin/python" ] || python3 -m venv "$V"
"$V/bin/python" -m pip install -q -U pip "huggingface_hub[cli]" >/dev/null

echo "· downloading $M  →  ./models/$M"
"$V/bin/hf" download --local-dir "models/$M" "$M"
echo "✓ ./models/$M   —   serve it in a recipe as   model: /models/$M"
