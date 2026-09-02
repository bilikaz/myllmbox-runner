#!/usr/bin/env bash
# First-boot self-conversion (hibrid46 — shared expert back to bf16): build the checkpoint if missing
# (fast-skips when complete), then serve. The runner passes [model, --port, ...] after the image when
# the entrypoint isn't "vllm", so the `serve` subcommand is added here.
set -euo pipefail
python3 /opt/mbx/make-hibrid46.py
exec vllm serve "$@"
