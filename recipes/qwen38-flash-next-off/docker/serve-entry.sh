#!/usr/bin/env bash
# First-boot self-conversion (hibrid46-off — the full-fat bf16-table reference): build if missing
# (fast-skips when complete), then serve. The runner passes [model, --port, ...] after the image when
# the entrypoint isn't "vllm", so the `serve` subcommand is added here.
set -euo pipefail
python3 /opt/mbx/make-hibrid46.py
exec vllm serve "$@"
