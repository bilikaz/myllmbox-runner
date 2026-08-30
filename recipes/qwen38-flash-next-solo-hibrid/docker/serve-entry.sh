#!/usr/bin/env bash
# First-boot self-conversion: if the hibrid checkpoint isn't at /models/myllmbox yet, build it from
# the Inferact source (CPU-only, ~5-10 min, once — make-hibrid.py fast-skips when complete), then
# serve. The runner passes [model, --port, ...] after the image when the entrypoint isn't "vllm",
# so the `serve` subcommand is added here.
set -euo pipefail
python3 /opt/mbx/make-hibrid.py
exec vllm serve "$@"
