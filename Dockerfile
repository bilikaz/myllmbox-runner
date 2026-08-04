# mbx base box — the normal, latest-vLLM container that EVERY recipe extends (recipes do `FROM mbx-base`).
# The official vLLM image is latest vLLM + all deps + a `vllm serve` entrypoint, and runs on GB10
# (aarch64 / SM121 — proven with Qwen3-0.6B). Put anything ALL recipes should share here (common pip
# deps, tools, patches); each recipe's own Dockerfile then adds only its model-specific bits on top.
#
# Built once by run.sh as  mbx-base:latest  (recipes FROM it), so the heavy layer is shared, not rebuilt.
FROM vllm/vllm-openai:latest
