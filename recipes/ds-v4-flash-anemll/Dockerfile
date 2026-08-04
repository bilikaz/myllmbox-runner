# ds-v4-flash-anemll — image provenance. This recipe runs anemll's PREBUILT dspark vLLM (built by anemll,
# not by us — we don't have their source). This Dockerfile documents the base and pins the serve entrypoint
# so it goes through the same build-and-copy.sh flow as our own recipes.
#
# What anemll built (from `docker inspect` + the running engine): vLLM 0.25.2.dev (g752a3a50) + FlashInfer
# 0.6.15 + torch 2.11.0+cu130, CUDA 13.0, arch 12.0+PTX, with their custom b12x MXFP4 MoE + dspark
# speculation. This is the reference that proved dspark serves DS-V4-Flash-0731 on our cluster.
#
#   ./build-and-copy.sh ds-v4-flash-anemll   →   mbx-ds-v4-flash-anemll
FROM ghcr.io/anemll/dspark-vllm-gx10:0.1.1
ENTRYPOINT ["vllm", "serve"]
