"""quantizer — the unified, recipe-driven model quantizer (parallel to `runner`).

`runner` serves models; `quantizer` produces the quantized checkpoints they serve. It runs INSIDE the
shared `mbx-quantizer` image (built from quantizer/Dockerfile), driven entirely by the recipe's
`quantize:` block (passed in as QUANT_* env by ./quantize.sh). One generic tool, many models: the yaml
says what to take, how to load it (diffusion|llm), which layers to keep in BF16, and the target format.

Output is a self-contained checkpoint at models/myllmbox/<name>-<format> — the serve recipe loads it
instantly (no on-the-fly quant), and it's a shareable artifact (publish → others ./download.sh + serve).

Backends are pluggable by format: `torchao` (nvfp4/mxfp8, diffusion+LLM) ships here; gguf (llama.cpp /
ComfyUI-gguf) and w4a16 (llm-compressor) are additional backend images selected by --to as they land.
"""
