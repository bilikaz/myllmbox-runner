"""torchao backend — NVFP4 (W4A4, Triton sm_121a, ~4x on GB10) and MXFP8 (W8A8).

Quantizes on GPU (Triton FP4 kernels JIT for the running arch) then saves a self-contained checkpoint.
safe_serialization=False everywhere: torchao tensor subclasses are pickled (.bin) — safetensors can't
hold them (the serve side loads with use_safetensors=False for the same reason).
"""
from __future__ import annotations

import os

import torch


def _ao_config(fmt: str):
    if fmt == "nvfp4":
        from torchao.prototype.mx_formats.inference_workflow import (
            NVFP4DynamicActivationNVFP4WeightConfig,
        )
        return NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True, use_triton_kernel=True)
    if fmt == "mxfp8":
        from torchao.prototype.mx_formats.constants import KernelPreference
        from torchao.prototype.mx_formats.inference_workflow import (
            MXDynamicActivationMXWeightConfig,
        )
        return MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float8_e4m3fn, weight_dtype=torch.float8_e4m3fn,
            kernel_preference=KernelPreference.AUTO)
    raise SystemExit(f"[torchao] format {fmt} is not nvfp4|mxfp8")


def quantize(model: str, out: str, fmt: str, model_type: str, target: str, skip: list[str]) -> None:
    ao = _ao_config(fmt)
    os.makedirs(out, exist_ok=True)

    if model_type in ("diffusion", "video"):
        # diffusion + most video models are diffusers pipelines: quantize the heavy denoiser (`target`,
        # usually "transformer"), keep VAE/text-encoder in BF16. Saves the FULL pipeline → self-contained.
        from diffusers import DiffusionPipeline, PipelineQuantizationConfig, TorchAoConfig
        tao = _torchao_cfg(TorchAoConfig, ao, skip)
        pqc = PipelineQuantizationConfig(quant_mapping={target: tao})
        pipe = DiffusionPipeline.from_pretrained(
            model, torch_dtype=torch.bfloat16, quantization_config=pqc).to("cuda")
        pipe.save_pretrained(out, safe_serialization=False)
        return

    if model_type == "llm":
        from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
        tao = _torchao_cfg(TorchAoConfig, ao, skip)
        m = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.bfloat16, quantization_config=tao, device_map="cuda")
        m.save_pretrained(out, safe_serialization=False)
        AutoTokenizer.from_pretrained(model).save_pretrained(out)  # ship the tokenizer with the quant
        return

    raise SystemExit(
        f"[torchao] model_type '{model_type}' has no loader yet (have: diffusion, video, llm). "
        "Add its from_pretrained/save_pretrained pair here — the torchao config is the same.")


def _torchao_cfg(TorchAoConfig, ao, skip):
    """diffusers' and transformers' TorchAoConfig both accept modules_to_not_convert on new versions."""
    try:
        return TorchAoConfig(ao, modules_to_not_convert=skip) if skip else TorchAoConfig(ao)
    except TypeError:
        print("[torchao] modules_to_not_convert unsupported on this version → full quant", flush=True)
        return TorchAoConfig(ao)
