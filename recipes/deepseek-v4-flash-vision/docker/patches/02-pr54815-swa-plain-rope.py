"""vLLM PR #54815 (merged 2026-09-02) — DeepSeek-V4 sliding-window layers use PLAIN RoPE.

Reference inference/model.py applies YaRN only to the compressor (CSA/HCA) layers; the sparse SWA
layers use rope_theta with no scaling. The image's build_deepseek_v4_rope applied YaRN to every
layer type whose rope_type != "default". Also handles the newer nested {"main","compress"} rope
dict layout. Verbatim port of the merged diff.
"""
import pathlib, vllm

p = pathlib.Path(vllm.__file__).parent / "models/deepseek_v4/common/rope.py"
s = p.read_text()
assert "Sliding-window layers use plain RoPE" not in s, "already patched"
old = '''    rope_parameters = config.rope_parameters
    rope_parameters["rope_theta"] = (
        config.compress_rope_theta if compress_ratio > 1 else config.rope_theta
    )
    if rope_parameters["rope_type"] != "default":
        rope_parameters["rope_type"] = (
            "deepseek_yarn"
            if rope_parameters.get("apply_yarn_scaling", True)
            else "deepseek_llama_scaling"
        )
    rope_parameters["mscale"] = 0  # Disable mscale
'''
new = '''    rope_parameters = config.rope_parameters
    # Newer checkpoints nest per-layer-type rope dicts ({"main", "compress"});
    # older ones ship a single flat dict shared by all layer types.
    if isinstance(rope_parameters.get("main"), dict) and isinstance(
        rope_parameters.get("compress"), dict
    ):
        key = "compress" if compress_ratio > 1 else "main"
        rope_parameters = dict(rope_parameters[key])
    else:
        rope_parameters = dict(rope_parameters)

    rope_parameters["rope_theta"] = (
        config.compress_rope_theta if compress_ratio > 1 else config.rope_theta
    )
    if compress_ratio > 1 and rope_parameters["rope_type"] != "default":
        # YaRN applies only to compressor (CSA/HCA) layers.
        rope_parameters["rope_type"] = (
            "deepseek_yarn"
            if rope_parameters.get("apply_yarn_scaling", True)
            else "deepseek_llama_scaling"
        )
    else:
        # Sliding-window layers use plain RoPE (theta=rope_theta, no YaRN).
        rope_parameters["rope_type"] = "deepseek_yarn"
        rope_parameters["factor"] = 1.0
        rope_parameters["original_max_position_embeddings"] = max_position_embeddings
    rope_parameters["mscale"] = 0  # Disable mscale
'''
assert s.count(old) == 1, "rope.py anchor not found/unique — upstream changed"
p.write_text(s.replace(old, new, 1))
print("rope.py: plain RoPE on sparse SWA layers, YaRN only on compressor layers (PR #54815)")
