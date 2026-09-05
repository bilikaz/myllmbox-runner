#!/usr/bin/env python3
"""MBX: GPU-resident NVFP4 PLE table — no CPU offload worker, no IPC, table row-sharded across TP ranks.

Activates when config.json declares `ple_quantization.format == "nvfp4"` (written by docker/make-ple-nvfp4.py) AND
VLLM_PLE_CPU_OFFLOAD is off. Then:
  1. `_get_ple_embedding_quant_method` returns `_MbxNvfp4EmbeddingMethod`, whose create_weights allocates the
     PACKED table for this rank's vocab partition (uint8 [rows/tp, 80] + fp8 scales [rows/tp, 10] + fp32 global)
     instead of a bf16 [rows/tp, 160] weight (that would be 51 GiB/2 — the reason the table ever left the GPU).
  2. `load_weights` routes `ngram_embedding.nvfp4_shard_k.{packed,scales}` by row range into the partition
     (same overlap logic the bf16 shards use) and `nvfp4_global` into its scalar.
  3. `embedding()` = the gather: index_select packed rows → nibble unpack → e2m1 LUT → × fp8 block scale ×
     global → bf16. VocabParallelEmbedding.forward then masks non-local rows and all-reduces across TP —
     the standard vocab-parallel path, so both boxes end with the full 16-row lookup.
On a Spark "GPU memory" is the same LPDDR5X as the worker's RAM: this moves nothing physically, it removes the
ZeroMQ → CPU gather → pinned buffer → DMA → semaphore detour per step, and the per-node offload worker (so the
multi-node offload patch is no longer needed). int3 checkpoints are untouched (no `format` key → stock path).
`--check` = assert anchors only.
"""
import sys

CHECK = "--check" in sys.argv
P = "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
s = open(P).read()
assert "_MbxNvfp4EmbeddingMethod" not in s, "already patched"

edits = []

# A. accept the nvfp4 declaration in the config reader
edits.append(("std-cfg keys",
    """        if isinstance(c, dict) and {"bits", "group", "rows", "dim"} <= set(c):""",
    """        if isinstance(c, dict) and ({"bits", "group", "rows", "dim"} <= set(c)
                                    or {"format", "block", "rows", "dim"} <= set(c)):   # MBX: nvfp4 table"""))

# B. quant-method selection: nvfp4-on-GPU wins when declared and offload is off
edits.append(("quant-method gate",
    '''    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    if not isinstance(quant_config, Fp8Config):''',
    '''    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    if _mbx_nvfp4_gpu():                 # MBX: NVFP4 table resident on the GPU, row-sharded over TP
        return _MbxNvfp4EmbeddingMethod()
    if not isinstance(quant_config, Fp8Config):'''))

# C. loader branch for the nvfp4 shard tensors (before the int3 qbits branch)
edits.append(("load branch",
    """            if name.startswith("ngram_embedding.qbits_"):""",
    """            if name.startswith("ngram_embedding.nvfp4_"):          # MBX: GPU-resident NVFP4 table
                loaded.update(_mbx_nvfp4_load(self, name, loaded_weight))
                continue
            if name.startswith("ngram_embedding.qbits_"):"""))

for label, a, b in edits:
    assert s.count(a) == 1, f"anchor '{label}' not found/unique — upstream changed"
    s = s.replace(a, b, 1)

# D. the method + helpers (module level, appended)
s += '''

# ---- MBX: GPU-resident NVFP4 PLE table ------------------------------------------------------------------
def _mbx_nvfp4_gpu() -> bool:
    """True when the checkpoint declares an NVFP4 table and the table is to live on the GPU (offload off)."""
    cfg = _mbx_ple_std_cfg()
    return bool(cfg) and str(cfg.get("format", "")).lower() == "nvfp4" and not envs.VLLM_PLE_CPU_OFFLOAD


class _MbxNvfp4EmbeddingMethod(QuantizeMethodBase):
    """Vocab-parallel embedding whose rows are NVFP4: uint8 packed e2m1 pairs + fp8 block scales + fp32 global."""

    BLOCK = 16
    _LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        n = int(sum(output_partition_sizes)); D = int(input_size_per_partition)
        assert D % self.BLOCK == 0 and D % 2 == 0, f"PLE head_dim {D} not NVFP4-blockable"
        layer.register_parameter("nvfp4_packed", nn.Parameter(torch.empty((n, D // 2), dtype=torch.uint8), requires_grad=False))
        layer.register_parameter("nvfp4_scales", nn.Parameter(torch.empty((n, D // self.BLOCK), dtype=torch.float8_e4m3fn), requires_grad=False))
        layer.register_parameter("nvfp4_global", nn.Parameter(torch.ones((), dtype=torch.float32), requires_grad=False))
        layer.register_buffer("nvfp4_lut", torch.tensor(self._LUT, dtype=torch.float32), persistent=False)
        layer._mbx_nvfp4_rows_loaded = 0
        __import__("logging").getLogger("mbx.ple").info(
            "MBX NVFP4 PLE: GPU-resident table partition %d rows x %d (%.2f GiB packed + %.2f GiB scales)",
            n, D, n * (D // 2) / 2**30, n * (D // self.BLOCK) / 2**30)

    def process_weights_after_loading(self, layer) -> None:
        exp = int(layer.nvfp4_packed.shape[0])
        got = int(getattr(layer, "_mbx_nvfp4_rows_loaded", 0))
        if got != exp:
            raise RuntimeError(f"MBX NVFP4 PLE: partition has {exp} rows but {got} were loaded from the checkpoint")
        print(f"PLE: NVFP4 table resident on GPU ({exp} rows this rank, global scale {float(layer.nvfp4_global):.4e})", flush=True)

    def apply(self, layer, x, bias=None):
        raise NotImplementedError("NVFP4 PLE table is gather-only")

    def embedding(self, layer, input_: torch.Tensor) -> torch.Tensor:
        pk = layer.nvfp4_packed.index_select(0, input_)                                  # [n, D/2] u8
        n = pk.shape[0]
        codes = torch.stack(((pk & 0x0F), (pk >> 4)), dim=-1).view(n, -1).long()        # [n, D]  even value = low nibble
        x = layer.nvfp4_lut[codes]                                                        # f32 [n, D]
        sc = layer.nvfp4_scales.index_select(0, input_).to(torch.float32) * layer.nvfp4_global   # [n, D/16]
        x = x.view(n, -1, self.BLOCK) * sc.unsqueeze(-1)
        return x.view(n, -1).to(layer.params_dtype)


def _mbx_nvfp4_load(self, name: str, loaded_weight: torch.Tensor) -> set[str]:
    """Route one `ngram_embedding.nvfp4_*` checkpoint tensor into this rank's partition. Returns param names touched."""
    emb = self.ngram_embedding
    if name.endswith("nvfp4_global"):
        emb.nvfp4_global.data.copy_(loaded_weight.to(torch.float32).reshape(()))
        return {"ngram_embedding.nvfp4_global"}
    import re as _re
    m = _re.match(r"ngram_embedding\\.nvfp4_shard_(\\d+)\\.(packed|scales)$", name)
    if not m:
        raise ValueError(f"unexpected NVFP4 PLE tensor {name}")
    k, kind = int(m.group(1)), m.group(2)
    cfg = _mbx_ple_std_cfg()
    S = int(cfg["shard_rows"])
    start = k * S
    n_rows = int(loaded_weight.shape[0])
    p0 = int(emb.shard_indices.padded_org_vocab_start_index)
    p1 = int(emb.shard_indices.padded_org_vocab_end_index)
    lo, hi = max(start, p0), min(start + n_rows, p1)
    param = emb.nvfp4_packed if kind == "packed" else emb.nvfp4_scales
    if hi > lo:
        param.data[lo - p0: hi - p0].copy_(loaded_weight[lo - start: hi - start].to(param.device, non_blocking=True))
        if kind == "packed":
            emb._mbx_nvfp4_rows_loaded = getattr(emb, "_mbx_nvfp4_rows_loaded", 0) + (hi - lo)
    return {f"ngram_embedding.nvfp4_{kind}"}
'''

if not CHECK:
    open(P, "w").write(s)
print("GPU-resident NVFP4 PLE patch:", "anchors OK" if CHECK else "applied (activates on ple_quantization.format=nvfp4 with VLLM_PLE_CPU_OFFLOAD off)")
