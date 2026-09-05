"""vLLM PR #52724 (merged 2026-09-01) — adaptive verification on FLASHINFER_MLA_SPARSE_DSV4.

The FlashInfer sparse-MLA decode kernel takes variable query lengths (cum_seq_lens_q / max_q_len),
but the backend's metadata builders declared AttentionCGSupport.UNIFORM_BATCH, so DSpark's
enable_adaptive_verification was refused at runtime. Thin builder subclasses declare ALWAYS and are
wired into the FlashInfer MLA backend + both attention classes (SM100 and SM120 paths). Verbatim
port of the merged diff.
"""
import pathlib, vllm

p = pathlib.Path(vllm.__file__).parent / "models/deepseek_v4/nvidia/flashinfer_sparse.py"
s = p.read_text()
assert "DeepseekSparseSWAFlashInferBackend" not in s, "already patched"

old = '''from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_dspark_swa_index_width,
)
'''
new = '''from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4
from vllm.v1.attention.backend import AttentionCGSupport, MultipleOf
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_dspark_swa_index_width,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadataBuilder,
)
'''
assert s.count(old) == 1, "flashinfer_sparse.py import anchor not found/unique"
s = s.replace(old, new, 1)

old = '''        return "FLASHINFER_MLA_SPARSE_DSV4 requires SM10x or SM12x"


class DeepseekV4FlashInferMLAAttention(DeepseekV4Attention):'''
new = '''        return "FLASHINFER_MLA_SPARSE_DSV4 requires SM10x or SM12x"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4FlashInferSparseMLAMetadataBuilder"]:
        return DeepseekV4FlashInferSparseMLAMetadataBuilder


class DeepseekV4FlashInferSparseMLAMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    """Varlen-capable metadata builder for the FlashInfer sparse MLA backend."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class DeepseekSparseSWAFlashInferMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    """SWA metadata for the FlashInfer sparse decode path (varlen decode)."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class DeepseekSparseSWAFlashInferBackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type[DeepseekSparseSWAFlashInferMetadataBuilder]:
        return DeepseekSparseSWAFlashInferMetadataBuilder


class DeepseekV4FlashInferMLAAttention(DeepseekV4Attention):'''
assert s.count(old) == 1, "flashinfer_sparse.py backend-class anchor not found/unique"
s = s.replace(old, new, 1)

for flag in ("False", "True"):
    old = f'''    backend_cls = DeepseekV4FlashInferMLASparseBackend
    use_fp8_ds_mla_layout: ClassVar[bool] = {flag}
'''
    new = f'''    backend_cls = DeepseekV4FlashInferMLASparseBackend
    swa_backend_cls = DeepseekSparseSWAFlashInferBackend
    use_fp8_ds_mla_layout: ClassVar[bool] = {flag}
'''
    assert s.count(old) == 1, f"flashinfer_sparse.py attention-class anchor ({flag}) not found/unique"
    s = s.replace(old, new, 1)
p.write_text(s)
print("flashinfer_sparse.py: ALWAYS-cudagraph builders wired for the FlashInfer DSV4 backend (PR #52724)")
