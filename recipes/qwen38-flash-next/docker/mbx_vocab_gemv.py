"""M=1 BF16 lm_head projection via Triton GEMV — vendored from b12x (local-inference-lab/b12x,
Apache-2.0, gemm/bf16_vocab_projection @ master 2026-08-29; NOT in the b12x==1.3.0 pip release).

WHY: at M=1, cuBLAS leaves ~30% of GB10 bandwidth on the floor for the 248320x2560 BF16 head —
b12x measured 6.77 ms -> 4.74 ms on THIS exact shape (their commit: "Qwen 3.8 Flash Next TP1
serving reduced the M=1 projection from 6.77 ms to 4.74 ms"). In our K=3 MTP serve the M=1 head
runs 3x per decode cycle (every draft step), so this is worth ~6 ms of an ~85 ms cycle.

Config: the loop kernel with BLOCK_K=1024 / num_warps=8 — the winner in b12x's embedded GB10
device profile for vocab-projection shapes. One program per vocab row, fp32 accumulate.

Wrapped as a torch.library custom op (mbx namespace) => CUDA-graph- and torch.compile-safe.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_K = 1024
_NUM_WARPS = 8


@triton.jit
def _row_loop_kernel(source, weight, output, K: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((), tl.float32)
    for start in range(0, K, BLOCK_K):
        positions = start + offsets
        mask = positions < K
        values = tl.load(source + positions, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + row * K + positions, mask=mask, other=0.0).to(tl.float32)
        accumulator += tl.sum(values * weights, axis=0)
    tl.store(output + row, accumulator)


@torch.library.custom_op("mbx::bf16_vocab_gemv", mutates_args=())
def _bf16_vocab_gemv(source: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out_features, in_features = weight.shape
    output = torch.empty((1, out_features), dtype=torch.bfloat16, device=source.device)
    _row_loop_kernel[(out_features,)](
        source, weight, output, K=in_features, BLOCK_K=_BLOCK_K, num_warps=_NUM_WARPS
    )
    return output


@_bf16_vocab_gemv.register_fake
def _bf16_vocab_gemv_fake(source: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return source.new_empty((source.shape[0], weight.shape[0]))


def maybe_vocab_gemv(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
    """The GEMV when it applies, None otherwise (caller falls through to the stock path)."""
    if (
        hidden_states.ndim == 2
        and hidden_states.shape[0] == 1
        and weight.ndim == 2
        and hidden_states.shape[1] == weight.shape[1]
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden_states.is_cuda
        and weight.is_cuda
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
    ):
        return _bf16_vocab_gemv(hidden_states, weight)
    return None
