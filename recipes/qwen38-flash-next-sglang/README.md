# Qwen3.8-Flash-Next on 2× Spark via SGLang — experiment log (2026-08-27)

Image: `mbx-qwen38-flash-next-sglang` = stock lmsysorg qwen38flashnext + MiaAI Triton QSA patch (SM121)
+ our PLE-from-disk mmap patch (`MBX_PLE_MMAP=1`). All boots TP=2, 262144 ctx, pytorch sampler, radix off.

## Boot/crash matrix (all on the MiaAI-patched image)

| # | checkpoint | mamba pool | spec | result |
|---|---|---|---|---|
| 1 | FP8 | auto=6 | NEXTN 3/1/4 | serves; `!` flood ~50 gen-tokens (accept 0.00) |
| 2 | FP8 | auto=6 | off | clean ≤600 tok; **device-assert CRASH** ~1300-1500 tok (long thinking) — bf16 AND fp32 SSM state both |
| 3 | FP8 | **97 + extra_buffer + interval 64 + ReplaySSM** | NEXTN | **no crash** (survived killer prompt); bangs remain — `accept rate 0.00` flat = fp8-quantized MTP head produces garbage drafts; 11.8 tok/s |
| 4 | Inferact NVFP4 (171G layout: nvfp4 experts, **bf16 PLE**, nvfp4 MTP experts) | 97 … | NEXTN | draft loader CRASH: `fused_moe_triton._load_w13` 2560 vs packed 1280 — sglang can't load nvfp4 MTP experts; `--speculative-draft-model-quantization modelopt_fp4` does NOT help |
| 5 | Inferact NVFP4 | 97 … | off | **STABLE + CLEAN** (killer prompt passed, quality = FP8 class); 11-12 tok/s |

## Root causes nailed today

- **Mamba pool starvation** (tonyd2wild INCIDENT-2026-08-27, reproduced): radix-off auto-sizes the SSM
  state pool to max-running-requests (6). Tracked checkpoints (1 per mamba-track-interval tokens) +
  ~4 spec states/request exhaust it → state corruption. Slow arrival = inf/nan device assert (~1300 tok
  of thinking); fast arrival (spec) = token-0 `!` flood. Fix: `max-mamba-cache-size: 97` +
  `mamba-scheduler-strategy: extra_buffer` + `mamba-track-interval: 64`.
- **fp8 MTP head is useless for spec** (accept 0.00 flat); Tony's 0.36-0.45 accept is on RadixArk's
  BF16 MTP head. sglang can't load Inferact's nvfp4 MTP experts at all. → spec on sglang requires RadixArk.
- **Disk-PLE (mmap) is cheap**: ~0.5 major faults/token warm; identical tok/s to previous pinned/in-pool
  boots. Bought a 2.05M-token KV pool on FP8 (~24G/node) and runs the 102G bf16 "real ngram" for free on
  Inferact (~35G/node in-RAM weights).

## Speed ledger (single stream, decode)

| config | tok/s |
|---|---|
| ours FP8 or Inferact NVFP4, spec off/dead | **11-12** (quant-independent → per-token overhead-bound) |
| tonyd2wild NVFP4 spec-off | ~20 |
| tonyd2wild NVFP4, MTP accept 0.36-0.45 | 44-50 cold / ~70 warm |

Gap suspects vs Tony (same hardware): (1) his real FlashInfer TRT-LLM QSA kernel (guard patch) vs our
Triton QSA fallback on every sparse layer; (2) his host NCCL 2.30.4 (LD_PRELOAD) + NCCL_MAX/MIN_NCHANNELS=4.
Spec tier additionally needs RadixArk (bf16 MTP head).

## Open items
- [ ] Port Tony's QSA guard patch (or find it in his DEPLOY-REPORT.md) → expected biggest single speedup
- [ ] NCCL channel tuning A/B (env-only)
- [ ] RadixArk download resume → NEXTN with a head that accepts (2.5-3× target)
- [ ] vLLM lane: image pulled, `vllm/v1/ple_offload/` located for the mmap port; Inferact is its official checkpoint
- [ ] 6-way concurrent load test as acceptance gate (Tony's rule; single-request smokes prove nothing)
