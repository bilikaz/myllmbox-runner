# qwen38-flash-next-uncensored-cluster — reports

Checkpoint: `/models/myllmbox/Qwen3.8-Flash-Next-hibrid47-uncensored` (99.0 GiB) = orcarouter's abliterated body
(compressed-tensors: experts NVFP4 W4A16, QSA/GDN/shared-expert/lm_head FP8 per-channel, rest bf16) + hibrid47's NVFP4 PLE
table (identity-verified on sampled rows) + the drafter's 512 experts re-quantized bf16 → per-expert NVFP4 in the body's
format (`builds/qwen38-flash-next/standardize-flash-next.py`, `quantize-drafter-experts.py`). Tokenizer/generation files
from hibrid47 (orcarouter's are a transformers-5.16 re-serialization; the forum reported their tokenizer breaking tool calls).
Image `mbx-qwen38-flash-next-uncensored-cluster` = the cluster lane's build (patches 01–04), own copy per house rule.

## Boots
| # | date | result | notes |
|---|---|---|---|
| 1 | 2026-09-08 ~13:10 | ✗ docker run: image missing | run.sh's build gate ran seconds before the lane's Dockerfile landed (timing, not code) |
| 2 | 2026-09-08 17:12 | ✗ drafter load | `Layer mtp.layers.48.mlp.experts has no parameter w2_weight_global_scale`: orcarouter's `quantization_config.ignore` carries `re:.*mtp\..*` → vLLM builds the whole drafter unquantized (that is why they shipped it bf16). Fix: drop that ignore entry (quantize-drafter-experts.py now does it). Body + table had loaded fine |
| 3 | 2026-09-08 17:43 | ✓ HEALTHY | load 577 s (worker 403 s), **64.95 GiB/rank** (hibrid47: 64.9), Marlin NVFP4 MoE, table resident replicated, torch.compile 31+3 s, **KV 2,846,834 tokens fp8** (= hibrid47 on the same 28 G pin), graph capture 36 s / 2.86 GiB, K=4 |

## Measured (bench/test.py, pasture, thinking off, K=4, kv 28 G fp8, 32 seats)
| date | c | gen tok/s | per-stream | steps/s | acc len | ttft s | nvme rd/s | err/s maj;min | steady/watched | vs hibrid47 cluster (same rung) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-08 | 1 | 75 (71–80) | 75 | **17.9 (17.8–18.0)** | 4.21 (3.96–4.52) | 0.58 | 1 | 1;13146 | 70 / 100 | steps 17.7, 73–76 avg, 80 peak, acc 4.1–4.2 → **+1 % steps**, same tokens |
| 2026-09-08 21:04 | 1 | 75 (74–76) | 75 | 17.9 (17.7–18.1) | 4.19 (4.11–4.31) | 0.57 | 1 | 1;13910 | 50 / 80 | acc by pos 0.91/0.84/0.77/0.67 |
| 2026-09-08 21:05 | 1 | 73 (69–77) | 73 | 17.9 (17.7–18.0) | 4.07 (3.82–4.37) | 0.57 | 0 | 0;13218 | 70 / 100 | acc by pos 0.89/0.81/0.74/0.64 |

Reading (3 runs, steps 17.9 every time, ttft 0.57–0.58 s): the abliterated body runs at hibrid47 speed plus a hair — its QSA/GDN/shared-expert linears are FP8 (W8A16 Marlin)
where hibrid47's are bf16, so a step moves slightly fewer bytes. Acceptance 4.21 with the re-quantized drafter = hibrid47's
4.1–4.3 (spec decoding is exact; the drafter only sets speed). First TTFT number on record: 0.58 s for the ~1.2k-token
pasture prompt at c=1. Quality: refusal removal not yet probed; tool calls + boss-animals gauntlet pending.
