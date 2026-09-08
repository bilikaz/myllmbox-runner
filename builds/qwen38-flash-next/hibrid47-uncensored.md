# hf.co/myllmbox/Qwen3.8-Flash-Next-hibrid47-uncensored — the abliterated body on the hibrid47 layout

**Built:** 2026-09-08 on ai1 (`builds/qwen38-flash-next/standardize-flash-next.py` + `quantize-drafter-experts.py`). **Size:** 99.0 GiB,
25 shard files (16 body + model-mtp + 8 ple-nvfp4). **Local:** `models/myllmbox/Qwen3.8-Flash-Next-hibrid47-uncensored` on both boxes.
**Published:** 2026-09-08 21:30 — https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid47-uncensored, public + **gated (auto)**,
commit `d4ceff2e`, 39 files, 106.3 GB. Uploaded from ai1 (v4 image's hf 1.28 + xet): only **2.16 + 0.99 GB actually transferred** —
xet deduplicated the 16 body shards against orcarouter's repo and the 8 table shards against hibrid47's. First commit attempt failed
with "Private repository storage limit reached" (free plan caps PRIVATE storage; the plan was create-private → upload → flip public):
for gated releases create the repo public+gated from the start, gating is what protects access.

**Source:** orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4 (gated, 189 GB; downloaded with the myllmbox HF account): abliterated
body in compressed-tensors (experts NVFP4 W4A16 group-16 fp8 scales; QSA/GDN/shared-expert/lm_head FP8 per-channel; rest bf16),
bf16 PLE table in shard 2 (102 GB), bf16 fused drafter. Abliteration = Arditi et al., one direction from layer 24, 149 residual
writers incl. o_proj, GDN out_proj, experts down_proj (49 layers incl. MTP), shared down_proj, embed_tokens + ple.value_proj.

## What the standardizer did
- Table: sampled rows from 6 bf16 shards re-quantized with our quantizer == hibrid47's NVFP4 codes+scales byte for byte →
  hibrid47's 8 shards hardlinked, shard 2 dropped. (Earlier probe: 4 ranges from 4 shards, same result.)
- Body: 16 shards + mtp hardlinked as-is. Config = orcarouter's + `ple_quantization`, `layer_types` `qwen_sparse_attention` →
  `full_attention` (vendor image spelling); tokenizer/generation/preprocessor files from hibrid47 (theirs = transformers 5.16
  re-serialization with a different pre-tokenizer regex/decoder flags and 5.16-only tokenizer_config keys; forum post 153
  reported the original uncensored release breaking tool calls via its tokenizer).
- Drafter: 512 experts bf16 fused [E,2I,H]/[E,H,I] → per-expert `weight_packed/weight_scale(fp8)/weight_global_scale` with
  compressed-tensors 0.17's generate_gparam/calculate_qparams/quantize/pack_fp4_to_uint8; gate/up order verified vs hibrid47's
  drafter (cos 0.995/0.995 vs 0.002 crosswise; down_proj 0.995 = abliteration delta visible); rel err 0.094–0.095; 4.86 → 1.49 GiB.
  First pass stored scales as fp32 (library returns fp32) → fixed to fp8. `quantization_config.ignore` `re:.*mtp\..*` removed
  (it made vLLM build the drafter unquantized → "no parameter w2_weight_global_scale"; the reason orcarouter shipped bf16).

## Size table (standardizer output)
routed experts 63.28 GiB · PLE table 26.82 · drafter 1.49 · embeddings/lm_head 2.37 · GDN 1.95 · other 1.37 · vision 0.84 ·
QSA 0.59 · shared 0.22 → **99.0 GiB** (hibrid47 98.9: drafter 1.49, QSA 1.15 bf16, GDN 1.11, shared 0.44).

## Measured (recipes/qwen38-flash-next-uncensored-cluster, boot 3, 2026-09-08)
64.95 GiB/rank, KV 2,846,834 fp8 @ 28 G, graph capture 2.86 GiB. c=1 pasture thinking off ×3: steps 17.9 (17.7–18.1),
73–75 tok/s avg, 80 peak, acc 4.07–4.21, ttft 0.57–0.58 s. hibrid47 same rung: 17.7 / 73–76 / 80. Quality (refusal rate,
tool calls, boss-animals gauntlet): pending.

## License
Base = Qwen Community License 1.0 (`license: other`); orcarouter labels its derivatives Apache-2.0 "inherited from the base" —
the base is not Apache, so our repo carries the Qwen license like hibrid47/Inferact, credits OrcaRouter (their edit offered
under Apache-2.0), and is gated with a research/responsibility agreement. Card: `hibrid47-uncensored-README.md`.

## Publish procedure (when approved)
1. `cp hibrid47-uncensored-README.md <ckpt>/README.md`; `cp <hibrid47>/LICENSE <ckpt>/LICENSE` (Qwen license text).
2. `HfApi().create_repo(…, private=False)` → `update_repo_settings(gated="auto")` (NOT private: the free plan's private-storage cap rejects the commit)
   → `hf upload` from ai1 (the v4 image's hf 1.28 + xet; table shards dedupe against hibrid47, body shards against orcarouter's
   if xet dedups across repos — else ~76 GiB transfer) → set public. Then fill the commit id above.
