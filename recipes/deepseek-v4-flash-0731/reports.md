# deepseek-v4-flash-0731 — benchmark report (Marlin build)

Our own SM121 build (vLLM `20260803.dev0`, jasl PR #41834, ~v0.26-era). MoE = **Marlin** (the compact-fp4 path
is NOT reachable here — that's the SM121 tax). 2× DGX Spark GB10, TP=2. Same-schema companion:
`recipes/deepseek-v4-flash/reports.md` (B12X).

## Config
- image `mbx-deepseek-v4-flash-0731` · model `DeepSeek-V4-Flash-0731`
- `gpu-memory-utilization 0.89` · `kv-cache-dtype fp8_ds_mla` · `block-size 256`
- `max-num-seqs 6` · `max-num-batched-tokens 12288`
- spec: **dspark**, K=5, probabilistic · MoE: **Marlin** (auto) · `enable-expert-parallel`
- **KV pool: 621,865 tok** (cudagraph tax: 0.89 → eff 0.883) · max concurrency @300k ctx = 2.07×

## Runs

| date | c | content | aggregate tok/s | per-req tok/s | mean acc len | draft accept % | KV use % | notes |
|------|---|---------|-----------------|---------------|--------------|----------------|----------|-------|
| _prior/memory_ | 1 | mixed | ~40–50 | ~40–50 | ~2.3–4 | — | — | remembered, NOT same-harness — re-measure |
| _prior/memory_ | 2 | mixed | ~53 | ~26 | — | — | — | remembered, re-measure |
| _TODO_ | 1 | structured | | | | | | measure in the shared harness |
| _TODO_ | 1 | prose | | | | | | |
| _TODO_ | 2 | mixed | | | | | | |
| _TODO_ | 4 | — | | | | | | |
| _TODO_ | 8 | — | | | | | | |

## Notes / verdict
- **Baseline keeper (production).** KV pool 621k @0.89/12288 — bigger than B12X's 443k @0.85/8192 → more
  concurrency headroom at long context.
- Prior rows are from memory / earlier sessions, **not** the shared A/B harness — treat as ~ until re-measured
  head-to-head with `recipes/deepseek-v4-flash` on identical prompts (same model + same dspark ⇒ same accept ⇒
  isolates per-step kernel speed, the real question).
- Emerging verdict vs B12X: **single-stream ~parity; concurrency Marlin wins** (B12X memory-bound/swaps, scales
  ~1.4× c1→c2 vs Marlin's stronger scaling). Lighter, less hassle. B12X kept as the fp4-MoE reference.
