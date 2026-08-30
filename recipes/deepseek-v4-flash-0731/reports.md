# deepseek-v4-flash-0731 — benchmark report (Marlin build)

Our own SM121 build (vLLM `20260803.dev0`, jasl PR #41834, ~v0.26-era). MoE = **Marlin** (the compact-fp4 path
is NOT reachable here — that's the SM121 tax). 2× DGX Spark GB10, TP=2. Same-schema companion:
`experiments/deepseek-v4-flash/reports.md (retired lane)` (B12X).

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
| _prior/memory_ | 2 | mixed | **~70–80** | ~35–40 | — | — | — | Valdas recollection — beats B12X c2 ~53; confirm in sweep |
| _prior/memory_ | 6 | mixed | **~120** | ~20 | — | — | — | Valdas recollection — confirm in sweep |
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
- Verdict vs B12X: **UNSETTLED — needs the shared sweep.** Single-stream ~parity (~50 peak both). B12X *does*
  scale with concurrency (c1 ~40 → c2 ~53 → c3 ~70, KV headroom) — the earlier "Marlin wins concurrency" call
  was based on a low-accept B12X c2 window and is retracted. Marlin's structural edges remain: bigger KV pool
  (621k vs 443k) and lighter/less-hassle build. Run the identical c1–c16 harness on both to decide.
