# deepseek-v4-flash — benchmark report (B12X build)

vLLM **B12X** fork (`local-inference-lab/vllm` @ dev/gilded-gnosis + `lukealonso/b12x` kernels). 2× DGX Spark
GB10 (SM121), TP=2 over ConnectX. Same-schema companion: `recipes/deepseek-v4-flash-0731/reports.md` (Marlin).

## Config (as measured below)
- image `mbx-deepseek-v4-flash` · model `DeepSeek-V4-Flash-0731`
- `gpu-memory-utilization 0.85` · `kv-cache-dtype fp8_ds_mla` · `block-size 256`
- `max-num-seqs 8` · `max-num-batched-tokens 8192`
- spec: **dspark**, K=5, probabilistic · MoE: **B12X Mxfp4 (expert_dtype fp4)** · mHC · InstantTensor loader
- **KV pool: 443,190 tok** (cudagraph tax: 0.85 → eff 0.837) · max concurrency @300k ctx = 1.48×

## Runs

| date | c | content | aggregate tok/s | per-req tok/s | mean acc len | draft accept % | KV use % | notes |
|------|---|---------|-----------------|---------------|--------------|----------------|----------|-------|
| 2026-08-13 | 1 | structured (best case) | ~50–51 | ~50–51 | 6.00 (max) | 100% | 0.6% | cold cache; box swapping ~3 GB |
| 2026-08-13 | 1 | prose/mixed | ~33–36 | ~33–36 | 3.8–4.2 | 56–64% | 0.5% | content-driven floor |
| 2026-08-13 | 2 | mixed | ~52–56 (peak 65) | ~26–28 | 4.0–5.3 | 55–86% | 1.6% | low-accept window; not a wall (see c3) |
| 2026-08-13 | 3 | mixed | ~65–72 (peak 82) | ~22–24 | 3.2–5.1 (~4.3) | 37–82% (~63%) | ~2% | healthy scaling; KV barely touched |
| 2026-08-13 | 4 | — | _TODO_ | | | | | sweep pending — KV headroom suggests higher |
| 2026-08-13 | 8 | — | _TODO_ | | | | | sweep pending |

## Notes / verdict
- **Throughput is dspark-acceptance-driven → content-dependent.** Single-stream swings ~33 (56% accept) → ~51
  (100% accept, mean acc 6.0). So compare only at matched content / same-prompt harness.
- **Single-stream: ~parity with Marlin (~50 peak).** **Concurrency: scales** — c1 ~40 → c2 ~53 → **c3 ~70**
  (~1.7× at c3), **KV usage only ~2% → lots of headroom** (c4/c8 pending). The earlier "weak c1→c2" read was a
  low-acceptance content window, NOT a wall. The ~3 GB swap at c1 was the one-time cold TileLang compile, not a
  steady-state ceiling (persistent cache now in the recipe should remove it on a warm boot).
- **Reaches compact-fp4 MoE** (the path Marlin can't on SM121) — but comes out **heavier & no faster**. KV pool
  443k < Marlin's 621k (0.89/12288). vLLM warned `max-num-batched-tokens 8192` is tight for spec decode.
- **TODO for a fair number:** warm-cache re-run (persistent TileLang cache now in the recipe — needs one warm
  reboot), try `max-num-batched-tokens 12288`, and the full c1–c16 sweep vs Marlin in one harness.
