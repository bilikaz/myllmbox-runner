# Qwen3.8-Flash-Next HIBRID45 — the fastest quality-gated single-Spark serve we know of

One DGX Spark (GB10, 119G unified memory) serving a 176B-A6B multimodal MoE at **50–57 tok/s on
code, ~31–33 tok/s averaged over a real mixed diet** — past Alibaba's own 53 tok/s hosting for this
model, and roughly double the community's dual-Spark TP=2 configs (~30). Pasture-gated: output
quality judged "near identical to pre-compression."

## What "hibrid45" is

Nobody ships a checkpoint like this, so the recipe builds it on first boot (CPU-only, ~10 min,
hardlinks — a "173G" checkpoint owns only ~1.2G of new bytes). It is a **per-module precision map**
of Inferact's NVFP4 checkpoint, learned the hard way:

| slice | precision | why |
|---|---|---|
| routed experts | nvfp4 (Inferact) | the bulk; tolerant |
| GDN in/out projections + shared expert | **nvfp4 (ours)** | multiplicative path — measured zero quality/acceptance cost |
| QSA q/k/v/o attention | **bf16, untouched** | *comparators*: q·k dot products rank tokens, and rankings flip on quant noise |
| lm_head / embeds / MTP / PLE / norms | bf16 | the quality-critical set (mirrors Alibaba's own fp8 exclusions) |

The name: converted modules are 4-bit weights + fp8 group-16 scales = **4.5 effective bits**.

### How the map was learned (one day, three checkpoints)

1. **fp8 dense ("hibrid", retired)** — all 300 dense modules to fp8: +30% steps, 50.0 peak. Champion for a day.
2. **hibrid4 (retired)** — all 300 modules to nvfp4: 53.5 peak but *intermittent intelligence drop*
   (pasture defects, prose acceptance collapse to ~2.1). 4-bit attention was the poison.
3. **hibrid45 (this)** — nvfp4 everywhere tolerant, bf16 attention. Kept the speed, fixed the brains —
   and its bf16 comparators *beat even the fp8 champion's acceptance* on a paired same-prompt A/B
   (2.55–3.04 vs 1.81–2.30). Higher-precision attention is literally faster here, because the MTP
   drafter agrees with the target more.

## The K story: why speculative depth is 3, and when to use 4

The MTP drafter proposes K tokens per engine step; the model verifies K+1 in one pass. Acceptance
length (tokens kept per step) times step rate is your throughput. We measured K=2 (old campaigns),
K=3, and K=4 on this exact stack, clean single-stream windows only (no prefill contamination):

| clean c=1, same box, same checkpoint | **K=3 (default)** | K=4 |
|---|---|---|
| session average | **32.6 tok/s** | 31.4 |
| median | **30.7** | 28.3 |
| floor (p10) | **24.5** | 22.6 |
| p90 | 44.1 | **46.1** |
| peak window | 52.8 | **56.9** |
| code sustained | 50–52 | **54–57** |
| steps/s | 12.07 | 10.13 |

What's going on:

- **Each +1 of K costs step rate twice**: one more sequential draft pass, and one more verified
  token dragging its own MoE experts through DRAM (the verify-width byte cost). Measured: −16%
  steps going 3→4.
- **Rejections cost more than acceptances**: a rejected draft forces a GDN/SSM state rollback and
  recompute, so low-acceptance content (prose, thinking tokens) pays the K tax double. High-accept
  code barely pays it — K=4 step rate recovers from 9.9 to 11.4 on clean code.
- **The drafter is scarily good at depth on code**: K=4 per-position acceptance hit
  0.991/0.991/0.983/0.983 (98.7% — a 4.95/5 window). Depth is not the limit; bandwidth is.
- **But real work is secretly full of prose**: even coding-agent sessions spend most tokens on
  reasoning. K=4 stretched the distribution (better top decile, worse everywhere else) and lost
  the average.

**Verdict: K=3 is the daily driver.** Flip `num_speculative_tokens` to 4 only for sessions that are
almost pure code generation — it buys the 54–57 band there and the 56.9 record. K=2 was tested
long ago and loses everywhere; K=5 is parked (the same verify-width math says it needs a
code-only diet to break even, at an even lower floor).

## Records (2026-08-30, this box, pasture-gated unless noted)

- **56.9 tok/s** peak window (K=4, code, accept 4.95/5)
- **54–57** sustained code (K=4) / **50–52** (K=3)
- **70.5 tok/s** two-stream aggregate (hibrid4, quality-fail checkpoint — expect ~similar here; the
  2nd stream costs only ~⅓ of per-stream speed)
- **~3019 tok/s warm ingest** (93% prefix-cache hit) — cold-prefill number not yet formally measured

## Reading the metrics (vocabulary)

- **decode** — tok/s while generating; judge configs on *clean c=1 windows* (no prefill in window).
- **drafted throughput ÷ K = steps/s** — the pure engine speed, acceptance-independent. Judge
  kernel/config changes by THIS, never by generation tok/s (accept-confounded).
- **acceptance length** — tokens kept per step (ceiling K+1). Judge *content* and *checkpoint
  quality* by this. A prose window at 2.5 is normal (the fp8 champion did 1.8–2.3); a *code* window
  below ~3.5 (K=3) means something is wrong.
- **cold prefill vs warm ingest** — fresh-prompt compute vs prefix-cache replay; the latter is what
  agent clients feel as time-to-first-token.

## Ops notes

- First boot converts the checkpoint automatically (entrypoint fast-skips once complete);
  `./make-hibrid45.sh` pre-builds it manually — CPU-only, safe while another model is serving.
- The image carries `MBX_INDEX_STRICT=1`: hardlinked hibrid checkpoints leave stale originals in
  the source shards, and vLLM iterates *files*, not the index — without the patch the packed 4-bit
  weights shape-assert on the stale bf16 copies at load.
- The 25,000,000,000-byte KV pin buys a **~800k-token KV pool** (~790k measured at the 25 GiB
  variant) — about **3× full 262k-context concurrency**, or dozens of typical agent sessions with
  prefix caching. The round number is deliberate: the full 25 GiB booted with ~240MB free and 5G of
  swap at the flashinfer-autotune peak. If a boot hard-OOMs anyway, warm the autotune cache once
  at a lower pin, then relaunch — don't raise `gpu-memory-utilization` for a boot-time transient.
- `VLLM_MARLIN_USE_ATOMIC_ADD=1` is self-gating (small-N GEMMs only); `MBX_VOCAB_GEMV=1` is the
  vendored M=1 lm_head Triton GEMV (+5.5% steps, fires every draft pass).
