# deepseek-v4-flash-r0b0tlab

DeepSeek-V4-Flash-0731 (304B MoE) served across **2× DGX Spark** (TP=2 over the ConnectX-7 link) using
**r0b0tlab's prebuilt vLLM 0.26.0 SM121 image** — **native regular cudagraphs** (`FULL_DECODE_ONLY`,
`torch.compile` OFF), **`flashinfer_b12x`** MoE, **`nvfp4_ds_mla`** 4-bit KV, and **dspark K6** speculation.
This is a **reference** box — a third serving stack to A/B against our own build and the anemll one.

## Credit — this is r0b0tlab's work, not ours

The container image **and** the engine configuration below are **taken from r0b0tlab's** release:

> **https://github.com/r0b0tlab/DeepSeek-V4-Flash-DSpark-v026-SM121**

All we did is wrap his pinned image in a myllmbox recipe so our runner (cluster launch + keepalive proxy +
tunnel) drives it instead of his `scripts/run-dspark-dual-gb10.sh`. The image, flags, and tuning are **his**;
credit and thanks go to r0b0tlab. His repo notes it is *not* an official DeepSeek or vLLM release, and model
weights are not redistributed (Apache-2.0 / MIT per his LICENSE/NOTICES). We only reference the image by digest.

## Files in this folder
| File | Purpose |
|------|---------|
| `myllmbox.yaml` | the whole recipe — his pinned image, his engine flags, env, and the `cluster:` block |
| `README.md` | this file |

No `Dockerfile` — this runs r0b0tlab's **prebuilt** image directly (`ghcr.io/r0b0tlab/…`, pinned by sha256).

## How it runs
Both Sparks must be free (a 304B serve is ~100G/node — can't co-host). From the repo root:

```bash
./run.sh deepseek-v4-flash-r0b0tlab      # model container + keepalive proxy + cloudflared tunnel
```

Because `myllmbox.yaml` has a `cluster:` block (`boxes: [box1, box2]`), the runner starts the model on **this
box** (node 0 = head, runs the OpenAI API) and ssh's a **`--headless` worker** onto the second Spark, joining
one TP=2 group over NCCL/RoCE (`mp`, no Ray); it pins all cross-node traffic to the ConnectX link and fronts
it with the keepalive proxy (`:8011`, gated by `BINDING_TOKEN`). Weights are local on both nodes → offline.

## Key config choices (all r0b0tlab's)
- `image: ghcr.io/r0b0tlab/deepseek-v4-flash-dspark-v026-sm121:v0.26.0-sm121-native-v11@sha256:ef852781…` — pinned.
- `compilation-config: {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_implementation":"regular",…}`
  — native/regular cudagraphs, `torch.compile` OFF (his note: it garbles output on dual-GB10).
- `moe-backend: flashinfer_b12x`, `kv-cache-dtype: nvfp4_ds_mla`, `kv-cache-memory-bytes: ~14.9 GiB`.
- `speculative-config … method: dspark, num_speculative_tokens: 6` (his "K6").
- `max-model-len: 327680`, `max-num-seqs: 16`, `gpu-memory-utilization: 0.835`.
- `override-generation-config: {"temperature":0.6,"top_p":0.95}` — **our** house default (no penalty; penalties
  corrupt JSON/tool-calls). Everything else is his.

## Contrast with the other two
- `../deepseek-v4-flash-0731/` — **our own** from-source vLLM 0.26.0 build (#41834), `fp8_ds_mla` KV, cudagraphs
  via that branch's SM12.0a hardening.
- `../deepseek-v4-flash-anemll/` — anemll's prebuilt (0.25.2-era), also `flashinfer_b12x` + `nvfp4_ds_mla`.

This recipe is the r0b0tlab reference to compare against — native-cudagraph + compile-off vs our path — on
speed, accept rate, and output quality.
