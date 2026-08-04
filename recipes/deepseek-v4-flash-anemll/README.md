# deepseek-v4-flash-anemll

DeepSeek-V4-Flash-0731 (304B MoE) served across **2× DGX Spark** (TP=2 over the ConnectX-7 link) using
**anemll's prebuilt dspark image** — with **dspark speculation ON** and **`nvfp4_ds_mla`** 4-bit KV, the
faithful jvr0x/dgx-spark-bench configuration. This is the **reference** box: it's how we confirmed dspark
serves this model on our cluster before building our own image.

## Files in this folder
| File | Purpose |
|------|---------|
| `myllmbox.yaml` | the whole recipe — model, engine flags, env, and the `cluster:` (which Sparks, NCCL) |
| `Dockerfile` | **provenance only** — documents anemll's image (their vLLM 0.25.2.dev / FlashInfer 0.6.15 / b12x MoE). We don't own that source; the recipe runs the prebuilt image directly. |
| `README.md` | this file |

## How it runs
Everything lives in this folder. From the repo root:

```bash
./run.sh deepseek-v4-flash-anemll  # launch: model container + keepalive proxy + cloudflared tunnel
```

`run.sh` reads `myllmbox.yaml`, and because it has a `cluster:` block with two `nodes`, the runner:
- starts the model on **this box** (node 0 = head, runs the OpenAI API) and ssh's a **`--headless` worker**
  onto the second Spark, joining one TP group over NCCL/RoCE (no Ray, `mp` executor);
- pins **all** cross-node traffic to the ConnectX link (`NCCL_*` + `VLLM_HOST_IP` + `GLOO_SOCKET_IFNAME`)
  because the LAN blocks arbitrary TCP ports;
- fronts it with the keepalive proxy (`:8011`); generation is gated by `BINDING_TOKEN`; `/v1/models` + `/health` public.

Weights are local at `models/deepseek-ai/DeepSeek-V4-Flash-0731` on **both** nodes → fully offline.

## Key config choices
- `image: ghcr.io/anemll/dspark-vllm-gx10:0.1.1` — prebuilt, already on both nodes (so no build/copy needed).
- `moe-backend: flashinfer_b12x` + the `VLLM_*B12X*` env — anemll's custom MXFP4 MoE (their image has it;
  our own build uses `deep_gemm` instead).
- `kv-cache-dtype: nvfp4_ds_mla`, `block-size: 256` — the 4-bit packed DS-MLA KV (jvr0x's original).
- `speculative-config … method: dspark, num_speculative_tokens: 5` — DSpark speculative decoding.
- `max-model-len: 300000`, `gpu-memory-utilization: 0.83`.

## Contrast with our own recipe
`../deepseek-v4-flash-0731/` is the **same model, our own image** — a from-source vLLM 0.26.0 build
(`Dockerfile` there), `moe-backend: deep_gemm`. Build & distribute it with `./build-and-copy.sh
deepseek-v4-flash-0731`, then `./run.sh deepseek-v4-flash-0731`. This anemll recipe is the reference to
compare against (speed / accuracy, `nvfp4_ds_mla` vs `fp8_ds_mla`).
