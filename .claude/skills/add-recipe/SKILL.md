---
name: add-recipe
description: Scaffold a myllmbox recipe for a new model — point it at a model repo / HF id and it writes recipes/<name>/ (myllmbox.yaml + optional Dockerfile) the house way. Use when the user wants to serve a new model, add a recipe, or port an old-structure recipe.
---

# Add a recipe — serve a new model on this box

A recipe is a folder: `recipes/<name>/myllmbox.yaml` (+ optional `Dockerfile`, `README.md`).
The **folder name IS the recipe** (`./run.sh <name>`), and the image it builds is `mbx-<name>`.
Nothing is enforced — the yaml is a thin pass-through to `docker run`/vLLM — so the skill's job is
to encode taste, not fight a schema. Read `CLAUDE.md` (the operating manual) before starting.

## 0 · What you need from the user

- The **source** — any of:
  - an **HF model id** (`org/name`) or model repo URL — you write the serving side;
  - **any GitHub repo that details how to run something** (someone's serving setup: a docker image,
    launch scripts, engine flags, a compose file) — you *port* it: wrap their runtime in a recipe so
    OUR runner drives it (see "Wrap someone else's serving repo" below).
  Read the card/README/config first — you must know: architecture (causal LM? diffusion? DiT video?),
  size (params → memory), and how it's served upstream (vLLM-supported? custom pipeline? their image?).
- **Which box(es)**: single Spark or the cluster (only if it can't fit one box).

## 1 · Pick the serve mode (the one real decision)

| Model kind | Mode | yaml shape |
|---|---|---|
| Text LLM / VLM that vLLM serves | **vLLM mode** (default) | `server.image` + `server.model` + `server.vllm:` flags — no `command:` |
| Diffusion / video / anything custom | **generic-server mode** | `server.command:` or `server.uvicorn:` — the recipe brings its own `server.py` (FastAPI, OpenAI-shaped) + `Dockerfile`; the runner injects NO vLLM flags |

Both get the identical front: keepalive proxy `:8011` + cloudflared tunnel + `BINDING_TOKEN` gate.
Generic servers should expose OpenAI-compatible routes (`/v1/images/generations`, `/v1/videos`, …)
and keep `server.port` = 8000 so the front never changes. `/v1/models` should answer (it's the public
model card the platform's sweep checks).

## 2 · Weights — local path, offline, automatic

Set `model: /models/org/name` (a **local path**). Do NOT pre-download by hand and do NOT add manual
download steps to a README — **`run.sh` resolves the model automatically**:

- present under `./models/` → serves it;
- a `/models/myllmbox/<name>-<fmt>` quant that isn't built → runs `./quantize.sh` (which downloads
  the BF16 source itself), then serves;
- a plain HF model that's missing → `./download.sh org/name`, then serves;
- cluster recipe → rsyncs the weights to every worker at the same absolute path (`--size-only`).

A bare HF id in `model:` also works (downloads into the pinned cache) — prefer the local path; it is
fully offline (`HF_HUB_OFFLINE` is auto-set for `/` paths).

## 3 · The quantizer block — add it when it applies

If the model benefits from our offline quant (diffusion transformers; large LLMs): serve the quant
(`model: /models/myllmbox/<Name>-nvfp4`) and add the `quantize:` block so `run.sh` can build it.
Why bother: the quantizer's torchao backend does **real W4A4 FP4 compute** (FP4 matmuls on Blackwell
tensor cores) — the checkpoint is both smaller AND faster (FLUX.2 measured ~3–4× vs BF16), unlike
weight-only quants which only save memory:

```yaml
quantize:
  source: /models/org/Name        # the BF16 base (quantize.sh downloads it if missing)
  model_type: diffusion           # diffusion | llm
  target: transformer             # the submodule to quantize (diffusion: transformer · llm: model)
  skip_modules: "..."             # accuracy-critical linears kept BF16 — VERIFY against the model's
                                  #   named_modules() on first boot; names differ per architecture
  format: nvfp4                   # default target
  formats: [nvfp4, mxfp8]         # what this model supports (quantize.sh rejects others)
```

The output is a **self-contained shareable checkpoint** under `models/myllmbox/`. It carries a
`myllmbox-quant.json` manifest; `run.sh` pre-flights torch/torchao versions against the serve image
and refuses on mismatch — so pin the same stack in the recipe `Dockerfile` as the quantizer used.

## 4 · Memory + cluster (GB10 hard rules)

- A Spark has **~119G usable unified memory**; keep `gpu-memory-utilization ≤ ~0.83`. Serve XOR
  build — a source build (~20–30G peak) does not fit next to a big serve.
- **Cluster only when one box can't hold it.** Then:
  ```yaml
  cluster:
    boxes: [box1, box2]       # NAMES from cluster.yaml — never raw IPs in a recipe (portable)
    master_port: 25000
  ```
  TP = box count is set automatically; NCCL/gloo pinning to the interconnect is the runner's job,
  not the recipe's. If the recipe has a `Dockerfile`, ship the image with `./build-and-copy.sh <name>`.

## 5 · Dockerfile — when and how

- Stock vLLM image or a hub image serves it? No Dockerfile — set `image:` directly. Pin third-party
  images **by digest** and treat them as references, not our build.
- Our own build (the norm for SM121/GB10 quirks and for generic servers): `Dockerfile` in the recipe
  folder — `run.sh` builds it as `mbx-<name>` automatically (`FROM mbx-base` for the vLLM stack, or
  its own stack for generic servers).
- **MANDATORY: if `image:` is any `mbx-*` (non-default, non-hub) image, THIS folder must contain the
  Dockerfile that builds it, named `mbx-<this-folder>`** — never point at another recipe's `mbx-*`
  image. A fresh user runs `./run.sh <name>`, hits "image not found", and has no way to know the image
  is born in a different folder — they'll conclude the recipe is broken. If two recipes need the same
  image content, COPY the Dockerfile into each folder (duplication is fine; cross-recipe image
  dependencies are not).
- **Review anything downloaded** before building/running — a recipe can do anything to the box.

## 6 · Write it, then verify for real

1. `recipes/<name>/myllmbox.yaml` — start from the closest existing recipe: `holo-3.1-35b` (single-node
   vLLM) · `deepseek-v4-flash-0731` (cluster vLLM) · `flux2-dev` (generic server + quantizer). Comment
   the non-obvious lines the way those do (the yaml is the recipe's documentation).
2. Optional `dashboard: sparkdash` if the user wants the monitoring UI.
3. `./run.sh <name>` — watch the streamed load logs.
4. **Verify with a real request, never just "it came up"**: a real completion / image / clip through
   the proxy (`127.0.0.1:8011`, `Authorization: Bearer $BINDING_TOKEN`) and confirm `/v1/models`
   answers. For a cluster: from the head, after both nodes join.
5. `README.md` in the recipe folder only when there's something non-obvious to say (licenses,
   known-good timings, resolution floors, flag rationale) — keep facts measured, not guessed.

## Wrap someone else's serving repo (any GitHub repo that runs a model)

A repo that documents how *they* serve a model — an image, launch scripts, NCCL/engine flags — ports
easily: the recipe wraps **their runtime** so our runner (cluster launch + keepalive proxy + tunnel)
drives it instead of their scripts. Two proven shapes, both live in this repo:

- **Their prebuilt image, pinned by digest** — `recipes/deepseek-v4-flash-r0b0tlab/`: `image:` is
  their GHCR tag `@sha256:…`, their engine flags translated into `server.env` + `server.vllm:`,
  `entrypoint: vllm` overriding their launcher. No Dockerfile at all.
- **Their image + their launch semantics as a reference A/B** — `recipes/deepseek-v4-flash-anemll/`
  (wraps github.com/Anemll/dspark-vllm-gx10): same idea, kept as a reference point next to our own
  build of the same model.

The flow — **everything from git goes to `.data/`, then review, then build the recipe from what's
needed**:

1. **Download it all to `.data/`** — clone their repo into `recipes/<name>/.data/` (the one
   gitignored runtime dir per recipe, same convention as dashboards). Their code is never committed
   here; the recipe carries only our yaml (+ optional Dockerfile).
2. **Review it** — before building anything: their run scripts, Dockerfile/compose, what the image
   fetches/executes, which host paths it mounts. Their image runs with GPU + mounts on your box; a
   repo can do anything to it. Only what survives review informs the recipe.
3. **Build the recipe according to what's needed** — translate only the engine configuration:
   their `docker run`/image → `server.image` (pin by digest) · their env → `server.env` · their CLI
   flags → `server.vllm:` (or `server.command:` for non-vLLM) · their entrypoint →
   `server.entrypoint`. **Drop their plumbing** — cluster launch, NCCL/gloo pinning, proxying,
   tunneling, keepalive are our runner's job.
4. Credit + license: name the source repo in the recipe comments/README, keep their flags labelled
   as theirs, respect their LICENSE (reference the image by digest; don't redistribute weights).
   Treat their image as a reference — prefer building our own once validated.

## Porting an old-structure recipe (pre-runner yamls)

Old yamls (`recipe_version: "1"`, `container:`, `mods:`, `run-recipe.sh`, LAN ports 8001/8444) map as:
`container:` → `server.image` · `command:` → `server.command`/`uvicorn:` · `env:` → `server.env`
(DROP the `HTTP2_PROXY_*` sidecar vars — the runner's proxy replaces that whole mod) · `mounts:` →
`server.mounts` (models dir is already mounted) · `solo_only` → just omit `cluster:`. Keep the model
facts (quant config, skip_modules, defaults, verified timings); rewrite the plumbing. Endpoints are
reached through the tunnel URL now, never `http://<host>:8001`.

## The gauntlet (every text-model recipe)

A recipe isn't done until `tests/` holds the TWO standing gauntlet outputs (canonical prompts in
`bench/`): text models render `pasture.html` + `fish.html` (bench/pasture-text.txt / fish-text.txt,
one-shot, raw HTML as generated); image models generate `pasture.png` + `fish.png`
(bench/pasture-image.txt / fish-image.txt — HD-quality directives included; edit-only models take a
plain reference input per their API). Best of 3 runs each. The USER runs the generations (never
fire test load yourself); commit their picked winners. Same checklist scenes across every model =
a countable capability comparison — the recipe's quality proof, shown on the site's model page
next to the speed stats. Reference example: `recipes/qwen38-flash-next/tests/`.
