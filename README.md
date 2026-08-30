# myllmbox

Turn a machine — or **N DGX Sparks** — into an LLM box with one command. A *recipe* is a folder; running it
brings up the model container, a keepalive proxy, and a cloudflared tunnel. You never touch a venv,
docker args, or NCCL flags — it's all in the recipe.

```bash
./run.sh qwen38-flash-next-solo-hibrid   # our best model: image build + weights download +
                                         # checkpoint self-conversion + serve — ONE command
```

That recipe is the house champion: **Qwen3.8-Flash-Next (176B multimodal MoE) on a single DGX Spark at
50–57 tok/s on code** — faster than the publisher's own hosting — via a custom-quantized checkpoint
(“hibrid45”) the recipe builds for itself on first boot. Its `README.md` tells the whole story.

The general shape, for any recipe:

```bash
./download.sh <hf-id>                  # fetch weights into ./models/ (run.sh also does this when needed)
./build-and-copy.sh <recipe>           # build the recipe's image, copy it to the other cluster nodes
./run.sh <recipe>                      # launch: model + proxy + tunnel (TP across the cluster)
```

The scripts self-bootstrap their own `.venv` on first use. The folder name **is** the recipe.

## Measured on the reference box (DGX Spark GB10, clean decode windows)

| recipe | model | boxes | speed | KV pool |
|---|---|---|---|---|
| `qwen38-flash-next-solo-hibrid` | Qwen3.8-Flash-Next 176B (VLM, thinks) | 1 | **50–57 tok/s code · ~32 avg** | ~800k tok @262k ctx |
| `deepseek-v4-flash-0731` | DeepSeek-V4-Flash 304B (thinks) | 2 | ~40–50 tok/s (re-measure pending) | 620k tok @300k ctx |
| `holo-3.1-35b` | Holo-3.1-35B-A3B (vision) | 1 | ~75 tok/s flat | 3.4M tok @131k ctx |
| `flux2-dev` | FLUX.2-dev (images) | 1 | ~45s / 1024² warm | — |

Speeds are honest numbers: sustained bands and clean single-stream averages, not cherry-picked peaks.

---

## Layout

```
run.sh  build-and-copy.sh  download.sh   ← the three entry points (each takes a recipe folder name)
Dockerfile                               ← the base box: FROM vllm/vllm-openai → mbx-base
runner/                                  ← the Python engine (cli · config · supervisor · vllm · proxy · tunnel)
recipes/<name>/                          ← one folder per model
    myllmbox.yaml                        ← the whole recipe (model, flags, env, cluster)
    Dockerfile      (optional)           ← how this recipe's image is built → mbx-<name>
    README.md       (optional)           ← notes for that recipe
models/                                  ← weights, gitignored, mounted into the container at /models
```

## A recipe: `recipes/<name>/myllmbox.yaml`

```yaml
server:
  image: mbx-<name>                    # image to run (mbx-<name> from the recipe Dockerfile, or any tag/hub image)
  entrypoint: vllm                     # optional: override the image's entrypoint (→ `vllm serve`);
                                       #   recipes with first-boot work point it at their own wrapper script
  model: /models/org/name              # a local /path (offline) OR an HF id (downloads to the cache)
  cache: recipes/<name>/.data/cache    # optional persistent host dir → /cache (compile caches, quant caches,
                                       #   autotune configs — later boots skip the first-boot cost)
  cpuset: "5-9,15-19"                  # optional --cpuset-cpus pin (e.g. the GB10 performance cores)
  env:                                 # name: value → -e NAME=value into the container
    HF_HUB_OFFLINE: "1"
  vllm:                                # vLLM CLI flags — a human-readable dict, NOT an alternating list:
    tensor-parallel-size: 2            #   name: value  → --name value
    trust-remote-code: true            #   name: true   → --name   (bare flag)
    kv-cache-dtype: fp8                #   name: false/null → omitted
  # sglang: {...}                      # a non-empty flag dict here = SGLang mode (launch_server) instead of vLLM
  # command: "python3 -m my_server"    # or: run ANY OpenAI-compatible server (image/video boxes use this)

quantize:                              # OPTIONAL: the model above is DERIVED — run.sh builds it when missing
  source: /models/org/source-ckpt      #   source downloaded first if absent
  script: make-<name>.sh               #   recipe-local converter (checkpoint surgery), or omit `script`
                                       #   to use the generic quantizer (nvfp4/mxfp8)
  out: /models/myllmbox/<derived-name>

cluster:                               # OMIT for single-node. Present = serve ONE model across these boxes.
  boxes: [box1, box2]                  #   names defined in cluster.yaml (host/interconnect/iface/ib_hca/
                                       #   ssh_user). box 0 = head (runs the API); the rest join --headless.
  master_port: 25000                   #   No machine IPs in the recipe → portable. Provision: cluster/setup.sh

dashboard: sparkdash                   # OPTIONAL web UI → dashboards/sparkdash/ (its up.sh/down.sh + port).
                                       #   Any UI is your taste (mia/sparkDash, a robot, your own).
```

The `vllm:` flags block is a **dict** (readable), not a list. `true` → bare flag; a value → `--name value`;
`false`/null → skipped.

**The dashboard is a pattern, not a product.** `dashboard: <name>` points at `dashboards/<name>/` (its own
`up.sh`/`down.sh` + a `dashboard.yaml` giving the `port`). The proxy then routes `/v1/*` to the model and
**every other path** to that UI at `127.0.0.1:port`. Set a `DASHBOARD_PASSWORD` in `.env` and the proxy gates
it with HTTP Basic auth (so a UI with no auth of its own — e.g. one with a shutdown button — is safe through
the tunnel); leave it empty and the UI is served ungated (its own guard rails, or your explicit public choice).
The runner is UI-agnostic — `up.sh` does whatever that UI needs (build, generate config, run its container),
and the recipe's box set arrives as `$MBX_BOXES`. Swap `<name>` for sparkDash/mia, a "robot", your own — same
routing, same URL. No `dashboard:` → model-only, exactly as before. See `dashboards/`.

**Nothing here is enforced.** The recipe is a thin pass-through, not a validated schema — `image` can be any
docker image, `extra_args` maps to *any* vLLM flag, `env` sets any variable, `mounts` binds any host path. You
configure a new model / backend / flag by editing this yaml, not the runner. That power cuts both ways: a recipe
can do anything to the box, so review a downloaded one before running it (below). See `CLAUDE.md` for the full
operating manual (conventions, cluster rules, memory limits) — especially if you're an agent working here.

## Adding a recipe (and installing a downloaded one)

A recipe is just a folder under `recipes/`. To add your own:

1. **Make `recipes/<name>/`** with a `myllmbox.yaml` (format above). Optionally add a `Dockerfile`
   (→ built as `mbx-<name>`) and a `README.md`.
2. **Get the weights** — `./download.sh <hf-id>` puts them in `models/<hf-id>`; then set
   `model: /models/<hf-id>` (local, fully offline). Or leave an HF id in `model:` to let vLLM download into
   the cache. Weights must exist at the **same path on every cluster node** (see the DS recipe).
3. **Build** — only if the recipe has a `Dockerfile`: `./build-and-copy.sh <name>` builds `mbx-<name>` and
   copies it to the other cluster nodes. Skip this if the recipe runs a stock/hub image directly.
4. **Run** — `./run.sh <name>`.

**Installing someone else's recipe.** If you download a recipe (a folder with a `myllmbox.yaml`, maybe a
`Dockerfile`), drop it in as `recipes/<name>/` and it works the same way — **but read it first.** A recipe can
name any docker `image:`, set arbitrary `env:` and `mounts:`, and its `Dockerfile` runs arbitrary steps on your
box during build. Before `build-and-copy`/`run`, check:
- **`image:`** — where it comes from (a hub image runs on trust; prefer building from a `Dockerfile` you can read).
- **`mounts:`** — which host paths it exposes into the container.
- the **`Dockerfile`** — what it fetches/executes.
- **`cluster.boxes`** — box names resolve through *your* `cluster.yaml` (hosts, interconnect IPs, ifaces),
  so a downloaded recipe never carries someone else's addresses — but make sure your `cluster.yaml` boxes
  match your hardware before running a multi-box recipe.

## What `run.sh` does
1. bootstrap `.venv` (aiohttp, pyyaml, click, python-dotenv)
2. build `mbx-base` once (the base box) if a recipe Dockerfile `FROM mbx-base` needs it
3. build `mbx-<recipe>` from `recipes/<recipe>/Dockerfile` if present
4. resolve the model: a missing `/models/myllmbox/*` model with a `quantize:` block triggers `./quantize.sh`
   (source downloaded if absent, then the recipe's converter builds the derived checkpoint)
5. `runner.cli up` — start the model (single-node or cluster), then the keepalive proxy + cloudflared tunnel

**Single node** = the degenerate case of a cluster with one node. **Multi-node** (`cluster:` with ≥2 nodes):
the runner runs the head locally and `ssh`'s a `--headless` worker onto each other node, one tensor-parallel
group over NCCL. It sets `--tensor-parallel-size` = node count automatically, and pins **all** cross-node
traffic (NCCL + gloo/mq) to the interconnect via `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA`, `GLOO_SOCKET_IFNAME`,
and a per-node `VLLM_HOST_IP` — because a management LAN often blocks the arbitrary TCP ports the mq needs.

## What `build-and-copy.sh` does
Builds `recipes/<recipe>/Dockerfile` → `mbx-<recipe>` on this node, then `docker save | ssh docker load`
(with a `pv` progress bar) to every **other** node in the recipe's `cluster:` block. Build once, distribute
to the workers — a heavy source build (e.g. vLLM from source for a specific GPU arch) runs on the head only.

## How it's exposed: keepalive proxy + cloudflared tunnel

The model listens only on `127.0.0.1` inside the box — never a public port. Two layers put it online:

**1. Keepalive proxy** (`runner/proxy.py`, port `8011`) — the box's local front door. It reverse-proxies to the
vLLM backend and does two jobs:
- **keeps long-silence connections alive.** A model that "thinks" for minutes before its first token would
  otherwise have its connection dropped mid-request — Cloudflare cuts idle connections at ~100s. The proxy
  sends periodic keepalive bytes so the stream survives the wait.
- **gates generation.** `/v1/chat/completions`, `/v1/completions`, … require
  `Authorization: Bearer <BINDING_TOKEN>`. `/v1/models` and health checks stay public. **No `BINDING_TOKEN` in
  `.env` = a FULLY PUBLIC box** (generation included) — the runner logs that loudly on startup, never silently.

**2. cloudflared tunnel** (`runner/tunnel.py`) — makes the proxy reachable on the internet with **no open inbound
ports and no public IP.** cloudflared dials *outbound* to Cloudflare's edge and holds a persistent connection;
Cloudflare then routes your subdomain's traffic back down that tunnel to the local proxy.

Request path:

```
client ──HTTPS──▶ Cloudflare edge ──tunnel──▶ cloudflared ──▶ keepalive proxy :8011 ──▶ vLLM 127.0.0.1:8000
                 (your subdomain)                             (auth + keepalive)         (the model)
```

### The two tokens (both from `.env`)
- **`TUNNEL_TOKEN`** *(required)* — the cloudflared tunnel token, and the **whole binding**: the tunnel and the
  DNS route for your subdomain were provisioned when this token was minted, so the box just *connects* with it —
  it never creates or configures anything. Paste it from your console's Box-setup reveal. A stale/revoked token
  shows up as `Tunnel not found` in `.mbx/cloudflared.log`.
- **`BINDING_TOKEN`** *(optional)* — the bearer token the proxy checks to allow generation. Absent = public.

Lifecycle: `up` starts the model → spawns the proxy → spawns the tunnel; `down` tears them down in reverse.
`wait_healthy` polls the **local** proxy `/v1/models`, so "up" means the model+proxy serve locally — it does not
itself prove the tunnel connected; check `.mbx/cloudflared.log` for that.

## `.env`
```
TUNNEL_TOKEN=...     # required — the cloudflared tunnel token (the whole binding)
BINDING_TOKEN=...    # optional — gate generation; ABSENT = fully public box (logged loudly)
PUBLIC_URL=...       # your subdomain, for display
```
