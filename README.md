# myllmbox

Turn a machine — or **N DGX Sparks** — into an LLM box with one command. A *recipe* is a folder; running it
brings up the vLLM model container, a keepalive proxy, and a cloudflared tunnel. You never touch a venv,
docker args, or NCCL flags — it's all in the recipe.

```bash
./download.sh Qwen/Qwen3-0.6B          # fetch weights into ./models/
./build-and-copy.sh <recipe>           # build the recipe's image, copy it to the other cluster nodes
./run.sh <recipe>                      # launch: model + proxy + tunnel (TP across the cluster)
```

The scripts self-bootstrap their own `.venv` on first use. The folder name **is** the recipe.

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
vllm:
  image: mbx-<name>                    # image to run (mbx-<name> from the recipe Dockerfile, or any tag/hub image)
  entrypoint: vllm                     # optional: override the image's entrypoint (→ `vllm serve`)
  model: /models/org/name              # a local /path (offline) OR an HF id (downloads to the cache)
  gpu_memory_utilization: 0.8
  env:                                 # name: value → -e NAME=value into the container
    HF_HUB_OFFLINE: "1"
  extra_args:                          # human-readable flags, NOT an alternating list:
    tensor-parallel-size: 2            #   name: value  → --name value
    trust-remote-code: true            #   name: true   → --name   (bare flag)
    kv-cache-dtype: fp8                #   name: false/null → omitted

cluster:                               # OMIT for single-node. Present = serve ONE model across these boxes.
  boxes: [box1, box2]                  #   names defined in cluster.yaml (host/interconnect/iface/ib_hca/
                                       #   ssh_user). box 0 = head (runs the API); the rest join --headless.
  master_port: 25000                   #   No machine IPs in the recipe → portable. Provision: cluster/setup.sh
```

`extra_args` is a **dict** (readable), not a list. `true` → bare flag; a value → `--name value`; `false`/null → skipped.

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
- **`cluster.nodes`** — set these to *your* interconnect IPs and `nccl_ifname`/`nccl_ib_hca`; someone else's
  values won't match your hardware.

## What `run.sh` does
1. bootstrap `.venv` (aiohttp, pyyaml, click, python-dotenv)
2. build `mbx-base` once (the base box) if a recipe Dockerfile `FROM mbx-base` needs it
3. build `mbx-<recipe>` from `recipes/<recipe>/Dockerfile` if present
4. `runner.cli up` — start the model (single-node or cluster), then the keepalive proxy + cloudflared tunnel

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
