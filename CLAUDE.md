# CLAUDE.md — working in this repo (for agents)

This is **myllmbox**: turn a machine (or N DGX Sparks) into an LLM box. A *recipe* is a folder under
`recipes/`; running it brings up the vLLM container + keepalive proxy + cloudflared tunnel. Read `README.md`
for the user-facing overview. This file is the operating manual — the conventions and the traps.

## The core idea: nothing is enforced — you can run anything

A recipe's `myllmbox.yaml` is a **thin pass-through**, not a validated schema. There is no allow-list, no
required fields beyond `vllm.model`, no sanitising. Whatever you put in flows straight to `docker run` / vLLM:

- **`vllm.image`** — *any* docker image (a hub tag, a local `mbx-*` you built, anything). Not checked.
- **`vllm.extra_args`** — a dict mapped verbatim to CLI flags: `name: value → --name value`, `name: true →
  --name`, `name: false/null → omitted`. Any vLLM flag works; you don't edit Python to add one.
- **`vllm.env`** — `NAME: value → -e NAME=value` into the container. Any env var.
- **`vllm.entrypoint`** — override the image's entrypoint (e.g. `vllm` → `vllm serve`).
- **`vllm.mounts`** — extra `host:container` binds, raw.
- **`cluster`** — presence turns on multi-node; absence = single node. You define the nodes.
- **`dashboard`** — OPTIONAL web UI. `dashboard.image` (+ `port`/`env`/`mounts`/`command`) is any container the
  runner runs on the head; the proxy then sends `/v1/*` to the model and **every other path** to it, HTTP-Basic-
  gated by `DASHBOARD_PASSWORD` (.env). UI-agnostic (sparkDash/mia, a robot, your own) — see `runner/proxy.py`,
  `runner/dashboard.py`. No block → model-only, unchanged.

So to serve a new model / try a flag / a different backend, you **edit the recipe's yaml** — you rarely touch
`runner/`. The runner just assembles the `docker run`, launches the cluster, and fronts it with proxy+tunnel.
This freedom is the point; it also means **a recipe can do anything to the box** — see "Installing someone
else's recipe" in the README before running a downloaded one.

## The three entry points (folder name = recipe)
```
./download.sh <hf-id>          # weights → models/<hf-id>
./build-and-copy.sh <folder>   # build recipes/<folder>/Dockerfile → mbx-<folder>, copy image to cluster nodes
./run.sh <folder>              # launch (single-node, or TP across the cluster)
```
Scripts self-bootstrap `.venv` — never activate anything by hand. To stop a box:
`.venv/bin/python -m runner.cli down`.

> Note: `run.sh` drives the default `mode: vllm` (build/`docker run` from the recipe yaml — the path this whole
> doc describes). The runner *also* contains two other, **un-scripted** modes — `mode: recipe` (delegate to a
> downloaded pack's `run-recipe.sh`) and `mode: attach` (front an already-running model) — plus a
> `recipes add/list` pack notion. There is no `run-recipe.sh` in this repo; ignore those paths unless you're
> deliberately building them out, and don't confuse a "recipe pack" with our `recipes/<name>/` folders.

## Adding a recipe (agent steps)
1. `recipes/<name>/myllmbox.yaml` (+ optional `Dockerfile` → `mbx-<name>`, + optional `README.md`).
2. weights via `./download.sh`, referenced as a local `/models/...` path (offline) — **must exist at the same
   absolute path on every cluster node**.
3. if it has a `Dockerfile`: `./build-and-copy.sh <name>` (builds on the head, copies to workers).
4. `./run.sh <name>`; verify with a real completion, not just that it "came up".

## The cluster — hard rules (this hardware bit us; don't relearn it)
The physical boxes live in **`cluster.yaml`** (gitignored, per-machine) as named boxes — each with its ssh
`host`, cross-node `interconnect` address, `iface`, optional RDMA `ib_hca`, and `ssh_user`. Recipes reference box
NAMES (`cluster.boxes: [box1, box2]`), never IPs, so they stay portable. Provision boxes with `cluster/setup.sh`.
Our dev cluster is 2× DGX Spark (GB10 / SM121) with a dedicated ConnectX interconnect; look up the actual
addresses in `cluster.yaml`.

- **The management LAN blocks arbitrary TCP ports** — only ssh. **All** cross-node traffic (NCCL *and* vLLM's
  gloo/message-queue) must go over the interconnect. The runner already pins this
  (`NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA`, `GLOO_SOCKET_IFNAME`, per-node `VLLM_HOST_IP` = the node's interconnect IP).
  If you see a hang right after `backend=nccl` with `gloo ... Connection timed out` to a management-LAN addr,
  that pin is the fix.
- **`cluster.boxes`** (names → `cluster.yaml`; legacy `cluster.nodes` = raw interconnect IPs still works); box 0
  = head (this box, runs the API), the rest join `--headless`. TP = box count is set automatically.
- **The repos on the boxes are SEPARATE copies** (not a shared mount). Files you write via the editor land on the
  head only. That's why `build-and-copy` ships the *image* to workers and `run.sh` rsyncs the *weights* — workers
  only need the image + the weights at the right path, never the recipe folder.
- **The editor mount**: the editor's `spark-vllm-docker` path IS the head box's `~/spark-vllm-docker`. Run
  commands on the boxes via `ssh <host>` (hosts are in `cluster.yaml`); ssh over the interconnect IPs also works.

## Operational rules (memory is the constraint)
- Each Spark has ~119G usable unified memory. A 304B model serve holds ~100G/node; a source compile peaks
  ~20–30G. **They don't both fit** → it's **serve XOR build** on the same box. Throttle a coexisting build with
  `BUILD_JOBS=6 ./build-and-copy.sh …`, or take the serve down first.
- **Never take a serving box down without asking.** UMA over-commit has needed a hard power cycle; a running
  serve is hard-won. Keep `gpu_memory_utilization` ≤ ~0.83.
- Prefer building **our own** image from a recipe `Dockerfile` (source, pinned vLLM ref) over pulling someone's
  prebuilt (anemll/eugr/etc.) — those are references at most. Our SM121 build lives in each recipe's `Dockerfile`.

## Verify, don't assume
`wait_healthy` proves the *local* proxy serves — it does NOT prove the tunnel connected (check
`.mbx/cloudflared.log`) or that generation is correct. Always hit `/v1/chat/completions` with a real prompt
(directly on `127.0.0.1:8000`, or via the proxy `:8011` with `Authorization: Bearer $BINDING_TOKEN`).
