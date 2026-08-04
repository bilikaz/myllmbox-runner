"""Config: CLI > env > YAML > defaults. Secrets (TUNNEL_TOKEN, BINDING_TOKEN) come from .env/env only.

TUNNEL_TOKEN is required — it is the whole binding (the platform provisioned the tunnel + DNS at token
mint; the runner just connects). BINDING_TOKEN is OPTIONAL by decision: absent means the user chose an
all-public box, generation included — say so loudly, never silently.
"""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

log = logging.getLogger("mbx.config")

DEFAULTS: dict[str, Any] = {
    # "server": how to run the served container. vLLM is just the default mode (no `command`); a `command`
    # runs any server (image/video/etc). Named `server` — not `vllm` — because it's not vLLM-specific.
    "server": {
        "image": "vllm/vllm-openai:latest",
        "model": "",
        "port": 8000,   # universal: the container's serve port (proxy → 127.0.0.1:<port>). vLLM + uvicorn both use it.
        # vLLM-mode CLI flags (name: value → --name value · true → --name · false/null → skip). Named `vllm`
        # because they ARE vLLM params (gpu-memory-utilization, tensor-parallel-size, kv-cache-dtype, …) —
        # only read in vLLM mode (no `command`). Generic servers use `command`/`uvicorn` instead.
        "vllm": {},
        "env": {},
        "runtime": "",
        "devices": "all",
        # override the image's default ENTRYPOINT (e.g. a dspark bootstrap) so the runner drives the launch
        # itself — "vllm" → `vllm serve <model> <flags>`. "" keeps the image's own entrypoint.
        "entrypoint": "",
        # Weights — everything under ./models (gitignored), downloaded once and reused every boot:
        #  - models_dir → mounted /models: drop pre-downloaded model FOLDERS here and serve a local path
        #    like model: /models/org/name (this is how the spark recipes use ~/models).
        #  - cache_dir  → the container's HF hub cache: an HF-id `model:` downloads here ONCE and is
        #    reused on every restart. This is what PINS where HF puts weights. "" opts out (re-downloads).
        "models_dir": "models",
        "cache_dir": "models/.hf-cache",
        "mounts": [],  # extra host:container binds, docker -v syntax
        # uvicorn convenience: a structured way to say "generic FastAPI server" instead of a raw `command`.
        # {app: server:app, host: 0.0.0.0} → the runner builds `uvicorn <app> --host <host> --port <port>`
        # using server.port (default 8000, same as vLLM → the proxy front is identical for every box).
        # An explicit uvicorn.port overrides server.port. Non-uvicorn servers still use `command`.
        "uvicorn": {},
        # GENERIC server mode: run the image's OWN server with this command instead of `vllm serve`, and
        # inject NO vLLM flags. "" = vLLM mode (default). Any OpenAI-compatible server works — e.g. the
        # image/video diffusion servers (FastAPI/uvicorn exposing /v1/images/*, /v1/videos/*). It still
        # gets env, mounts, cache, the port mapping, and the same proxy+tunnel front.
        "command": "",
        # Persistent CACHE mount: host dir → /cache in the container, created on node 0 if missing. For
        # on-the-fly quant caches (FLUX torchao NVFP4), compiled kernels, etc. — survives container
        # recreation so only the first boot pays the quantize/compile cost. "" = no cache mount.
        "cache": "",
    },
    "proxy": {"port": 8011, "ping_secs": 30},
    "tunnel": {"binary": "cloudflared", "extra_args": []},
    # model bring-up mode: "vllm" (built-in docker run — laptop smoke) · "recipe" (delegate to a recipe
    # PACK's own entry script — the serious boxes: clusters, NCCL/ConnectX-7, mods) · "attach" (manage
    # NOTHING model-side — proxy+tunnel onto whatever already serves upstream)
    "mode": "vllm",
    # recipes.root is the designated place people download packs into: one folder per pack (a git clone
    # of anyone's script collection). A recipe reference is "<pack>/<recipe>", resolved against the root;
    # the pack's run-recipe.sh is the universal entry contract.
    "recipes": {"root": "recipes"},
    "recipe": {"file": "", "extra_args": []},
    # ./quantize.sh reads this and drives the shared `quantizer/` image (mbx-quantizer): offline-quantize
    # `source` (BF16) → models/myllmbox/<name>-<format>, a self-contained shareable checkpoint the serve
    # recipe then loads instantly (QUANT_PRELOADED). All params live here (yaml), the quantizer is generic:
    #   source       — the BF16 model dir (container path, /models/...); "" = recipe isn't quantizable
    #   format       — default target: nvfp4 | mxfp8 (overridable with ./quantize.sh --to)
    #   model_type   — diffusion | llm  (how to load/save)
    #   target       — which submodule to quantize (diffusion: transformer; llm: model)
    #   skip_modules — comma-sep substrings kept in BF16 (accuracy-critical layers)
    #   formats      — which targets THIS model supports (quantize.sh rejects a --to outside it); e.g. a
    #                  diffusion model can't do llama.cpp-gguf. [] = allow any the backend knows.
    #   out          — override output dir ("" = models/myllmbox/<name>-<format>)
    "quantize": {"source": "", "format": "nvfp4", "model_type": "diffusion", "target": "transformer",
                 "skip_modules": "", "formats": [], "out": ""},
    # cluster: serve ONE model across N Sparks (TP = len(nodes)). node 0 = this box (master — runs the API
    # server + proxy + tunnel); nodes 1..N-1 are ssh'd and join --headless over NCCL. [] = single-node
    # (solo is just N=1, same launch path). nodes = the high-speed (ConnectX) IPs, reachable by ssh too.
    # Two ways to name the nodes:
    #   boxes: [box1, box2]  → resolved from cluster.yaml (name → host/interconnect/iface/ib_hca/ssh_user).
    #                          PORTABLE — no machine IPs in the recipe (see cluster/setup.sh, cluster.yaml).
    #   nodes: [ip, ...]     → legacy: raw interconnect IPs + cluster-wide nccl_ifname/nccl_ib_hca/ssh_user.
    # Either way load() normalises to per-node lists (nodes/ssh_hosts/ssh_users/ifaces/ib_hcas), node 0 = head.
    "cluster": {"boxes": [], "nodes": [], "master_port": 25000, "nccl_ifname": "", "nccl_ib_hca": "", "ssh_user": ""},
    "upstream_port": 0,  # attach/recipe: where the model already listens; 0 = vllm.port
    # dashboard: OPTIONAL web UI the recipe wants fronted under the SAME public URL. The proxy sends /v1/* to
    # the model and EVERYTHING ELSE to this (see runner/proxy.py), Basic-auth'd by DASHBOARD_PASSWORD (.env).
    # `image` = any container serving a UI on `port` (sparkDash/mia, a "robot", your own) — the runner just runs
    # it on the head; nothing here is dashboard-specific. Empty image = no dashboard (proxy → model only).
    "dashboard": {"image": "", "port": 5555, "command": "", "env": {}, "mounts": []},
    "tunnel_token": "",
    "binding_token": "",
    "dashboard_password": "",
    "public_url": "",
}

# .env / environment → config keys (secrets and the status probe target)
ENV_KEYS = {"TUNNEL_TOKEN": "tunnel_token", "BINDING_TOKEN": "binding_token",
            "DASHBOARD_PASSWORD": "dashboard_password", "PUBLIC_URL": "public_url"}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out


def _resolve_cluster(cfg: dict[str, Any], base_dir: str | Path = ".") -> None:
    """Normalise cfg['cluster'] to per-node lists the runner consumes: nodes (interconnect, for NCCL/gloo/mq),
    ssh_hosts + ssh_users (the control plane), ifaces (NCCL_SOCKET_IFNAME), ib_hcas (NCCL_IB_HCA). node 0 = head.

    `boxes: [name, ...]` resolves each name from cluster.yaml (portable recipe). Legacy `nodes: [ip, ...]` keeps
    working: interconnect==ssh over the same IP, cluster-wide iface/hca/ssh_user broadcast to every node."""
    c = cfg.get("cluster") or {}
    boxes = c.get("boxes") or []
    if boxes:
        cy = Path(base_dir) / "cluster.yaml"
        if not cy.exists():
            raise SystemExit("cluster.boxes is set but there's no cluster.yaml — run cluster/setup.sh "
                             "(or cp cluster.yaml.example cluster.yaml and fill it in)")
        spec = (yaml.safe_load(cy.read_text()) or {}).get("boxes") or {}
        rs = []
        for name in boxes:
            b = spec.get(name)
            if not b:
                raise SystemExit(f"cluster.yaml has no box '{name}' (defined: {', '.join(spec) or 'none'})")
            rs.append(b)
        c["nodes"]     = [b.get("interconnect") or b["host"] for b in rs]   # data plane (falls back to host)
        c["ssh_hosts"] = [b["host"] for b in rs]                            # control plane (ssh)
        c["ssh_users"] = [b.get("ssh_user") or "" for b in rs]
        c["ifaces"]    = [b.get("iface") or "" for b in rs]
        c["ib_hcas"]   = [b.get("ib_hca") or "" for b in rs]
    else:
        n = len(c.get("nodes") or [])
        c["ssh_hosts"] = list(c.get("nodes") or [])                         # legacy ssh'd over the interconnect IP
        c["ssh_users"] = [c.get("ssh_user") or ""] * n
        c["ifaces"]    = [c.get("nccl_ifname") or ""] * n
        c["ib_hcas"]   = [c.get("nccl_ib_hca") or ""] * n
    cfg["cluster"] = c


def load(cli: dict[str, Any] | None = None, env: dict[str, str] | None = None, yaml_path: str | Path | None = None) -> dict[str, Any]:
    """Merge the four layers. `env` defaults to os.environ overlaid on .env in the cwd."""
    if env is None:
        env = {**dotenv_values(".env"), **os.environ}  # real env wins over the file

    cfg = copy.deepcopy(DEFAULTS)
    if yaml_path and Path(yaml_path).exists():
        y = yaml.safe_load(Path(yaml_path).read_text()) or {}
        if "vllm" in y and "server" not in y:  # back-compat: the block used to be named `vllm:`
            y["server"] = y.pop("vllm")
        srv = y.get("server")                  # back-compat: the flags sub-block used to be `extra_args:`
        if isinstance(srv, dict) and "extra_args" in srv and "vllm" not in srv:
            srv["vllm"] = srv.pop("extra_args")
        cfg = deep_merge(cfg, y)
    # one serve port everywhere: an explicit uvicorn.port wins, else server.port (default 8000).
    u = cfg["server"].get("uvicorn")
    if isinstance(u, dict) and u.get("port"):
        cfg["server"]["port"] = u["port"]
    cfg = deep_merge(cfg, {dest: env[src] for src, dest in ENV_KEYS.items() if env.get(src)})
    if cli:
        cfg = deep_merge(cfg, {k: v for k, v in cli.items() if v is not None})

    if not cfg["tunnel_token"]:
        raise SystemExit("TUNNEL_TOKEN is required — copy it from the console's Box setup reveal into .env")
    if not cfg["binding_token"]:
        # the user's explicit choice, but it must never be a silent one
        log.warning("no BINDING_TOKEN — the box will be FULLY PUBLIC, generation included")
        cfg["public_mode"] = True
    else:
        cfg["public_mode"] = False
    _resolve_cluster(cfg)   # boxes/nodes → per-node lists (nodes/ssh_hosts/ssh_users/ifaces/ib_hcas)
    return cfg
