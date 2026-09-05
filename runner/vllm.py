"""The vLLM container: `docker run` from the recipe. Single-node OR multi-node — one model spread across
N Sparks (TP = number of nodes). node 0 = this box (master, runs the API server); nodes 1..N-1 are ssh'd
and join --headless over NCCL. Published on 127.0.0.1 only; the keepalive proxy is the public face."""
from __future__ import annotations

import logging
import shlex
import time
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger("mbx.vllm")

CONTAINER = "mbx-vllm"


def nodes(cfg: dict[str, Any]) -> list[str]:
    return list((cfg.get("cluster") or {}).get("nodes") or [])


def _ssh(cfg: dict[str, Any], rank: int) -> list[str]:
    """ssh to a node's CONTROL plane by rank. config._resolve_cluster fills ssh_hosts/ssh_users; fall back to
    the interconnect node + cluster-wide ssh_user (legacy / a `down`-reconstructed cfg that only has nodes)."""
    c = cfg.get("cluster") or {}
    hosts = c.get("ssh_hosts") or c.get("nodes") or []
    users = c.get("ssh_users") or []
    host = hosts[rank] if rank < len(hosts) else ""
    user = users[rank] if rank < len(users) else (c.get("ssh_user") or "")
    return ["ssh", "-o", "BatchMode=yes", f"{user}@{host}" if user else host]


def run_args(cfg: dict[str, Any], node_rank: int = 0, nnodes: int = 1, master_addr: str = "") -> list[str]:
    v = cfg["server"]
    c = cfg.get("cluster") or {}
    multi = nnodes > 1
    # SYS_PTRACE lets `py-spy dump`/gdb attach inside the container to diagnose a stalled load or a
    # wedged worker (the runner drops all caps otherwise). Harmless for serving; invaluable when a
    # cross-node load livelocks with one core spinning and no log.
    args = ["docker", "run", "-d", "--name", CONTAINER, "--gpus", str(v["devices"]), "--ipc=host", "--cap-add", "SYS_PTRACE"]
    # multi-node NCCL needs host networking (nodes talk over the ConnectX link); single-node maps a port.
    args += ["--network", "host"] if multi else ["-p", f"127.0.0.1:{v['port']}:{v['port']}"]
    if multi:
        # RDMA for real: NCCL_IB_HCA (below) names the ConnectX verbs device, but a plain container can't
        # OPEN it — no /dev/infiniband, memlock capped at 8 MB, no IPC_LOCK — so NCCL silently fell back to
        # TCP sockets over the same link (found 2026-09-04: iface tx 82 MB/s, RDMA port counters flat at 0,
        # ~2x slower TP step than the community launchers, which all pass exactly these three flags).
        args += ["--device", "/dev/infiniband", "--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1:-1"]
    # optional CPU pinning (server.cpuset, e.g. "5-9,15-19" = GB10 performance cores — MiaAI's dual-Spark
    # recipe pins these; keeps the scheduler/tokenizer off the efficiency cores).
    if v.get("cpuset"):
        args += ["--cpuset-cpus", str(v["cpuset"])]
    if v.get("runtime"):
        args += ["--runtime", str(v["runtime"])]
    local_model = str(v["model"]).startswith("/")
    # A local /model path is served offline (HF_HUB_OFFLINE below) — the HF hub cache is never used, so
    # don't create/mount it (it only made an empty models/.hf-cache). Cache is for HF-id models only.
    if v.get("cache_dir") and not local_model:
        cache = Path(v["cache_dir"]).expanduser().resolve()
        if node_rank == 0:
            cache.mkdir(parents=True, exist_ok=True)   # only create locally; workers must already have it
        args += ["-v", f"{cache}:/root/.cache/huggingface", "-e", "HF_HOME=/root/.cache/huggingface"]
    if v.get("models_dir"):
        models = Path(v["models_dir"]).expanduser().resolve()
        if node_rank == 0:
            models.mkdir(parents=True, exist_ok=True)
        args += ["-v", f"{models}:/models"]
    for m in v.get("mounts") or []:
        host, _, dest = str(m).partition(":")
        args += ["-v", f"{Path(host).expanduser().resolve()}:{dest or host}"]
    # persistent cache mount → /cache (on-the-fly quant, compiled kernels): only the first boot pays the cost
    if v.get("cache"):
        cdir = Path(v["cache"]).expanduser().resolve()
        if node_rank == 0:
            cdir.mkdir(parents=True, exist_ok=True)   # create locally; workers must already have it
        args += ["-v", f"{cdir}:/cache"]
    for k, val in (v.get("env") or {}).items():
        args += ["-e", f"{k}={val}"]
    if multi:  # ALL cross-node traffic goes over the interconnect (the LAN blocks arbitrary TCP ports).
        cnodes = c.get("nodes") or []
        node_ip = cnodes[node_rank] if node_rank < len(cnodes) else ""
        ifaces, hcas = c.get("ifaces") or [], c.get("ib_hcas") or []       # per-node (config._resolve_cluster)
        iface = ifaces[node_rank] if node_rank < len(ifaces) else c.get("nccl_ifname")   # legacy fallback
        hca = hcas[node_rank] if node_rank < len(hcas) else c.get("nccl_ib_hca")
        for k, val in (("NCCL_SOCKET_IFNAME", iface),
                       ("NCCL_IB_HCA", hca),
                       ("NCCL_IB_DISABLE", "0"),
                       ("GLOO_SOCKET_IFNAME", iface),                 # gloo on the same interconnect iface as NCCL
                       ("VLLM_HOST_IP", node_ip),                     # advertise THIS node's interconnect IP for the mq
                       ("SGLANG_HOST_IP", node_ip),                   # sglang's equivalent (utils/network.py get_ip) — without it the
                                                                      # shm_broadcast mq advertises the default-route (management-LAN)
                                                                      # IP whose ports are blocked → wait_until_ready hangs forever
                       # engine-agnostic cluster facts, for generic-command recipes (a wrapped engine — sglang
                       # etc. — needs its own --nnodes/--node-rank/--dist-init-addr; vLLM gets them as CLI flags
                       # below, a server.command can read these instead: `bash -c '... --node-rank $MBX_NODE_RANK'`)
                       ("MBX_NNODES", str(nnodes)),
                       ("MBX_NODE_RANK", str(node_rank)),
                       ("MBX_MASTER_ADDR", str(master_addr)),
                       ("MBX_MASTER_PORT", str(c.get("master_port", 25000)))):
            if val:
                args += ["-e", f"{k}={val}"]
    if str(v["model"]).startswith("/"):
        args += ["-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1"]
    ep = str(v.get("entrypoint") or "")
    # GENERIC server mode: run the image's own server command, inject NO vLLM flags. Everything above
    # (env, mounts, cache, the -p port map) still applies — the proxy/tunnel front it exactly the same.
    cmd = v.get("command")
    u = v.get("uvicorn")
    if not cmd and isinstance(u, dict) and u:  # uvicorn convenience → build the command (port = server.port)
        cmd = f"uvicorn {u.get('app', 'server:app')} --host {u.get('host', '0.0.0.0')} --port {v['port']}"
    if cmd:
        if ep:
            args += ["--entrypoint", ep]
        args += [v["image"]]
        args += shlex.split(cmd) if isinstance(cmd, str) else [str(c) for c in cmd]
        return args
    # SGLANG mode (a non-empty server.sglang dict): same box contract as vLLM mode — model from server.model,
    # API on server.port, TP = node count on a cluster — but the engine is sglang. Flags map like server.vllm.
    sg = v.get("sglang")
    if isinstance(sg, dict) and sg:
        launcher = ["python3", "-m", "sglang.launch_server"]
        if ep:
            args += ["--entrypoint", ep]
            if ep in ("python", "python3"):   # entrypoint already IS the interpreter — don't double it
                launcher = ["-m", "sglang.launch_server"]
        args += [v["image"], *launcher, "--model-path", v["model"],
                 "--host", "0.0.0.0", "--port", str(v["port"])]   # 0.0.0.0: the -p 127.0.0.1 map needs it (sglang defaults to 127.0.0.1)
        if multi:  # sglang's native multi-node: every rank runs the same launcher, rank 0 serves the API
            args += ["--dist-init-addr", f"{master_addr}:{c.get('master_port', 25000)}",
                     "--nnodes", str(nnodes), "--node-rank", str(node_rank)]
            if not any(k in sg for k in ("tp", "tp-size", "tensor-parallel-size")):
                args += ["--tp", str(nnodes)]   # Sparks are 1 GPU each → TP = node count (same rule as vLLM mode)
        for k, val in sg.items():
            if val is True:
                args += [f"--{k}"]
            elif val not in (False, None):
                args += [f"--{k}", str(val)]
        return args
    if ep:
        args += ["--entrypoint", ep]   # override the image's default (e.g. a bootstrap) → run vllm ourselves
    # official vllm-openai ENTRYPOINT is `vllm serve` — pass model + flags only. When we override to "vllm"
    # the `serve` subcommand is implied, so add it back.
    args += [v["image"]]
    if ep == "vllm":
        args += ["serve"]
    args += [v["model"], "--port", str(v["port"])]  # gpu-memory-utilization & friends come from server.vllm
    if multi:
        args += ["--nnodes", str(nnodes), "--node-rank", str(node_rank),
                 "--master-addr", str(master_addr), "--master-port", str(c.get("master_port", 25000))]
        # cluster owns the parallelism: world_size must == nnodes × gpus/node. Sparks are 1 GPU each, so
        # TP = node count. Set it unless the recipe pinned tensor-parallel-size itself.
        ea_now = v.get("vllm") or {}
        if not (isinstance(ea_now, dict) and "tensor-parallel-size" in ea_now):
            args += ["--tensor-parallel-size", str(nnodes)]
        if node_rank > 0:
            args += ["--headless"]   # workers only join the TP group — no API server
    # vllm flags mapping (server.vllm):  name: value → --name value ·  name: true → --name ·  false/null → skip (list ok too)
    ea = v.get("vllm") or {}
    if isinstance(ea, dict):
        for k, val in ea.items():
            if val is True:
                args += [f"--{k}"]
            elif val not in (False, None):
                args += [f"--{k}", str(val)]
    else:
        args += [str(a) for a in ea]
    return args


def _node_sh(cfg: dict[str, Any], rank: int, script: str) -> str:
    """Run a bash snippet on node <rank> (0 = local, else ssh); return stdout ('' on failure)."""
    cmd = ["bash", "-c", script] if rank == 0 else [*_ssh(cfg, rank), shlex.join(["bash", "-c", script])]
    out = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return out.stdout.strip()


def _prelaunch_memory(cfg: dict[str, Any]) -> None:
    """Unified-memory launch hygiene, same tricks the solo kits use, no root anywhere:
    1. wait until every node has >= server.min_avail_gb GiB AVAILABLE (a just-killed container's GPU pages come back
       over ~30-60 s; launching earlier = phantom CUDA OOM). 0 = skip the wait.
    2. evict the model's shard files from the page cache (`dd iflag=nocache count=0` = POSIX_FADV_DONTNEED) on every
       node — the GPU driver wants FREE pages; a page cache full of shards during load has livelocked the loader."""
    s = cfg["server"]
    ns = nodes(cfg) or ["local"]
    need = int(s.get("min_avail_gb") or 0)
    if need > 0:
        waited = 0
        while True:
            avail = [int(_node_sh(cfg, r, "free -g | awk '/^Mem:/{print $7}'") or 0) for r in range(len(ns))]
            if all(a >= need for a in avail):
                log.info("memory: %s GiB available on nodes — ok (need %d)", "/".join(map(str, avail)), need)
                break
            if waited >= 120:
                raise SystemExit(f"memory: only {'/'.join(map(str, avail))} GiB available (need {need} on every node) after 120 s — "
                                 "another serve still running? (docker ps on each box)")
            if waited == 0:
                log.info("memory: %s GiB available, need %d — waiting for unified memory to come back…", "/".join(map(str, avail)), need)
            time.sleep(5); waited += 5
    model = str(s.get("model") or "")
    if model.startswith("/models/") and s.get("models_dir"):
        host_dir = str(Path(s["models_dir"]).expanduser().resolve() / model[len("/models/"):])
        script = (f'for f in "{host_dir}"/*.safetensors; do [ -f "$f" ] && dd if="$f" iflag=nocache count=0 status=none 2>/dev/null; done; '
                  "awk '/^MemFree/{printf \"%d\", $2/1048576}' /proc/meminfo")
        free_after = [_node_sh(cfg, r, script) or "?" for r in range(len(ns))]
        log.info("memory: evicted %s from the page cache on every node (no root) — MemFree now %s GiB", host_dir, "/".join(free_after))


def start(cfg: dict[str, Any]) -> None:
    s = cfg["server"]
    if not s.get("command") and not s["model"]:
        raise SystemExit("server.model is required (or set server.command for a generic server)")
    ns = nodes(cfg)
    stop(cfg)  # clear stale containers on every node first
    _prelaunch_memory(cfg)
    if len(ns) > 1:
        nn, master = len(ns), ns[0]
        log.info("cluster: %d nodes (TP=%d), master=%s:%s", nn, nn, master, (cfg.get("cluster") or {}).get("master_port"))
        log.info("cluster: head rank 0 (local)")
        subprocess.run(run_args(cfg, 0, nn, master), check=True)
        for rank, node in enumerate(ns[1:], start=1):
            log.info("cluster: worker rank %d on %s (--headless)", rank, node)
            subprocess.run([*_ssh(cfg, rank), shlex.join(run_args(cfg, rank, nn, master))], check=True)
    else:
        s = cfg["server"]
        log.info("starting container (%s · %s)", s["image"], "generic:" + s["command"].split()[0] if s.get("command") else s["model"])
        subprocess.run(run_args(cfg), check=True)


def stop(cfg: dict[str, Any] | None = None) -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER], check=False, capture_output=True)
    for rank in range(1, len(nodes(cfg or {}))):   # also clear the container off every worker
        subprocess.run([*_ssh(cfg or {}, rank), "docker", "rm", "-f", CONTAINER], check=False, capture_output=True)


def running() -> bool:
    out = subprocess.run(["docker", "ps", "-q", "-f", f"name=^{CONTAINER}$"], capture_output=True, text=True)
    return bool(out.stdout.strip())
