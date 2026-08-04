"""Orchestration: up = vLLM container → keepalive proxy → cloudflared, each detached with a pidfile under
./.mbx/; down tears them back in reverse, idempotently. State on disk, not in memory — `up`, `status` and
`down` are separate invocations."""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from . import tunnel, vllm

log = logging.getLogger("mbx.supervisor")

# one state dir per box — run several boxes on one host by giving each its own working dir or MBX_STATE
STATE = Path(os.environ.get("MBX_STATE", ".mbx"))


def _pidfile(name: str) -> Path:
    return STATE / f"{name}.pid"


def _alive(name: str) -> bool:
    pf = _pidfile(name)
    if not pf.exists():
        return False
    try:
        os.kill(int(pf.read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def _spawn_proxy(cfg: dict[str, Any]) -> subprocess.Popen:
    env = {
        **os.environ,
        "MBX_UPSTREAM": f"http://127.0.0.1:{_upstream_port(cfg)}",
        "MBX_PROXY_PORT": str(cfg["proxy"]["port"]),
        "MBX_PING_SECS": str(cfg["proxy"]["ping_secs"]),
        "BINDING_TOKEN": cfg["binding_token"],
    }
    logf = open(STATE / "proxy.log", "ab")
    # start_new_session: own process group so Ctrl-C on run.sh (while watching the load stream) doesn't kill
    # the proxy — it's supervised via its pidfile, not the terminal (matches this module's "detached" contract).
    return subprocess.Popen([sys.executable, "-m", "runner.proxy"], env=env, stdout=logf,
                            stderr=subprocess.STDOUT, start_new_session=True)


def _upstream_port(cfg: dict[str, Any]) -> int:
    return int(cfg.get("upstream_port") or cfg["server"]["port"])


def resolve_recipe(cfg: dict[str, Any]) -> tuple[Path, str]:
    """"<pack>/<recipe>" → (the pack's folder under recipes.root, the recipe reference the pack's own
    entry script understands). The universal contract is only: a pack is a folder with run-recipe.sh."""
    ref = cfg["recipe"]["file"]
    if "/" not in ref:
        raise SystemExit(f'recipe reference must be "<pack>/<recipe>", got: {ref}')
    pack, _, rest = ref.partition("/")
    pack_dir = Path(cfg["recipes"]["root"]).expanduser() / pack
    if not (pack_dir / "run-recipe.sh").exists():
        raise SystemExit(f"pack '{pack}' has no run-recipe.sh under {pack_dir} — download it there first (mbx-runner recipes add <git-url> {pack})")
    return pack_dir, rest


def up(cfg: dict[str, Any]) -> None:
    STATE.mkdir(exist_ok=True)
    if cfg.get("public_mode"):
        log.warning("ALL-PUBLIC box: no BINDING_TOKEN set — anyone can use the generation endpoints")
    mode = cfg.get("mode", "vllm")
    if mode == "vllm":
        vllm.start(cfg)
        ns = vllm.nodes(cfg)
        if len(ns) > 1:  # remember the RESOLVED cluster (per-node ssh_hosts/users) so `down` stops the workers too
            (STATE / "cluster.json").write_text(json.dumps(cfg.get("cluster") or {}))
    elif mode == "recipe":
        # the model belongs to a recipe PACK (a downloaded script collection — clusters, NCCL/ConnectX-7,
        # Ray, mods are ITS concern); the reference "<pack>/<recipe>" resolves against recipes.root
        pack_dir, recipe_ref = resolve_recipe(cfg)
        script = pack_dir / "run-recipe.sh"
        args = [str(script), recipe_ref, *[str(a) for a in cfg["recipe"].get("extra_args") or []]]
        if "-d" not in args:
            args.append("-d")  # the runner supervises the proxy+tunnel; the recipe detaches
        log.info("bringing the model up via pack %s: %s", pack_dir.name, " ".join(args[1:]))
        subprocess.run(args, cwd=pack_dir, check=True)
    elif mode == "attach":
        log.info("attach mode — expecting a model already serving on 127.0.0.1:%s", _upstream_port(cfg))
    else:
        raise SystemExit(f"unknown mode: {mode}")
    proxy = _spawn_proxy(cfg)
    _pidfile("proxy").write_text(str(proxy.pid))
    tun = tunnel.spawn(cfg, STATE / "cloudflared.log")
    _pidfile("tunnel").write_text(str(tun.pid))
    wait_healthy(cfg)
    log.info("box is up — proxy :%s, tunnel connected (see .mbx/*.log)", cfg["proxy"]["port"])


def down() -> None:
    # reverse order; each step idempotent — a half-up box tears down cleanly
    for name in ("tunnel", "proxy"):
        pf = _pidfile(name)
        if pf.exists():
            try:
                os.kill(int(pf.read_text()), signal.SIGTERM)
                log.info("stopped %s", name)
            except (OSError, ValueError):
                pass
            pf.unlink(missing_ok=True)
    # rebuild the resolved cluster (if any) so we stop the workers too
    cfg: dict[str, Any] = {}
    cj = STATE / "cluster.json"
    nf = STATE / "nodes"                                        # legacy state from an older `up`
    if cj.exists():
        cfg = {"cluster": json.loads(cj.read_text())}
        cj.unlink(missing_ok=True)
    elif nf.exists():
        su = STATE / "ssh_user"
        cfg = {"cluster": {"nodes": [n for n in nf.read_text().split(",") if n],
                           "ssh_user": su.read_text() if su.exists() else ""}}
        nf.unlink(missing_ok=True)
        (STATE / "ssh_user").unlink(missing_ok=True)
    vllm.stop(cfg)  # local container + any cluster workers (no-op when the model was never ours)
    log.info("box is down")


def _container_running() -> bool:
    out = subprocess.run(["docker", "ps", "-q", "-f", f"name=^{vllm.CONTAINER}$"],
                         capture_output=True, text=True)
    return bool(out.stdout.strip())


def wait_healthy(cfg: dict[str, Any], timeout_s: int = 1800) -> None:
    """The one health that matters: the model card answers through the proxy. vLLM can take many minutes
    to load weights — a 304B MoE across the cluster (weights + spec-decode draft + cudagraph capture) can
    need 15+ min — so poll patiently (30 min), fail loudly. Timing out here does NOT kill the container;
    it just stops us waiting — the model may still be finishing init (check `docker logs mbx-vllm`).

    While we wait, STREAM the head container's `docker logs -f` to the terminal so that long silent load
    is visible (not "looks stuck") — we tail in the background and poll health in parallel; first 200 stops
    the tail and proceeds to proxy+tunnel. If the container dies during load, the tail ends and we fail fast
    with the logs already on screen instead of blocking for the full timeout. Ctrl-C leaves the box running."""
    url = f"http://127.0.0.1:{cfg['proxy']['port']}/v1/models"
    deadline = time.monotonic() + timeout_s
    tail = None
    if _container_running():  # local head container (vllm mode); recipe/attach modes just poll silently
        print(f"\n── loading model · streaming `docker logs -f {vllm.CONTAINER}` until healthy. "
              f"Ctrl-C only detaches this view — model, proxy & tunnel keep running "
              f"(re-attach: docker logs -f {vllm.CONTAINER} · full stop: .venv/bin/python -m runner.cli down) ──",
              flush=True)
        tail = subprocess.Popen(["docker", "logs", "-f", "--tail", "40", vllm.CONTAINER])
    try:
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=5) as res:
                    if res.status == 200:
                        if tail is not None:
                            print(f"\n── model healthy — proxy :{cfg['proxy']['port']} serving ──", flush=True)
                        return
            except OSError:
                pass
            # `docker logs -f` exits when the container stops → the model died during load; don't wait 30 min.
            if tail is not None and tail.poll() is not None and not _container_running():
                raise SystemExit(f"model container {vllm.CONTAINER} exited during load — see the logs above "
                                 f"(or `docker logs {vllm.CONTAINER}`)")
            time.sleep(3)
        raise SystemExit(f"box did not become healthy within {timeout_s}s — check .mbx/*.log and `docker logs {vllm.CONTAINER}`")
    finally:
        if tail is not None and tail.poll() is None:
            tail.terminate()
            try:
                tail.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tail.kill()


def status(cfg: dict[str, Any]) -> dict[str, Any]:
    st: dict[str, Any] = {
        "vllm": vllm.running(),
        "proxy": _alive("proxy"),
        "tunnel": _alive("tunnel"),
        "local_model_card": False,
        "public_model_card": None,
    }
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{cfg['proxy']['port']}/v1/models", timeout=5) as res:
            st["local_model_card"] = res.status == 200
    except OSError:
        pass
    if cfg.get("public_url"):
        try:
            with urllib.request.urlopen(f"{cfg['public_url'].rstrip('/')}/v1/models", timeout=8) as res:
                st["public_model_card"] = res.status == 200
        except OSError:
            st["public_model_card"] = False
    return st
