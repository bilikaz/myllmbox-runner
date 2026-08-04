"""Optional dashboard container. The recipe's `dashboard.image` is ANY web UI (sparkDash/mia, a "robot", your
own) — the runner is dashboard-agnostic: it just `docker run`s it on the HEAD, bound to localhost:`port`. The
keepalive proxy then fronts every non-/v1 path with it, gated by DASHBOARD_PASSWORD (see runner/proxy.py). So
the pattern is fixed; what loads is the recipe's taste."""
from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger("mbx.dashboard")
CONTAINER = "mbx-dashboard"


def enabled(cfg: dict[str, Any]) -> bool:
    return bool((cfg.get("dashboard") or {}).get("image"))


def port(cfg: dict[str, Any]) -> int:
    return int((cfg.get("dashboard") or {}).get("port") or 5555)


def start(cfg: dict[str, Any]) -> None:
    d = cfg["dashboard"]
    p = port(cfg)
    stop()  # clear a stale one
    # localhost-only publish: the dashboard is reachable ONLY through the proxy (which password-gates it),
    # never directly. env/mounts are the recipe's job (e.g. a UI that ssh's to boxes mounts ~/.ssh + its config).
    args = ["docker", "run", "-d", "--name", CONTAINER, "--restart", "unless-stopped",
            "-p", f"127.0.0.1:{p}:{p}"]
    for k, v in (d.get("env") or {}).items():
        args += ["-e", f"{k}={v}"]
    for m in d.get("mounts") or []:
        host, _, dest = str(m).partition(":")
        args += ["-v", f"{Path(host).expanduser().resolve()}:{dest or host}"]
    args += [d["image"]]
    cmd = d.get("command")
    if cmd:
        args += shlex.split(cmd) if isinstance(cmd, str) else [str(c) for c in cmd]
    log.info("dashboard: %s on 127.0.0.1:%s (fronted by the proxy for non-/v1 paths)", d["image"], p)
    subprocess.run(args, check=True)


def stop() -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER], check=False, capture_output=True)
