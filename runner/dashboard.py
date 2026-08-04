"""Optional dashboard. A recipe's `dashboard: <name>` points at dashboards/<name>/, which owns its lifecycle:

    dashboards/<name>/
      dashboard.yaml   # just `port:` (what the proxy forwards to) + a description
      up.sh            # does WHATEVER the UI needs: build, generate config, docker run a web container on $PORT
      down.sh          # tears it ALL down — the head container AND anything up.sh started on other boxes

The runner is UI-agnostic: it runs up.sh / down.sh and passes context in the env (PORT, DASHBOARD_PASSWORD,
MBX_BOXES = this recipe's box set as JSON — the script shapes it, e.g. mia → sparks.json). The active dashboard
name is marked in .mbx so `down` knows which down.sh to call. The proxy fronts 127.0.0.1:port for non-/v1 paths.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger("mbx.dashboard")
DIR = Path("dashboards")


def enabled(cfg: dict[str, Any]) -> bool:
    return bool((cfg.get("dashboard") or {}).get("name"))


def name(cfg: dict[str, Any]) -> str:
    return (cfg.get("dashboard") or {}).get("name") or ""


def port(cfg: dict[str, Any]) -> int:
    return int((cfg.get("dashboard") or {}).get("port") or 5555)


def boxes(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """This RECIPE's box set (box 0 = head), resolved from cluster.yaml — 1 for single-node, N for a cluster.
    Handed to up.sh as MBX_BOXES; a UI that monitors boxes (mia) turns it into its own config, others ignore it."""
    c = cfg.get("cluster") or {}
    names, hosts = c.get("boxes") or [], c.get("ssh_hosts") or []
    ics, users = c.get("nodes") or [], c.get("ssh_users") or []
    if hosts:
        return [{"name": names[i] if i < len(names) else f"box{i+1}",
                 "role": "head" if i == 0 else "worker",
                 "host": hosts[i],
                 "interconnect": ics[i] if i < len(ics) else "",
                 "ssh_user": users[i] if i < len(users) else ""} for i in range(len(hosts))]
    return [{"name": "local", "role": "head", "host": "127.0.0.1", "interconnect": "", "ssh_user": ""}]


def _env(cfg: dict[str, Any]) -> dict[str, str]:
    e = dict(os.environ)
    e["PORT"] = str(port(cfg))
    e["MBX_BOXES"] = json.dumps(boxes(cfg))
    e["DASHBOARD_PASSWORD"] = cfg.get("dashboard_password", "")
    for k, v in ((cfg.get("dashboard") or {}).get("env") or {}).items():
        e[k] = str(v)
    return e


def up(cfg: dict[str, Any]) -> None:
    n = name(cfg)
    script = DIR / n / "up.sh"
    if not script.exists():
        raise SystemExit(f"dashboard '{n}': missing {script}")
    log.info("dashboard %s → up on 127.0.0.1:%s", n, port(cfg))
    subprocess.run(["bash", str(script)], env=_env(cfg), check=True)


def down(dash_name: str) -> None:
    """Tear down a dashboard by name (from the .mbx marker). Prefer its down.sh (it knows about any cross-box
    bits it started); fall back to removing the conventional container."""
    if not dash_name:
        return
    script = DIR / dash_name / "down.sh"
    if script.exists():
        log.info("dashboard %s → down", dash_name)
        subprocess.run(["bash", str(script)], check=False)
    else:
        subprocess.run(["docker", "rm", "-f", "mbx-dashboard"], check=False, capture_output=True)
