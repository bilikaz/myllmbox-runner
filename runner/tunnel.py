"""cloudflared lifecycle. The tunnel is remotely managed — myllmbox provisioned it (with its ingress and
DNS) when the binding token was minted, so the ONLY thing needed here is `tunnel run --token <…>`."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def spawn(cfg: dict[str, Any], log_path: Path) -> subprocess.Popen:
    binary = shutil.which(cfg["tunnel"]["binary"]) or next((c for c in [os.path.expanduser("~/.local/bin/cloudflared"), "/usr/local/bin/cloudflared"] if os.path.exists(c)), cfg["tunnel"]["binary"])
    args = [binary, "tunnel", "--no-autoupdate", "run", "--token", cfg["tunnel_token"]]
    args += [str(a) for a in cfg["tunnel"].get("extra_args") or []]
    logf = open(log_path, "ab")
    # start_new_session: own process group, so a Ctrl-C on run.sh (e.g. while watching the load stream) does
    # NOT kill the tunnel — it's supervised via its pidfile, not the terminal. `down` stops it.
    return subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
