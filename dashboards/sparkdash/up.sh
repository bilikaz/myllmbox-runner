#!/usr/bin/env bash
# Bring sparkDash (mia) up on $PORT, monitoring THIS recipe's boxes ($MBX_BOXES). Run by the runner on the head.
#
# ⚠ BEST-EFFORT / UNVERIFIED: sparkDash is a Node/React app built with its own docker compose, and I couldn't
#   build it or confirm its config schema on this hardware. Treat the build + sparks.json shaping below as a
#   STARTING POINT to validate against https://github.com/MiaAI-Lab/sparkDash (its config/sparks.json fields,
#   how it takes ssh creds, and whether `docker compose` vs a single image is the right run). The runner
#   contract is solid; only these sparkDash-specific lines need confirming when you actually stand mia up.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="mbx-dash-sparkdash"
CONTAINER="mbx-dashboard"        # keep this name → default down.sh + no collision with the model container
PORT="${PORT:-5555}"
SRC="$HERE/src"                  # sparkDash checkout (gitignored)

# 1. get + build sparkDash once
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  [ -d "$SRC/.git" ] || git clone --depth 1 https://github.com/MiaAI-Lab/sparkDash "$SRC"
  docker build -t "$IMAGE" "$SRC"        # ⚠ confirm sparkDash ships a root Dockerfile that serves on 5555
fi

# 2. generate sparks.json from the recipe's boxes (MBX_BOXES). ⚠ field names are a GUESS — verify vs sparkDash.
CFG="$HERE/config"; mkdir -p "$CFG"
python3 - "$CFG/sparks.json" <<'PY'
import json, os, sys
boxes = json.loads(os.environ.get("MBX_BOXES", "[]"))
out = [{"name": b["name"], "ip": b["host"], "cxIp": b.get("interconnect", ""),
        "sshUser": b.get("ssh_user", "")} for b in boxes]
json.dump(out, open(sys.argv[1], "w"), indent=2)
print(f"wrote {len(out)} box(es) to {sys.argv[1]}", file=sys.stderr)
PY

# 3. run it, localhost-only (the proxy is its only door). Mount the config + ssh key (it ssh's to boxes).
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" --restart unless-stopped \
  -p "127.0.0.1:${PORT}:5555" \
  -v "$CFG:/app/config" \
  -v "$HOME/.ssh:/root/.ssh:ro" \
  "$IMAGE"
echo "sparkDash up on 127.0.0.1:${PORT} (fronted by the proxy)"
