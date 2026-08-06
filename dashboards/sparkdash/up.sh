#!/usr/bin/env bash
# Bring sparkDash (mia) up on $PORT for THIS recipe's boxes ($MBX_BOXES). Runtime mirrors sparkDash's own
# src/docker-compose.yml (the authoritative spec): host net + privileged + pid:host + nvidia-smi/NVML mounts.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="mbx-dash-sparkdash"
CONTAINER="mbx-dashboard"        # keep this name → default down.sh + no collision with the model container
PORT="${PORT:-5555}"
DATA="$HERE/.data"               # ALL runtime junk (clone + config) — the one gitignored dir per dashboard
SRC="$DATA/src"                  # sparkDash checkout
CFG="$DATA/config"; mkdir -p "$CFG"

# 1. get + build once
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  [ -d "$SRC/.git" ] || git clone --depth 1 https://github.com/MiaAI-Lab/sparkDash "$SRC"
  docker build -t "$IMAGE" "$SRC"
fi

# 1b. fix ownership of .data. sparkDash runs as ROOT and writes into this bind mount, so files it lands come
#     back root-owned — and the next run (us, non-root) then can't overwrite sparks.json ("PermissionError").
#     chown the whole .data back to us via a throwaway ROOT container (docker is root → no host sudo needed).
#     chown only changes OWNERSHIP — it deletes nothing — so src/ etc. are preserved, just owned by us again.
if [ -d "$DATA" ]; then
  # --entrypoint chown so the image's own ENTRYPOINT (if any) can't swallow the command; busybox is the fallback.
  docker run --rm --entrypoint chown -v "$DATA:/data" "$IMAGE" -R "$(id -u):$(id -g)" /data >/dev/null 2>&1 \
    || docker run --rm -v "$DATA:/data" busybox chown -R "$(id -u):$(id -g)" /data >/dev/null 2>&1 || true
fi

# 2. (RE)generate config/sparks.json from MBX_BOXES EVERY run — the recipe's box SET varies run-to-run (box1+box2
#    now, other boxes next), so a stale list is worse than losing UI tweaks. Skip only when MBX_BOXES is empty
#    (a manual re-run) so we don't wipe it to nothing. box 0 = head+local (LLM served here, metrics via /host
#    mounts); the rest = workers reached over SSH (isLocal:false).
if [ -n "${MBX_BOXES:-}" ] && [ "$MBX_BOXES" != "[]" ]; then
  python3 - "$CFG/sparks.json" <<'PY'
import json, os, sys
boxes = json.loads(os.environ.get("MBX_BOXES", "[]"))
head = boxes[0]["name"] if boxes else None
sparks = [{
    "id": b["name"], "name": b["name"], "lanIp": b["host"], "cx7Ip": b.get("interconnect"),
    "macAddress": None, "detectedMacAddress": None, "isLocal": (i == 0),
    "ssh": {"host": b["host"], "user": b.get("ssh_user", ""), "auth": "key"},
    "llmPorts": [8000], "role": "head" if i == 0 else "worker", "workerNode": i != 0,
    "workerLabel": None, "workerHeadId": None if i == 0 else head, "llmMonitoring": i == 0,
    "disabledDevices": [], "disabledInterfaces": [], "storagePollDisabled": False,
} for i, b in enumerate(boxes)]
json.dump({"sparks": sparks}, open(sys.argv[1], "w"), indent=2)
print(f"regenerated {len(sparks)} box(es)", file=sys.stderr)
PY
fi

# 3. run — mirror src/docker-compose.yml. host net so LLM probes reach the head's 127.0.0.1:8000 (bridge can't);
#    BIND_HOST=127.0.0.1 keeps the UI localhost-only (the proxy is its door — NOT 0.0.0.0, which is LAN-open).
#    privileged + pid:host so nvidia-smi compute-apps sees GPU procs (unified-mem "used" on GB10). SSH to remote
#    boxes uses the DEDICATED, DISPOSABLE key cluster/setup.sh made (~/.ssh/id_myllmbox), already authorized on
#    the boxes there — NEVER the cluster admin key. We COPY it in root-owned (step 4); host ~/.ssh is never
#    mounted (uid 1000 → ssh "bad owner"), and no key is ever kept in this repo.
DASH_KEY="$HOME/.ssh/id_myllmbox"
keymount=(); [ -f "$DASH_KEY" ] && keymount=(-v "$DASH_KEY:/keys/id_myllmbox:ro")
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" --restart unless-stopped \
  --network host --privileged --pid host \
  -e BIND_HOST=127.0.0.1 -e PORT="$PORT" -e NODE_ENV=production \
  -e HOST_PROC_PATH=/host/proc -e HOST_SYS_PATH=/host/sys -e HOST_ROOT_PATH=/host/root \
  -v "$CFG:/app/config" \
  "${keymount[@]}" \
  -v /proc:/host/proc:ro -v /sys:/host/sys:ro -v /:/host/root:ro \
  -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
  -v /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:ro \
  "$IMAGE"

# 4. copy the disposable key in as ROOT-owned (ssh's owner check needs uid 0; the host source stays uid-1000 and
#    is never chowned). Named id_ed25519 so ssh picks it by default. Nothing persists in this repo.
if [ -f "$DASH_KEY" ]; then
  docker exec "$CONTAINER" sh -c 'mkdir -p /root/.ssh && chmod 700 /root/.ssh && \
    install -m 600 /keys/id_myllmbox /root/.ssh/id_ed25519 && chown -R root:root /root/.ssh'
else
  echo "⚠ ~/.ssh/id_myllmbox missing — remote-box metrics need it. Run cluster/setup.sh to create + authorize it."
fi
echo "sparkDash up on 127.0.0.1:${PORT} (fronted by the proxy)"
