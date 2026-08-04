#!/usr/bin/env bash
# build-and-copy.sh <recipe>  —  build recipes/<recipe>/Dockerfile into the box image mbx-<recipe>, then
# copy that image to every OTHER node in the recipe's cluster: block (over ssh, on the ConnectX link).
# Mirror of run.sh: the folder name IS the recipe. Build ONCE here (the head), distribute to the workers.
#
#   ./build-and-copy.sh deepseek-v4-flash-0731
#
set -euo pipefail
cd "$(dirname "$0")"

R="${1:-}"
if [ -z "$R" ]; then
  echo "usage: ./build-and-copy.sh <recipe>"
  echo "recipes: $(ls recipes 2>/dev/null | tr '\n' ' ')"
  exit 1
fi
D="recipes/$R"
[ -f "$D/Dockerfile" ] || { echo "no Dockerfile: $D/Dockerfile (a recipe needs a Dockerfile to build)"; exit 1; }

# venv — only used here to read the recipe's cluster node list. Same bootstrap as run.sh; you touch nothing.
V=.venv
[ -x "$V/bin/python" ] || python3 -m venv "$V"
"$V/bin/python" -m pip install -q -U pip pyyaml >/dev/null

# base box: recipes whose Dockerfile does `FROM mbx-base` need it present to build. Workers don't — the
# base layers are baked into the copied image.
if grep -qiE '^[[:space:]]*FROM[[:space:]]+mbx-base' "$D/Dockerfile" && ! docker image inspect mbx-base:latest >/dev/null 2>&1; then
  echo "· building base box (./Dockerfile) → mbx-base"
  docker build -t mbx-base:latest -f Dockerfile .
fi

IMG="mbx-$R"
# BUILD_JOBS env throttles compile parallelism (lower it to coexist with a running serve on the same box).
JOBS_ARG=(); [ -n "${BUILD_JOBS:-}" ] && JOBS_ARG=(--build-arg "BUILD_JOBS=$BUILD_JOBS")
echo "· building $D/Dockerfile → $IMG${BUILD_JOBS:+  (BUILD_JOBS=$BUILD_JOBS)}"
docker build -t "$IMG" "${JOBS_ARG[@]}" "$D"

# copy to every OTHER cluster node (node 0 = this box, already built). Over the ConnectX link via ssh.
NODES=$("$V/bin/python" - "$D/myllmbox.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
nodes = ((cfg.get("cluster") or {}).get("nodes") or [])
print(" ".join(str(n) for n in nodes[1:]))
PY
)
U="${SSH_USER:-$USER}"
SIZE=$(docker image inspect -f '{{.Size}}' "$IMG" 2>/dev/null || echo 0)
HSIZE=$(numfmt --to=iec "$SIZE" 2>/dev/null || echo "${SIZE}B")
if command -v pv >/dev/null 2>&1; then
  PROG=(pv -s "$SIZE" -N "$IMG")            # live progress bar: % · rate · ETA
else
  PROG=(cat); echo "  (tip: install 'pv' for a copy progress bar — falling back to a silent stream)"
fi
for node in $NODES; do
  echo "· copying $IMG ($HSIZE) → $node  (docker save | ssh docker load)"
  docker save "$IMG" | "${PROG[@]}" | ssh -o BatchMode=yes "${U}@${node}" docker load
done
echo "· done — $IMG ready on this node${NODES:+ + $NODES}"
