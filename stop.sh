#!/usr/bin/env bash
# stop.sh — bring THIS serve down (mirror of run.sh): model + ITS cluster workers + proxy + tunnel + dashboard.
#
# It tears down ONLY what this serve started, tracked in .mbx/.running (its pids + the exact worker boxes it
# used + its dashboard). It NEVER touches other boxes just because they're in cluster.yaml — those may be
# running independent serves (a video job mid-flight, another model), and a blind cluster-wide sweep would
# kill them. cluster.yaml is an address book, not "boxes owned by this serve."
set -euo pipefail
cd "$(dirname "$0")"
V=.venv

# The runner's teardown: reads .mbx/.running → kills proxy + tunnel (by pid), stops the model + THIS serve's
# workers (ssh only the boxes recorded in .running), runs the dashboard's down.sh, then deletes .running.
[ -x "$V/bin/python" ] && "$V/bin/python" -m runner.cli down || echo "· runner down skipped (no venv)"

# Head-local safety net — THIS box only (safe: it's this serve's head). Covers a lost/stale .running for the
# host processes + local containers. We do NOT reach onto other boxes here — if .running was lost, a worker
# elsewhere must be cleaned by hand (docker rm -f mbx-vllm on that box), never swept blindly.
pkill -f "runner\.proxy"        2>/dev/null || true
pkill -f "cloudflared.*tunnel"  2>/dev/null || true
docker rm -f mbx-vllm mbx-dashboard >/dev/null 2>&1 || true

echo "✓ stopped this serve — proxy, tunnel, dashboard, model + its recorded workers."
echo "  (if .mbx/.running was lost, a worker on another box may need a manual: docker rm -f mbx-vllm)"
