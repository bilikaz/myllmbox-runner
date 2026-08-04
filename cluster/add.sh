#!/usr/bin/env bash
# cluster/add.sh <box> — add ONE box to a running cluster. If <box> isn't in cluster.yaml yet, it asks for its
# ssh IP + user (+ password once), installs a key, auto-detects its network/GPU and appends it. Then it
# provisions the new box and, crucially, logs INTO every existing box to ufw-allow the newcomer (and allows
# the peers on the new box) — otherwise the old boxes' firewalls would drop it. Idempotent. Run on the head.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=cluster/lib.sh
source cluster/lib.sh

BOX="${1:-}"
[ -n "$BOX" ] || { echo "usage: cluster/add.sh <box-name>"; exit 1; }
[ -f cluster.yaml ] || { echo "no cluster.yaml — run cluster/setup.sh first"; exit 1; }

if cy_boxes | tr ' ' '\n' | grep -qx "$BOX"; then
  echo "── '$BOX' already in cluster.yaml — re-provisioning + wiring firewall ──"
  NEW="$BOX"
else
  echo "── new box '$BOX' — enter its details (name defaults to $BOX) ──"
  wizard_add_box "$BOX" || { echo "nothing added"; exit 1; }
  NEW="$(cy_boxes | tr ' ' '\n' | tail -1)"     # the box just appended (whatever name was confirmed)
fi

echo "── provisioning $NEW ──"
install_key "$NEW"
provision_box "$NEW"

echo "── firewall: allow $NEW on every existing box, and the peers on $NEW (sudo may prompt) ──"
mesh_new_box "$NEW"

echo "✓ $NEW wired in. Cluster now: $(cy_boxes) — add it to a recipe's cluster.boxes to serve across it."
