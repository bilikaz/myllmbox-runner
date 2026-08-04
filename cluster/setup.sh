#!/usr/bin/env bash
# cluster/setup.sh — provision every box in cluster.yaml so the cluster can serve:
#   · install the head's ssh key on each worker (passwordless thereafter; prompts once per box)
#   · verify docker + GPU, create the models dir
#   · open ufw on each box for every peer's interconnect IP (the firewall mesh)
# Idempotent + rerunnable. Run it ON THE HEAD (box 1). `cluster/add.sh <box>` adds one more box later.
#
# You supply per box in cluster.yaml: host / interconnect / iface / ib_hca / ssh_user. Passwords are entered
# interactively (ssh-copy-id + sudo prompts) and never stored.
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
# shellcheck source=cluster/lib.sh
source cluster/lib.sh

[ -f cluster.yaml ] || { echo "no cluster.yaml — cp cluster.yaml.example cluster.yaml and fill it in"; exit 1; }
BOXES="$(cy_boxes)"
[ -n "$BOXES" ] || { echo "cluster.yaml has no boxes"; exit 1; }
echo "cluster boxes: $BOXES"

for box in $BOXES; do
  echo "── $box  (ssh $(box_target "$box"), interconnect $(box_ic "$box")) ──"
  install_key "$box"
  provision_box "$box"
done

echo "── firewall mesh (ufw allow each peer's interconnect; sudo may prompt per box) ──"
mesh_firewall

echo "✓ cluster setup complete — recipes can now use these box names (cluster.boxes: [$(echo "$BOXES" | tr ' ' ',')])"
