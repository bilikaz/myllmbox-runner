#!/usr/bin/env bash
# cluster/add.sh <box> — add ONE box (already defined in cluster.yaml) to the running cluster:
#   · install the head's ssh key on it, verify docker + GPU, make the models dir
#   · re-mesh the firewall so every box allows the new one and the new one allows every peer
# Idempotent. Run on the head after adding the box's entry to cluster.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=cluster/lib.sh
source cluster/lib.sh

BOX="${1:-}"
[ -n "$BOX" ] || { echo "usage: cluster/add.sh <box-name>   (boxes in cluster.yaml: $(cy_boxes))"; exit 1; }
cy_boxes | tr ' ' '\n' | grep -qx "$BOX" || { echo "no box '$BOX' in cluster.yaml (have: $(cy_boxes))"; exit 1; }

echo "── provisioning $BOX  (ssh $(box_target "$BOX"), interconnect $(box_ic "$BOX")) ──"
install_key "$BOX"
provision_box "$BOX"

echo "── re-meshing firewall across all boxes (sudo may prompt) ──"
mesh_firewall

echo "✓ $BOX added — reference it in a recipe's cluster.boxes to include it in a serve"
