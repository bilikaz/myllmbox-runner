#!/usr/bin/env bash
# cluster/mesh.sh — discover the interconnect TOPOLOGY and configure NCCL to use every working link.
#
# Run this LAST, after cluster/setup.sh (all boxes provisioned + firewalls open) — probing needs the firewall
# open. RERUN it anytime you re-cable (add/remove cables, splitters 1→N, etc.). For each box it pings every
# peer over EACH of its interconnect interfaces; the interfaces that reach peers become that box's NCCL link
# set (comma-list → NCCL uses them all for bandwidth). It prints the adjacency and flags any box that can't
# reach a peer (a partial mesh / ring — which needs a switch or explicit routing; mesh.sh does NOT touch
# routing tables).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=cluster/lib.sh
source cluster/lib.sh
[ -f cluster.yaml ] || { echo "no cluster.yaml — run cluster/setup.sh first"; exit 1; }

BOXES="$(cy_boxes)"
[ -n "$BOXES" ] || { echo "cluster.yaml has no boxes"; exit 1; }
echo "== probing interconnect topology across: $BOXES =="

declare -A CANDS PIPS IFLIST PRIMARY REACH
for b in $BOXES; do
  CANDS[$b]="$(box_candidates "$b")"                                   # "iface ip hca" lines
  PIPS[$b]="$(echo "${CANDS[$b]}" | awk '{print $2}' | tr '\n' ' ')"   # this box's candidate IPs
done

for a in $BOXES; do
  echo "── $a ──"
  while read -r aif aip _; do
    [ -n "$aif" ] || continue
    reached=""
    for b in $BOXES; do
      [ "$b" = "$a" ] && continue
      for bip in ${PIPS[$b]}; do
        if run_on "$a" "ping -I $aif -c1 -W1 $bip >/dev/null 2>&1"; then reached+=" $b"; break; fi
      done
    done
    if [ -n "$reached" ]; then
      echo "  $aif ($aip) → reaches:$reached"
      IFLIST[$a]="${IFLIST[$a]:-}${IFLIST[$a]:+,}$aif"
      REACH[$a]="${REACH[$a]:-}$reached"
      [ -z "${PRIMARY[$a]:-}" ] && PRIMARY[$a]="$aip"
    else
      echo "  $aif ($aip) → (no peers — not an interconnect link)"
    fi
  done <<< "${CANDS[$a]}"
  [ -n "${IFLIST[$a]:-}" ] || echo "  ⚠ $a reaches NO peers over any interface!"
done

# all-to-all check (NCCL bootstrap + vLLM mq need every box to reach every other)
echo "== connectivity =="
FULL=1
for a in $BOXES; do
  for b in $BOXES; do
    [ "$b" = "$a" ] && continue
    case " ${REACH[$a]:-} " in *" $b "*) : ;; *) echo "  ✗ $a cannot reach $b"; FULL=0;; esac
  done
done
if [ "$FULL" = 1 ]; then echo "  ✓ fully connected — all-to-all over the interconnect"
else echo "  ⚠ NOT all-to-all: a partial mesh / ring. NCCL will hang. Use a switch (one interconnect subnet)"
     echo "    or add explicit routes on the boxes; mesh.sh only discovers, it won't change routing tables."
fi

# Recommended rank ORDER: walk the adjacency from the HEAD (rank 0 must stay the head) following direct links,
# so on a ring/chain the NCCL ranks line up with the physical neighbours. On a full mesh order is arbitrary.
declare -A NB VIS
for a in $BOXES; do NB[$a]="$(echo "${REACH[$a]:-}" | tr ' ' '\n' | sort -u | xargs)"; done
start=""; for a in $BOXES; do is_local "$(box_host "$a")" && start="$a"; done
[ -n "$start" ] || start="$(echo "$BOXES" | awk '{print $1}')"
order=""; cur="$start"
while [ -n "$cur" ]; do
  order+=" $cur"; VIS[$cur]=1; nxt=""
  for n in ${NB[$cur]}; do [ -z "${VIS[$n]:-}" ] && { nxt="$n"; break; }; done
  cur="$nxt"
done
for a in $BOXES; do [ -z "${VIS[$a]:-}" ] && order+=" $a"; done   # any disconnected boxes last
echo "== recommended recipe order (rank 0 = head, then along the links) =="
echo "    cluster: {boxes: [$(echo "$order" | xargs | tr ' ' ',' | sed 's/,/, /g')]}"

# write discovered links (comma-list of ifaces) + primary interconnect IP per box into cluster.yaml
{ for a in $BOXES; do echo "$a|${IFLIST[$a]:-}|${PRIMARY[$a]:-}"; done; } | "$(_py)" - cluster.yaml <<'PY'
import sys, yaml
path = sys.argv[1]
d = yaml.safe_load(open(path)) or {}
boxes = d.get("boxes") or {}
for line in sys.stdin:
    name, ifl, prim = (line.rstrip("\n").split("|", 2) + ["", ""])[:3]
    b = boxes.get(name)
    if not b:
        continue
    if ifl:
        b["iface"] = ifl          # comma-list → NCCL_SOCKET_IFNAME uses every working link
    if prim:
        b["interconnect"] = prim  # advertised IP for VLLM_HOST_IP / rendezvous
yaml.safe_dump(d, open(path, "w"), sort_keys=False, default_flow_style=False)
print("→ updated cluster.yaml iface/interconnect from discovery", file=sys.stderr)
PY

echo "✓ mesh done — rerun after any re-cabling"
