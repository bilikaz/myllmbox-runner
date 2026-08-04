# Shared helpers for cluster/setup.sh and cluster/add.sh. Sourced from the repo root (the scripts cd there).
# Reads cluster.yaml via python (pyyaml). Runs ON THE HEAD; provisions the head locally + workers over ssh.

_py() { [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3; }

# cluster.yaml accessors
cy_boxes() { "$(_py)" -c 'import yaml;print(" ".join((yaml.safe_load(open("cluster.yaml"))or{}).get("boxes") or {}))'; }
cy_field() {  # <box> <field>  → value ("" if unset)
  "$(_py)" -c 'import sys,yaml
b=((yaml.safe_load(open("cluster.yaml"))or{}).get("boxes") or {}).get(sys.argv[1]) or {}
print(b.get(sys.argv[2]) or "")' "$1" "$2"
}
box_host()   { cy_field "$1" host; }
box_user()   { cy_field "$1" ssh_user; }
box_ic()     { local v; v="$(cy_field "$1" interconnect)"; [ -n "$v" ] && echo "$v" || box_host "$1"; }
box_target() { local u h; u="$(box_user "$1")"; h="$(box_host "$1")"; [ -n "$u" ] && echo "$u@$h" || echo "$h"; }

is_local() { ip -o addr 2>/dev/null | grep -qw "$1"; }   # is IP $1 on THIS machine? (→ head runs locally)

# Ensure passwordless ssh to a raw target (user@host). ssh-copy-id prompts for the box PASSWORD the first
# time (that's the only place we ever touch it — never stored). No-op if keys already work.
copy_key() {  # <user@host>
  local tgt="$1"
  is_local "${tgt##*@}" && { echo "  local head — no key needed" >&2; return 0; }
  if ssh -o BatchMode=yes -o ConnectTimeout=6 "$tgt" true 2>/dev/null; then
    echo "  passwordless ssh already works" >&2; return 0; fi
  [ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 >&2
  echo "  installing ssh key on $tgt — enter its password when prompted:" >&2
  ssh-copy-id -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519.pub "$tgt" >&2
}

# Probe a box over ssh: emit "IFACE <name> <ip>" per global IPv4 iface, "HCA <dev>" per RDMA device, plus
# GPU/DOCKER lines. So the user never has to know cryptic iface/HCA names — the script discovers them.
detect_box() {  # <user@host> — probes over ssh, or locally when it's this machine (head)
  local host="${1##*@}" script='
    ip -o -4 addr show 2>/dev/null | while read -r _ ifc _ cidr _; do
      case "$ifc" in lo|docker*|veth*|br-*|virbr*|cni*|flannel*|tailscale*|wg*) continue;; esac  # skip virtual
      hca=$(ls /sys/class/net/$ifc/device/infiniband/ 2>/dev/null | head -1)   # RDMA HCA bound to this iface
      echo "IFACE $ifc ${cidr%%/*} ${hca:--}"
    done
    echo "GPU $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
    echo "DOCKER $(docker --version 2>/dev/null || echo none)"'
  if is_local "$host"; then bash -c "$script"; else ssh -o BatchMode=yes "$1" "$script"; fi
}

# All non-mgmt interconnect-candidate interfaces on a box → lines "iface ip hca" (hca "-" if none). Used by
# mesh.sh to probe every possible link (incl. splitter sub-interfaces), not just the one setup.sh picked.
box_candidates() {  # <box>
  local host; host="$(box_host "$1")"
  detect_box "$(box_target "$1")" | awk -v h="$host" '$1=="IFACE" && $3!=h {print $2, $3, $4}'
}

# The RDMA HCA bound to a given iface (so we pin NCCL_IB_HCA to the RIGHT device on multi-HCA boxes), "" if none.
iface_hca() {  # <user@host> <iface>
  local host="${1##*@}" cmd="ls /sys/class/net/$2/device/infiniband/ 2>/dev/null | head -1"
  if is_local "$host"; then bash -c "$cmd"; else ssh -o BatchMode=yes "$1" "$cmd"; fi
}

# Full interactive add of ONE box: ask ip+user, install the key, auto-detect the network/GPU, let the user
# pick the interconnect iface (defaults to the ssh iface = single-LAN), and APPEND the block to cluster.yaml.
# Returns 1 when no host is entered (caller stops the loop).
wizard_add_box() {  # <default-name>
  local def="$1" name host user tgt out
  read -rp "Box name [$def]: " name; name="${name:-$def}"
  read -rp "  ssh host / IP (blank = done): " host
  [ -n "$host" ] || return 1
  read -rp "  ssh user [$USER]: " user; user="${user:-$USER}"
  tgt="${user}@${host}"
  copy_key "$tgt" || { echo "  ⚠ ssh to $tgt failed — skipping" >&2; return 0; }
  echo "  probing $name…" >&2
  out="$(detect_box "$tgt")" || { echo "  ⚠ probe failed" >&2; return 0; }
  echo "$out" | grep -E "^(GPU|DOCKER) " | sed 's/^/    /' >&2
  # parse "IFACE <name> <ip> <hca|->" into parallel arrays
  local names=() ips=() hcas=() _k n ip h
  while read -r _k n ip h; do names+=("$n"); ips+=("$ip"); hcas+=("$h"); done < <(echo "$out" | grep '^IFACE ')
  [ "${#names[@]}" -ge 1 ] || { echo "  ⚠ no usable interfaces on $name — skipping" >&2; return 0; }
  # AUTO-DISCOVER the interconnect: candidates = every iface EXCEPT the mgmt one (the IP we ssh'd in on).
  # 0 candidates → single-LAN (interconnect IS the mgmt iface).  1 → use it, no questions (Spark/switch).
  # 2+ → a real ring/mesh (multiple interconnect NICs): only then ask (topology auto-map is a future step).
  local cand=() k idx
  for k in "${!names[@]}"; do [ "${ips[$k]}" != "$host" ] && cand+=("$k"); done
  if [ "${#cand[@]}" -eq 0 ]; then
    for k in "${!names[@]}"; do [ "${ips[$k]}" = "$host" ] && idx=$k; done
    echo "  no separate interconnect — using LAN iface ${names[$idx]} (${ips[$idx]})" >&2
  elif [ "${#cand[@]}" -eq 1 ]; then
    idx="${cand[0]}"
    echo "  auto-detected interconnect: ${names[$idx]} ${ips[$idx]}$([ "${hcas[$idx]}" != - ] && echo " (RDMA ${hcas[$idx]})")" >&2
  else
    echo "  multiple interconnect links (ring/mesh) — pick the one to use:" >&2
    for k in "${cand[@]}"; do
      printf "    [%d] %-14s %-16s %s\n" "$((k+1))" "${names[$k]}" "${ips[$k]}" \
        "$([ "${hcas[$k]}" != - ] && echo "RDMA:${hcas[$k]}")" >&2
    done
    local sel; read -rp "  choice [$((cand[0]+1))]: " sel; idx=$(( ${sel:-$((cand[0]+1))} - 1 ))
  fi
  local ic_iface="${names[$idx]}" ic_ip="${ips[$idx]}" hca="${hcas[$idx]}"
  [ "$hca" = "-" ] && hca=""
  {
    echo "  $name:"
    echo "    host: $host"
    echo "    interconnect: $ic_ip"
    echo "    iface: $ic_iface"
    [ -n "$hca" ] && echo "    ib_hca: $hca"
    echo "    ssh_user: $user"
  } >> cluster.yaml
  echo "  ✓ $name → interconnect $ic_ip via $ic_iface${hca:+, hca $hca}" >&2
}

# run a command on a box: locally if it's this machine, else over ssh. Extra ssh opts via $SSH_OPTS.
run_on() {
  local box="$1"; shift
  if is_local "$(box_host "$box")"; then bash -lc "$*"
  else ssh -o BatchMode=yes ${SSH_OPTS:-} "$(box_target "$box")" "$*"; fi
}

# Ensure the head can ssh to a box passwordlessly (install the head's pubkey; prompts for the box password once).
install_key() {
  local box="$1" tgt; tgt="$(box_target "$box")"
  if is_local "$(box_host "$box")"; then echo "  [$box] local head — no key needed"; return; fi
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$tgt" true 2>/dev/null; then
    echo "  [$box] passwordless ssh already works"; return
  fi
  [ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519   # head keypair (once)
  echo "  [$box] installing ssh key → $tgt (enter its password when asked)"
  ssh-copy-id -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519.pub "$tgt"
}

# docker + GPU + models dir on a box (idempotent, read-mostly).
provision_box() {
  local box="$1"
  echo "  [$box] docker: $(run_on "$box" 'command -v docker >/dev/null && docker --version || echo MISSING ⚠')"
  echo "  [$box] gpu:    $(run_on "$box" 'nvidia-smi -L 2>/dev/null | head -1 || echo "none / nvidia-smi missing ⚠"')"
  run_on "$box" 'mkdir -p ~/spark-vllm-docker/models' && echo "  [$box] models dir ✓"
}

# Open ufw for one or more peer IPs on a box in a SINGLE sudo call → at most ONE sudo prompt per box, not one
# per rule (idempotent; ufw skips duplicates). Local → sudo bash -c; remote → ssh -t (tty so sudo can prompt).
ufw_allow_from() {  # <box> <peer_ip>...
  local box="$1"; shift
  [ "$#" -ge 1 ] || return 0
  local inner="" ip
  for ip in "$@"; do inner+="ufw allow from $ip comment myllmbox-peer; "; done
  if is_local "$(box_host "$box")"; then sudo bash -c "$inner"
  else ssh -t "$(box_target "$box")" "sudo bash -c '$inner'"; fi
}

# Full mesh: on every box, allow every OTHER box's interconnect IP (one sudo call per box). Works whether the
# interconnect is a dedicated range (ConnectX 169.254.x) or the same LAN as mgmt — we whitelist peer IPs, not a
# subnet, so the management LAN stays otherwise closed.
mesh_firewall() {
  local all box; all="$(cy_boxes)"
  # EVERY interconnect IP of EVERY box (all candidate ifaces — splitters / many cables), so whichever link a
  # peer uses is allowed. Detected fresh so re-cabling is picked up. One sudo call per box.
  local allips=""; for box in $all; do allips+=" $(box_candidates "$box" | awk '{print $2}')"; done
  allips="$(echo "$allips" | xargs)"
  [ -n "$allips" ] || { echo "  (no interconnect IPs detected)"; return 0; }
  for box in $all; do
    echo "  [$box] ufw allow from all interconnect IPs ($(echo "$allips" | wc -w) addrs)"
    ufw_allow_from "$box" $allips || echo "  ⚠ [$box] ufw failed (installed/active? sudo?)"
  done
}

# Wire ONE new box into an existing mesh (add.sh): open ALL the new box's interconnect IPs on every existing
# box, and every peer's IPs on the new box. Logs INTO the old boxes — else their firewalls drop the newcomer.
mesh_new_box() {  # <newbox>
  local new="$1" newips peerips="" peer
  newips="$(box_candidates "$new" | awk '{print $2}' | xargs)"
  for peer in $(cy_boxes); do
    [ "$peer" = "$new" ] && continue
    echo "  [$peer] ufw allow from $new ($newips)"; ufw_allow_from "$peer" $newips || echo "  ⚠ [$peer] failed"
    peerips+=" $(box_candidates "$peer" | awk '{print $2}')"
  done
  peerips="$(echo "$peerips" | xargs)"
  [ -n "$peerips" ] && { echo "  [$new] ufw allow from peers ($peerips)"; ufw_allow_from "$new" $peerips || echo "  ⚠ [$new] failed"; }
}
