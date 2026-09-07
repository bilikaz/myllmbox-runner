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

# Pre-flight: every box's ssh port must be reachable BEFORE we touch anything. If a box's ufw is already
# blocking ssh we CANNOT fix it remotely (chicken-and-egg) — error with the exact unblock command instead of
# failing cryptically mid-run. The head is skipped (we're already on it).
preflight_reachable() {
  local box host bad=0
  for box in $(cy_boxes); do
    host="$(box_host "$box")"
    if is_local "$host"; then echo "  [$box] $host — local head ✓"; continue; fi
    if timeout 4 bash -c ">/dev/tcp/$host/22" 2>/dev/null; then
      echo "  [$box] $host:22 reachable ✓"
    else
      echo "  ✗ [$box] cannot reach $host:22 — ssh blocked or box down." >&2
      echo "      If its firewall is blocking ssh, on $box run (console/keyboard):" >&2
      echo "          sudo ufw allow OpenSSH        # or: sudo ufw disable" >&2
      echo "      then rerun. (We can't open it remotely — ssh is how we'd get in.)" >&2
      bad=1
    fi
  done
  [ "$bad" = 0 ] || return 1
}

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

# The OTHER PCIe half of the same RDMA card: on a DGX Spark the ConnectX-7 sits on two PCIe Gen5 x4 links and shows as
# two verbs devices (rocep1s0f1 + roceP2p1s0f1); each carries ~13 GB/s, NCCL striped over both ~20 GB/s. Emits
# "SIB <hca> <netdev> <ipv4|->" for every ACTIVE device of the same port speed other than the given one.
hca_siblings() {  # <user@host> <hca>
  local host="${1##*@}" script='
    h="$1"; rt=$(cat /sys/class/infiniband/$h/ports/1/rate 2>/dev/null | awk "{print \$1}")
    for d in /sys/class/infiniband/*; do n=$(basename $d); [ "$n" = "$h" ] && continue
      st=$(cat $d/ports/1/state 2>/dev/null | awk "{print \$2}"); r=$(cat $d/ports/1/rate 2>/dev/null | awk "{print \$1}")
      [ "$st" = ACTIVE ] && [ "$r" = "$rt" ] || continue
      nd=$(ls $d/device/net 2>/dev/null | head -1); ip=$(ip -4 -o addr show "$nd" 2>/dev/null | awk "{print \$4}" | head -1)
      echo "SIB $n ${nd:--} ${ip:--}"; done'
  if is_local "$host"; then bash -c "$script" _ "$2"; else ssh -o BatchMode=yes "$1" "bash -c $(printf %q "$script") _ $(printf %q "$2")"; fi
}

# First RDMA device of a box's ib_hca (a string or a list in cluster.yaml), "" if none.
cy_hca_first() {  # <box>
  "$(_py)" -c 'import sys,yaml
b=((yaml.safe_load(open("cluster.yaml"))or{}).get("boxes") or {}).get(sys.argv[1]) or {}
h=b.get("ib_hca") or ""; print((h[0] if h else "") if isinstance(h,list) else str(h).split(",")[0])' "$1"
}

# Give a netdev a persistent link-local IPv4 through NetworkManager (what the Spark's first ConnectX port already
# uses). ONE sudo call on that box, same pattern as ufw_allow_from: local → sudo bash -c, remote → ssh -t.
nm_linklocal() {  # <box> <netdev>
  local box="$1" nd="$2"
  local inner="con=\$(nmcli -t -f NAME,DEVICE con show | grep ':$nd\$' | cut -d: -f1 | head -1);
if [ -z \"\$con\" ]; then con=myllmbox-$nd; nmcli con add type ethernet ifname $nd con-name \"\$con\" >/dev/null; fi;
nmcli con mod \"\$con\" ipv4.method link-local ipv6.method disabled connection.autoconnect yes && nmcli con up \"\$con\" >/dev/null && echo \"  ✓ $nd \$(ip -4 -br addr show $nd | tr -s ' ' | cut -d' ' -f3)\""
  if is_local "$(box_host "$box")"; then sudo bash -c "$inner"
  else ssh -t "$(box_target "$box")" "sudo bash -c '$inner'"; fi
}

# The second PCIe half of every box's RDMA card (see hca_siblings). A half whose interface has no IPv4 cannot carry
# RoCE v2 at all; give it a link-local address (consent-gated sudo, once per box), then — only when EVERY box with
# an RDMA device has an addressed sibling — write the pair into cluster.yaml so the runner stripes NCCL over both.
mesh_rdma_halves() {
  local box hca tgt out sn snd sip pairs="" missing=0 any=0
  for box in $(cy_boxes); do
    hca="$(cy_hca_first "$box")"; [ -n "$hca" ] || continue
    any=1; tgt="$(box_target "$box")"
    out="$(hca_siblings "$tgt" "$hca" | grep '^SIB ' | head -1 || true)"
    if [ -z "$out" ]; then echo "  [$box] $hca — no second ACTIVE port of the same speed (single-link card)"; missing=1; continue; fi
    read -r _ sn snd sip <<<"$out"
    if [ "$sip" = "-" ]; then
      echo "  [$box] $sn is the other PCIe half of the card ($hca's twin, ~13 GB/s more) but $snd has no IPv4."
      read -rp "        give $snd a link-local address on $box now (sudo on $box; persistent)? [y/N]: " ok
      case "$ok" in [Yy]*) nm_linklocal "$box" "$snd" || echo "  ⚠ [$box] nmcli failed";; *) echo "  · skipped";; esac
      out="$(hca_siblings "$tgt" "$hca" | grep '^SIB ' | head -1 || true)"; read -r _ sn snd sip <<<"$out"
    fi
    if [ "$sip" != "-" ]; then echo "  [$box] ✓ $hca + $sn ($snd $sip)"; pairs+="$box|$hca|$sn"$'\n'
    else missing=1; fi
  done
  [ "$any" = 1 ] || { echo "  (no RDMA devices in cluster.yaml)"; return 0; }
  if [ "$missing" = 0 ]; then
    # data via an env var — a pipe into `python - <<heredoc` is silently swallowed by the heredoc (the bug that made
    # this step and mesh.sh print success while writing nothing, 2026-09-07)
    MBX_PAIRS="$pairs" "$(_py)" - cluster.yaml <<'PY'
import os, sys, yaml
path = sys.argv[1]; d = yaml.safe_load(open(path)) or {}; boxes = d.get("boxes") or {}
for line in os.environ.get("MBX_PAIRS", "").splitlines():
    if not line.strip(): continue
    name, a, b = line.rstrip("\n").split("|")
    if name in boxes: boxes[name]["ib_hca"] = [a, b]
yaml.safe_dump(d, open(path, "w"), sort_keys=False, default_flow_style=None)
print("→ cluster.yaml: ib_hca = [first, second] on every box — NCCL stripes over both PCIe halves", file=sys.stderr)
PY
  else
    echo "  → keeping one device per box in cluster.yaml (a listed half without an address would fail NCCL init). Rerun cluster/setup.sh after fixing."
  fi
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
  local cand=() k idx sibs allsib
  for k in "${!names[@]}"; do [ "${ips[$k]}" != "$host" ] && cand+=("$k"); done
  if [ "${#cand[@]}" -eq 0 ]; then
    for k in "${!names[@]}"; do [ "${ips[$k]}" = "$host" ] && idx=$k; done
    echo "  no separate interconnect — using LAN iface ${names[$idx]} (${ips[$idx]})" >&2
  elif [ "${#cand[@]}" -eq 1 ]; then
    idx="${cand[0]}"
    echo "  auto-detected interconnect: ${names[$idx]} ${ips[$idx]}$([ "${hcas[$idx]}" != - ] && echo " (RDMA ${hcas[$idx]})")" >&2
  elif sibs="$([ "${hcas[${cand[0]}]}" != - ] && hca_siblings "$tgt" "${hcas[${cand[0]}]}" | awk '{print $2}')" && [ -n "$sibs" ] &&
       { allsib=1; for k in "${cand[@]:1}"; do echo "$sibs" | grep -qx "${hcas[$k]}" || allsib=0; done; [ "$allsib" = 1 ]; }; then
    # the PCIe halves of ONE card show up as several links — they are one link. Take the first; the pair lands in ib_hca.
    idx="${cand[0]}"
    echo "  auto-detected interconnect: ${names[$idx]} ${ips[$idx]} (RDMA ${hcas[$idx]}; ${#cand[@]} PCIe halves of one card)" >&2
  else
    echo "  multiple interconnect links (ring/mesh) — pick the one to use:" >&2
    for k in "${cand[@]}"; do
      printf "    [%d] %-14s %-16s %s\n" "$((k+1))" "${names[$k]}" "${ips[$k]}" \
        "$([ "${hcas[$k]}" != - ] && echo "RDMA:${hcas[$k]}")" >&2
    done
    local sel; read -rp "  choice [$((cand[0]+1))]: " sel; idx=$(( ${sel:-$((cand[0]+1))} - 1 ))
  fi
  local ic_iface="${names[$idx]}" ic_ip="${ips[$idx]}" hca="${hcas[$idx]}" hca_yaml=""
  [ "$hca" = "-" ] && hca=""
  if [ -n "$hca" ]; then
    # the second PCIe half of the card: use it when its netdev already has an IPv4 (RoCE v2 needs one); otherwise
    # say exactly what to do (root — this script never runs it) and pin the one device that works.
    hca_yaml="$hca"; local _s sn snd sip
    while read -r _s sn snd sip; do
      if [ "$sip" != "-" ]; then hca_yaml="[$hca, $sn]"; echo "  ✓ second PCIe half $sn ($snd $sip) — NCCL will stripe over both" >&2
      else
        echo "  ⚠ $sn is the other PCIe half of the same card (ACTIVE, ~13 GB/s more) but $snd has no IPv4, so it stays unused." >&2
        echo "    To enable (root, once, on this box), then rerun cluster/setup.sh:" >&2
        echo "      sudo nmcli con mod \"\$(nmcli -t -f NAME,DEVICE con show | grep ':$snd\$' | cut -d: -f1)\" ipv4.method link-local ipv6.method disabled" >&2
        echo "      sudo nmcli con up \"\$(nmcli -t -f NAME,DEVICE con show | grep ':$snd\$' | cut -d: -f1)\"" >&2
      fi
    done < <(hca_siblings "$tgt" "$hca" | grep '^SIB ')
  fi
  {
    echo "  $name:"
    echo "    host: $host"
    echo "    interconnect: $ic_ip"
    echo "    iface: $ic_iface"
    [ -n "$hca" ] && echo "    ib_hca: $hca_yaml"
    echo "    ssh_user: $user"
  } >> cluster.yaml
  echo "  ✓ $name → interconnect $ic_ip via $ic_iface${hca:+, hca $hca_yaml}" >&2
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

# Install the DEDICATED, DISPOSABLE dashboard key (id_myllmbox) on a box. This is the ONLY key the dashboard
# container ever gets (copied in read-only) — so if it leaks you rotate JUST this key (rm ~/.ssh/id_myllmbox*,
# strip it from each box's authorized_keys, rerun setup); the cluster admin key id_ed25519 is never exposed.
# Idempotent, uses the passwordless ssh install_key just established (no password prompt).
install_dashboard_key() {
  local box="$1" tgt; tgt="$(box_target "$box")"
  is_local "$(box_host "$box")" && { echo "  [$box] local head — dashboard reads it directly (no key)"; return; }
  [ -f ~/.ssh/id_myllmbox ] || ssh-keygen -t ed25519 -N "" -C myllmbox-dashboard -f ~/.ssh/id_myllmbox
  local pub; pub="$(cat ~/.ssh/id_myllmbox.pub)"
  ssh -o BatchMode=yes -o ConnectTimeout=6 "$tgt" \
    "mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && grep -qxF '$pub' ~/.ssh/authorized_keys || echo '$pub' >> ~/.ssh/authorized_keys" \
    && echo "  [$box] dashboard key (id_myllmbox) authorized" || echo "  [$box] ⚠ could not authorize dashboard key"
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
  # ALWAYS re-allow ssh first, in the same batch — so touching ufw can never lock us out (admin ssh over the
  # mgmt LAN AND inter-box ssh). Belt-and-suspenders: harmless if already allowed / ufw inactive.
  local inner="ufw allow 22/tcp comment myllmbox-ssh; " ip
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
