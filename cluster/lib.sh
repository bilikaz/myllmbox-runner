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

# Open ufw on a box for a peer's interconnect IP (idempotent; ufw skips duplicate rules). sudo may prompt.
ufw_allow() {  # <box> <peer_ip>
  local box="$1" peer="$2" cmd="sudo ufw allow from $peer comment 'myllmbox cluster peer'"
  if is_local "$(box_host "$box")"; then eval "$cmd"
  else ssh -t -o BatchMode=yes "$(box_target "$box")" "$cmd"; fi   # -t: tty so sudo can prompt
}

# Full mesh: on every box, allow traffic from every OTHER box's interconnect IP. Works whether the interconnect
# is a dedicated range (ConnectX 169.254.x) or the same LAN as mgmt (192.168.1.x) — we whitelist peer IPs, not
# a subnet, so the management LAN stays otherwise closed.
mesh_firewall() {
  local all box peer pip; all="$(cy_boxes)"
  for box in $all; do
    for peer in $all; do
      [ "$peer" = "$box" ] && continue
      pip="$(box_ic "$peer")"
      echo "  [$box] ufw allow from $peer ($pip)"
      ufw_allow "$box" "$pip" || echo "  ⚠ [$box] ufw rule for $pip failed (ufw installed/active? sudo ok?)"
    done
  done
}
