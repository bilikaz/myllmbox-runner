#!/usr/bin/env bash
# cluster/server-profile.sh — make DGX Sparks BOOT into a lean, headless server form.
#
# Run once ON THE HEAD (box1); it configures EVERY box in cluster.yaml over ssh. It only writes persistent
# config — it never reboots, never touches the serve, never relaunches anything. The lean form takes effect
# on each box's NEXT boot (you reboot on your own schedule; bring the serve back however you normally do).
#
# What "lean form" means here:
#   • boots to CLI (multi-user.target) — no gdm/Xorg/gnome-shell/desktop session
#   • desktop + peripheral services disabled (see DISABLE) — a headless, ssh-only, monitor-less box needs none
#   • journald capped so logs never eat RAM/disk on a long-lived server
#   • (optional) an NVMe swapfile + low vm.swappiness as cold-start OOM insurance
#
# ── NEVER TOUCHED (hardcoded KEEP guard — a typo in DISABLE can't reach these) ─────────────────────
#   rdma-ndd  nvidia-persistenced  systemd-oomd  ssh sshd docker containerd NetworkManager
#   systemd-networkd systemd-resolved dbus polkit cron rsyslog  avahi-daemon (.local)  snapd (if snaps)
#   rdma-ndd is load-bearing: ALL cross-node NCCL + gloo rides the ConnectX. Kill it and TP dies.
#
# ── USAGE ──────────────────────────────────────────────────────────────────────────────────────────
#   cluster/server-profile.sh                 # configure both boxes for a lean NEXT boot (nothing live)
#   cluster/server-profile.sh --now           # ALSO stop the desktop services now (drops the GUI session live)
#   cluster/server-profile.sh --now --swap 32G # ... plus a >=32G NVMe swapfile + vm.swappiness=10
#   cluster/server-profile.sh --now --swap 32G --reboot   # ... then reboot every box into the lean form
#   cluster/server-profile.sh --dry-run        # print every action, touch nothing, ask no password
#   cluster/server-profile.sh --revert         # back to graphical.target + re-enable + undo swap
#   cluster/server-profile.sh box2             # restrict to named box(es)
#
# The lean boot config is persistent; reboot when convenient, or pass --reboot to have this script do it.
# --reboot takes down the live serve — the reboot itself brings the containers down. It reboots WORKERS FIRST
# and the HEAD LAST (the head is the orchestrator — it must go last, or it dies before it can reboot the
# workers), and asks for a typed confirmation. It does NOT relaunch the serve; bring it back up as you normally do.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=cluster/lib.sh
source cluster/lib.sh

DISABLE=(
  gdm gdm3 gnome-remote-desktop bluetooth cups cups-browsed ModemManager
  colord switcheroo-control rtkit-daemon accounts-daemon upower power-profiles-daemon
  apport whoopsie kerneloops
  dgx-dashboard dgx-dashboard-admin   # NVIDIA vendor dashboard — unused
  multipathd                          # multipath SAN storage — box is on local NVMe
  wpa_supplicant                      # wifi — box is wired
  lldpd                               # link-layer neighbour discovery — not used for NCCL/vLLM
)
KEEP=(
  rdma-ndd nvidia-persistenced systemd-oomd
  ssh sshd docker containerd NetworkManager systemd-networkd systemd-resolved
  dbus polkit cron rsyslog avahi-daemon snapd
)

REVERT=0; NOW=0; DRY=0; REBOOT=0; SWAP=""; BOXES_ARG=()
while [ "$#" -gt 0 ]; do case "$1" in
  --revert) REVERT=1;; --now) NOW=1;; --dry-run) DRY=1;; --reboot) REBOOT=1;;
  --swap) SWAP="${2:?--swap needs a size, e.g. 32G}"; shift;;
  --*) echo "unknown flag: $1" >&2; exit 2;;
  *) BOXES_ARG+=("$1");;
esac; shift; done

BOXES="${BOXES_ARG[*]:-$(cy_boxes)}"
[ -n "$BOXES" ] || { echo "no boxes"; exit 1; }

# head = the box that is this machine (the orchestrator); it must reboot LAST. Everything else is a worker.
HEAD=""; WORKERS=()
for b in $BOXES; do if is_local "$(box_host "$b")"; then HEAD="$b"; else WORKERS+=("$b"); fi; done

# effective disable list = DISABLE minus anything protected by KEEP (defensive: a typo can't nuke rdma-ndd etc.)
keep_re="$(printf '%s\n' "${KEEP[@]}" | paste -sd'|' -)"
EFF=(); for u in "${DISABLE[@]}"; do
  [[ "$u" =~ ^(${keep_re})$ ]] && { echo "  ⚠ refusing to disable protected unit: $u"; continue; }
  EFF+=("$u")
done
EFF_STR="${EFF[*]}"

# Run a script (stdin) as root on a box, letting the box ASK for its sudo password the normal way. sudo reads
# the password from the TTY (not stdin), so we hand the script in over stdin (base64'd to dodge all quoting)
# and use `ssh -t` to give the remote sudo a terminal to prompt on. No password is ever read into a variable,
# piped, or placed on argv — so nothing can leak to the shell or land in history. (Same pattern as lib.sh.)
sudo_script() {  # <box>   (script on stdin)
  local box="$1" s b64; s="$(cat)"
  if [ "$DRY" = 1 ]; then echo "----- [$box] would run as root: -----"; printf '%s\n' "$s"; echo "-------------------------------------"; return 0; fi
  b64="$(printf '%s' "$s" | base64 -w0)"
  if is_local "$(box_host "$box")"; then echo "$b64" | base64 -d | sudo bash
  else ssh -t "$(box_target "$box")" "echo '$b64' | base64 -d | sudo bash" </dev/tty; fi
}

config_script() {
  cat <<EOF
set -e
$( [ "$REVERT" = 1 ] && cat <<R
systemctl set-default graphical.target
systemctl unmask ${EFF_STR} 2>/dev/null || true
systemctl enable ${EFF_STR} 2>/dev/null || true
rm -f /etc/sysctl.d/99-myllmbox-swap.conf; sysctl -q vm.swappiness=60 || true
rm -f /etc/systemd/journald.conf.d/99-myllmbox.conf
echo "  reverted to desktop profile (reboot to restore GUI)"
R
)
$( [ "$REVERT" = 0 ] && cat <<C
echo "  → boot target: multi-user.target"
systemctl set-default multi-user.target >/dev/null
echo "  → disable + mask: ${EFF_STR}"
systemctl disable ${EFF_STR} 2>/dev/null || true
systemctl mask ${EFF_STR} 2>/dev/null || true
$( [ "$NOW" = 1 ] && echo "echo '  → stopping them now'; systemctl stop ${EFF_STR} 2>/dev/null || true" )
echo "  → checking snapd (8s timeout)..."
snaps=\$(timeout 8 snap list 2>/dev/null | tail -n +2)
if [ -n "\$snaps" ]; then
  echo "  → snapd: KEPT (snaps present) — holding auto-refresh"; timeout 15 snap refresh --hold 2>/dev/null || true
else
  echo "  → snapd: none found / unresponsive — left untouched"
fi
echo "  → journald cap: 500M"
mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nSystemMaxUse=500M\nRuntimeMaxUse=100M\n' > /etc/systemd/journald.conf.d/99-myllmbox.conf
C
)
$( [ "$REVERT" = 0 ] && [ -n "$SWAP" ] && cat <<S
# Make TOTAL swap = $SWAP by TOPPING UP: keep whatever swap already exists (e.g. Ubuntu's /swap.img) and
# size OUR /swapfile to just the difference. Non-destructive — never deletes the distro's swap, and only
# ever swapoffs OUR own idle /swapfile, so it can't push pages into a full RAM.
want=\$(numfmt --from=iec "$SWAP")
other=\$(swapon --show=NAME,SIZE --noheadings --bytes 2>/dev/null | awk '\$1!="/swapfile"{s+=\$2} END{print s+0}')
need=\$(( want - other )); og=\$(( other/1073741824 ))
if [ "\$need" -le 0 ]; then
  swapoff /swapfile 2>/dev/null || true; rm -f /swapfile; sed -i '\|^/swapfile |d' /etc/fstab 2>/dev/null || true
  echo "  → swap: requested $SWAP is below existing \${og}G (/swap.img) — removed /swapfile; minimum is \${og}G, total now \${og}G"
else
  ng=\$(( need/1073741824 )); cur=\$([ -f /swapfile ] && stat -c %s /swapfile 2>/dev/null || echo 0)
  if [ "\$cur" != "\$need" ]; then
    swapoff /swapfile 2>/dev/null || true; rm -f /swapfile
    fallocate -l "\$need" /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=\$(( need/1048576 ))
    chmod 600 /swapfile; mkswap /swapfile >/dev/null
  fi
  swapon --show=NAME --noheadings 2>/dev/null | grep -qx /swapfile || swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  echo "  → swap: existing \${og}G + /swapfile \${ng}G = $SWAP total"
fi
printf 'vm.swappiness=10\n' > /etc/sysctl.d/99-myllmbox-swap.conf; sysctl -q vm.swappiness=10
S
)
EOF
}

echo "── boxes: $BOXES   (head=$HEAD, workers=${WORKERS[*]:-none}) ──"
for b in $BOXES; do echo "── $b ──"; config_script | sudo_script "$b"; done
echo "✓ lean boot config applied."

# ── reboot into lean form (opt-in, destructive: kills the live serve; does NOT relaunch it) ───────────
if [ "$REBOOT" = 1 ] && [ "$REVERT" = 0 ]; then
  echo
  echo "→ --reboot: rebooting ${WORKERS[*]:-(no workers)} then the head ($HEAD) LAST — the live serve goes down."
  reboot_box() { sudo_script "$1" <<<'systemctl reboot'; }
  # reboot every worker (no wait/validate — they're all reachable over the LAN), then the head LAST simply
  # because THIS script runs on the head: once it reboots, the process is gone and can't reboot anyone else.
  for w in "${WORKERS[@]:-}"; do [ -n "$w" ] || continue
    echo "→ rebooting worker $w"
    if [ "$DRY" = 1 ]; then echo "  (dry-run) reboot_box $w"; else reboot_box "$w" || true; fi
  done
  echo "→ rebooting the head ($HEAD) LAST — this terminates the script."
  [ "$DRY" = 1 ] || reboot_box "$HEAD" || true
  exit 0
fi

echo "  Reboot each box when convenient — it'll come up in server form (or rerun with --reboot)."
if [ "$NOW" = 0 ] && [ "$REVERT" = 0 ]; then
  echo "  (nothing changed live; pass --now to drop the GUI session without waiting for a reboot)"
fi
exit 0