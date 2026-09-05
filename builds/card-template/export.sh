#!/usr/bin/env bash
# export.sh <card.html> <spark-photo.png> <out.png> — embed the photo, make the borderless export variant, render
# with Windows Chrome headless (from WSL), crop to content. Mirrors the workflow in README.md.
set -euo pipefail
CARD=$1; PHOTO=$2; OUT=$3
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
python3 - "$CARD" "$PHOTO" "$S/export.html" <<'PY'
import sys, base64
card, photo, out = sys.argv[1:4]
s = open(card).read()
b64 = base64.b64encode(open(photo, "rb").read()).decode()
s = s.replace("__SPARK_PNG__", "data:image/png;base64," + b64)
# export variant: no page padding, fixed card width, square corners
s = s.replace("min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px}", "padding:0}")
s = s.replace(".card{width:min(1500px,100%);", ".card{width:1500px;").replace("border-radius:18px;padding:44px 52px 36px;", "border-radius:0;padding:44px 52px 36px;")
open(out, "w").write(s)
PY
# renderer: a local Linux Chromium (Playwright's cache) if present — no Windows interop needed; else Windows Chrome.
LOCAL=$(find ~/.cache/ms-playwright -type f \( -name chrome -o -name headless_shell \) 2>/dev/null | sort | tail -1)
if [ -n "$LOCAL" ]; then
  "$LOCAL" --headless --no-sandbox --disable-gpu --hide-scrollbars --screenshot="$S/shot.png" \
    --window-size=1500,1400 --virtual-time-budget=8000 "file://$S/export.html" >/dev/null 2>&1
else
  WIN=$(wslpath -w "$S/export.html")
  "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless --disable-gpu --hide-scrollbars \
    --screenshot="$(wslpath -w "$S/shot.png")" --window-size=1500,1400 --virtual-time-budget=8000 "file:///$WIN" 2>/dev/null
fi
[ -s "$S/shot.png" ] || { echo "✗ render produced nothing (no Chromium found or it failed)"; exit 1; }
python3 - "$S/shot.png" "$OUT" <<'PY'
import sys
from PIL import Image
im = Image.open(sys.argv[1]).convert("RGB"); w, h = im.size; bg = im.getpixel((w - 1, h - 1))
last = max(y for y in range(h) if any(im.getpixel((x, y)) != bg for x in range(0, w, 8)))
im.crop((0, 0, w, min(h, last + 44))).save(sys.argv[2]); print("saved", sys.argv[2], im.size, "->", (w, min(h, last + 44)))
PY
