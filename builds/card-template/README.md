# Release share-card template

The myllmbox model-release card (first used: Qwen3.8-Flash-Next hibrid46, 2026-09-02):
header/brand bar, model title + kicker, chip row, 3 stat dials (one green screamer, one orange,
one white), product photo with the red ×1 quantity badge, concurrency-ladder chart (data-driven
from the LADDER array — set per-rung values, pending:true renders hollow), hardware bar + links.

## To make a new card
1. Copy `card-template.html`, edit: title, kicker, chips, dial values/labels, LADDER data,
   hardware line, links. Photo: replace `__SPARK_PNG__` with a base64 data URI
   (`python3 -c "import base64; print(base64.b64encode(open('photo.png','rb').read()).decode())"`).
2. Preview: publish as an artifact or open locally.
3. Export PNG (WSL, borderless): make the export variant (body padding 0, card width fixed,
   radius 0 — see git history of this file for the two sed replacements), render with Windows
   Chrome headless:
   `chrome.exe --headless --screenshot=out.png --window-size=1500,1400 --virtual-time-budget=8000 file.html`
   then crop to content bottom + ~44px padding (PIL scan for last non-background row).

House rules that shaped it (builds/instructions.md applies): numbers measured with conditions,
one green number per card, no internal names, chart points only from reproducible configs,
vendor casing for model names, ×1 badge = quantity (bottom of photo), no footer clutter.
