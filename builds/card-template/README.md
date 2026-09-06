# Release share-card template

The myllmbox model-release card (first used: Qwen3.8-Flash-Next hibrid46, 2026-09-02):
header/brand bar, model title + kicker, chip row, 3 stat dials (one green screamer, one orange,
one white), product photo with the red ×1 quantity badge, concurrency-ladder chart (data-driven
from the LADDER array — set per-rung values, pending:true renders hollow), hardware bar + links.

## To make a new card
1. Copy `card-template.html`, edit: title, kicker, chips, dial values/labels, LADDER data,
   hardware line, links. Photo: replace `__SPARK_PNG__` with a base64 data URI
   — the photo lives here as `spark.png` (the rendered DGX Spark box, used on every card); export.sh embeds it.
2. Preview: publish as an artifact or open locally.
3. Export PNG: `builds/card-template/export.sh <card.html> <spark-photo.png> <out.png>` — embeds the photo,
   makes the borderless export variant (body padding 0, fixed 1500px card, radius 0), renders with a local
   headless Chromium (Playwright's cache) or Windows Chrome, crops exactly at the card's bottom edge (PIL; no page ground below).
   Cards made so far: `builds/qwen38-flash-next/cluster/card-qwen38-cluster.html` (2× Spark, 2026-09-05).

House rules that shaped it (builds/instructions.md applies): numbers measured with conditions,
one green number per card, no internal names, chart points only from reproducible configs,
vendor casing for model names, ×1 badge = quantity (bottom of photo), no footer clutter.

## Also in this folder
- `benchmark-banner.py` — the earlier pure-PIL stat card (deterministic typography; the HTML template above replaced
  it for releases, kept as the no-browser fallback). `announcement-art.md` — its usage plus the illustrated-poster
  workflow on the qwen-image-edit serve ("the model paints, the code types").

