# bench/ — how myllmbox models are tested

Two questions, always both: **how fast is it** and **what can it actually build**. A model card
never ships a speed without a quality proof next to it, and never ships a peak without the average.

## The gauntlet — the standing quality test

The SAME two scenes are asked of every model, in the form its modality understands. The prompts are
checklists on purpose: legs, spots, nostrils, ray counts — pass/fail features you can count, so
comparing models is arithmetic, not taste.

| file | modality | output |
|---|---|---|
| `pasture-text.txt` | text/code | one self-contained animated HTML/SVG file |
| `fish-text.txt` | text/code | one self-contained animated HTML/SVG file |
| `pasture-image.txt` | image | one HD illustration (same scene, same checklist) |
| `fish-image.txt` | image | one HD illustration |

The same scene specs extend to video models later (the MOTION sections become literal).

**Protocol — best of 3:**
- Text models: 3 one-shot generations, save raw as generated (fix only syntax garbage that would
  break rendering — logic quirks stay; they're part of the result).
- Image/video models (seeded generators): 3 runs at the **canonical seeds 123123123, 456456456,
  789789789** — never the server default; a fixed default seed returns the identical image N times.
- The USER runs/picks (agents never fire test load at a serve unasked — house rule); the two
  winners are committed as `recipes/<recipe>/tests/{pasture,fish}.{html,png}` — the recipe's
  quality proof, ported to the model's page on myllmbox.com.

## Resolution tiers (image models — exact same on every model)

- **Speed tier: 1280×720.** The quoted seconds-per-image = the 3-seed average at 720p.
- **Results tier: 2560×1440 (QHD).** The showcase renders / committed winners.
- Note: changing resolution re-rolls the composition (new noise grid), so the tiers are separate
  runs, not rescales. Qwen-Image-Edit quirk: avoid exactly 1024×1024 (the one known trap size —
  breaks only when output==input size AND a dim is exactly 1024).

## Tools

- **`gauntlet-image.py`** — the image gauntlet in one command:
  `./bench/gauntlet-image.py <ip:port|url> [--mode generate] [--size 1280x720] [--token …]`
  Runs both prompts × the canonical seeds, saves to `results/<name>/` with a `run.json` of
  timings. `--mode edit` (default) posts a generated blank-canvas reference to
  `/v1/images/edits`; `--mode generate` posts JSON to `/v1/images/generations`. Stdlib-only;
  run it on the serving box or through the tunnel (the management LAN blocks raw ports).
- **`sweep.py`** — the LLM concurrency harness: one fixed prompt (+ nonce to defeat prefix
  caching) at increasing concurrency, client-side aggregate tok/s + acceptance from /metrics,
  ready-to-paste reports.md rows.

## LLM speed methodology (text models)

- **Decode** numbers come from clean 10-second engine windows: single stream, zero prompt
  throughput in-window. Report the sustained band AND the session average — never a bare peak.
- **steps/s = drafted throughput ÷ K** — the pure engine speed, acceptance-independent; judge
  kernel/config changes by this, never by generation tok/s (accept-confounded).
- **Acceptance length** judges content/checkpoint quality (prose sags are often inherent MTP
  behavior — verify against a second box on the SAME prompt before blaming a quantization).
- **Cold prefill ≠ warm ingest** — label which one a number is (warm = prefix-cache replay).

## `results/` (gitignored)

Everything the tools produce lands in `results/<name>/` — scratch by design. Only the picked
winners graduate into `recipes/<recipe>/tests/`, and only measured numbers graduate onto model
cards and READMEs.
