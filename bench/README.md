# bench/ — how myllmbox models are tested

Two questions, always both: **how fast is it** and **what can it actually build**. A model card
never ships a speed without a quality proof next to it, and never ships a peak without the average.

## The gauntlet — the standing quality test

The SAME two scenes are asked of every model, in the form its modality understands. The prompts are
checklists on purpose: legs, spots, nostrils, ray counts — pass/fail features you can count, so
comparing models is arithmetic, not taste.

| file | modality | output |
|---|---|---|
| `pasture-text.txt` | text/code | one self-contained animated HTML/SVG file (checklist prompt — text models READ) |
| `fish-text.txt` | text/code | one self-contained animated HTML/SVG file |
| `pasture-image.txt` | image | one HD illustration — CAPTION prompt (see below) |
| `fish-image.txt` | image | one HD illustration |
| `pasture-video.txt` | video | one clip — same caption with the motion literal |
| `fish-video.txt` | video | one clip |

**Diffusion prompts are captions, not checklists.** Text models read specs — headers, bullets,
"exactly N" — and satisfy them item by item; the `*-text.txt` prompts stay checklists, and those
checklists are also the JUDGING RUBRIC for every modality. Diffusion models (image AND video)
never read: the text encoder embeds the prompt like a training caption, so sectioned checklists
are noise, numerals have no visual anchor, and unbound repetitions of a noun become clone-pressure.
Measured on LTX-2.3: every checklist-style fish prompt variant produced 3–5 identical fish
("never a third fish" made it WORSE — negations don't bind, tokens do); the caption rewrite went
3/3 on the exact same seeds, and the pasture roster went from hybrids/duplicates to clean 4-species
casts. The caption carries the SAME countable items, woven into scene prose with every subject
individuated as a character ("a big round orange fish… a much smaller slender green fish" — never
a bare count). The image captions are the video captions minus motion/sound (motion becomes a
still cue: "caught mid-hop").

> Committed image winners generated before the caption rewrite (flux2-dev, qwen-image-edit
> `tests/`) are checklist-era — re-gauntlet them under the captions when the media slot next
> serves each model (bench debt; FLUX's missing pig+horse is the expected beneficiary).

**Protocol — best of 3:**
- Text models: 3 one-shot generations, save raw as generated (fix only syntax garbage that would
  break rendering — logic quirks stay; they're part of the result).
- Image/video models (seeded generators): 3 runs at the **canonical seeds 123123123, 456456456,
  789789789** — never the server default; a fixed default seed returns the identical image N times.
- The USER runs/picks (agents never fire test load at a serve unasked — house rule); the two
  winners are committed as `recipes/<recipe>/tests/{pasture,fish}.{html,png}` — the recipe's
  quality proof, ported to the model's page on myllmbox.com.

## Resolution tiers (image models — exact same on every model)

Speed is measured at ALL THREE standard resolutions, each as the 3-seed average:

| tier | size |
|---|---|
| HD | 1280×720 |
| FHD | 1920×1080 |
| QHD | 2560×1440 |

- **Results/showcase tier: QHD** — the committed `tests/` winners are picked from the QHD runs.
- Note: changing resolution re-rolls the composition (new noise grid), so tiers are separate
  runs, not rescales. Qwen-Image-Edit quirk: avoid exactly 1024×1024 (the one known trap size —
  breaks only when output==input size AND a dim is exactly 1024).

## Video tiers (video models — exact same on every model)

TWO tiers (no QHD — a 5s QHD clip is where time and memory explode, with no showcase payoff);
frames constant across tiers: **121 frames @ 24fps (~5s)**, speed = the 3-seed average per tier.
LTX-class pipelines snap dims to /64, so the tiers land on slightly non-standard numbers:

| tier | size (snapped /64) |
|---|---|
| HD | 1280×704 |
| FHD | 1920×1088 |

- **HD is the canonical spec** (the July-verified size). **FHD is the showcase tier** — committed
  `tests/{pasture,fish}.mp4` winners come from FHD *once a first FHD run proves it fits in memory*
  (the 121-frame VAE decode is the peak); until then HD winners stand.
- Audio, when the model makes it, is part of the judgment (the prompts carry a SOUND section).

## Tools

- **`gauntlet-image.py`** — the image gauntlet in one command:
  `./bench/gauntlet-image.py <ip:port|url> [--mode generate] [--size 1280x720] [--token …]`
  Runs both prompts × the canonical seeds, saves to `results/<name>/` with a `run.json` of
  timings. `--mode edit` (default) posts a generated blank-canvas reference to
  `/v1/images/edits`; `--mode generate` posts JSON to `/v1/images/generations`. Stdlib-only;
  run it on the serving box or through the tunnel (the management LAN blocks raw ports).
- **`gauntlet-video.py`** — same idea for `/v1/videos` servers:
  `./bench/gauntlet-video.py <ip:port|url> [--size 1280x704] [--frames 121] [--fps 24] [--token …]`
  Both video prompts × the canonical seeds at the canonical clip spec, saves mp4s +
  `run.json` to `results/<name>/`.
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
