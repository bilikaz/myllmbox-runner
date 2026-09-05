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

> Caption re-gauntlet DONE 2026-08-31 for both image models: FLUX's missing pig+horse came back
> (roster 3/3 seeds), qwen's motion cue now renders (all four animals mid-bounce on the winning
> seed). Standing cross-model datapoint: fish swim DIRECTIONS — qwen 3/3 correct, FLUX 0/3.

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
### LLM speed: `test.py` → `summary.py`

Two tools, one flow: **`test.py` measures** (one rung or a ladder, one JSON per run), **`summary.py` pools**
(every run of a model → one table). Both are stdlib-only; run them ON the serving box (the management LAN
blocks raw ports) or through the proxy with `--token`. The served model is auto-detected from `/v1/models`.

**`test.py` — hold a concurrency, measure the steady state.**
```
./bench/test.py --c 8 --thinking off --prompt pasture                 # one rung
./bench/test.py --c 1,1,1,2,4,8,16,32,64 --thinking off --prompt pasture --seconds 400 --warmup 40   # a ladder
```
- Fires exactly C streams ONCE, together, with a big token budget (`--max-tokens`, default 32768). NO re-firing:
  a fresh request mid-window drags a prefill into the decode numbers.
- Samples the engine's `/metrics` every 10 s. A sample is **steady** only when all C streams are running AND the
  prompt-token counter did not move (pure decode). Everything else is printed but not counted.
- The window = `--warmup` (30 s, covers the C prefills) + `--seconds` measured (600 s default), or until the first
  stream finishes, whichever comes first. Either way the window's end ABORTS every remaining stream (requests are
  streamed; closing them makes vLLM drop them) so the next rung starts at once — nobody waits for stragglers.
  `--seconds 0` = no cap.
- A ladder (`--c 8,16,32`) runs rung after rung; each rung drains fully before the next fires. Repeats
  (`--c 1,1,1`) are fine — repeatability for free.
- `--thinking on|off` → the model's own `enable_thinking` switch. `--prompt code` (built-in LRU-cache task,
  short), `pasture` / `fish` (the gauntlet texts — long, hold a rung for minutes), or any file path.
- Output: live line per sample; a summary of avg (min–max) over the steady samples for gen tok/s, per-stream,
  **engine steps/s** (ms/step), acceptance, running, KV%; the watched period; a reports.md row. Every run is
  saved as **`results/<model>/<timestamp>-c<N>.json`** — `params` first, then `summary`, `samples`, `requests`.

**`summary.py` — pool the runs into a ladder table.**
```
./bench/summary.py                                              # one model folder → picked automatically
./bench/summary.py --model Qwen/Qwen3.8-Flash-Next --start 2026-09-05 --prompt pasture      # code + thinking + average
```
- Groups the selected runs by rung c, pools the STEADY samples of all runs in a group, and prints: runs · samples ·
  gen tok/s · per-stream · engine steps/s · ms/step · acc len · KV% — each avg (min–max) — plus finished/aborted
  request counts. Markdown, ready to paste into a recipe's reports.md.
- **Bands:** thinking OFF runs = the **code** band, thinking ON = the **thinking** band. `--thinking off|on` prints
  one band with every metric; `--thinking both` (default) prints ONE combined table, a row per rung: code tok/s +
  per-stream · thinking tok/s + per-stream · **AVERAGE** tok/s = (code band avg + thinking band avg) / 2 — arithmetic,
  NOT a pooled sample mean, so the band with more samples does not pull the result — and AVERAGE per-stream = that / c,
  plus steps/s and acceptance for both bands. Runs/samples/KV live only in the single-band tables.
- `--start` / `--end` take `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` (a bare `--end` date = the whole day; default now).
  `--json <path>` dumps the same numbers.

**Reading the numbers (the two that matter):**
- **engine steps/s** = drafts ÷ running requests per second — the raw engine speed, independent of what the model
  is writing. Judge kernels, TP, K, images, boxes by THIS. (The `/metrics` drafts counter is per sequence per
  step; test.py and summary.py divide it back out — the engine's own log line `Drafted throughput ÷ K` at c=1
  is the same number.)
- **acceptance length** = tokens per step per sequence — set by the content and the checkpoint, capped at K+1.
  gen tok/s = steps × acceptance × running: the same engine reads 45 tok/s on prose and 70 on code.

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
