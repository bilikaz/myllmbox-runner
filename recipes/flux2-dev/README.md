# flux2-dev — FLUX.2-dev (NVFP4) image server

Text→image (and reference-image edit) on a single DGX Spark, served as a pre-quantized NVFP4
checkpoint (autodetected via `myllmbox-quant.json`), block-compile always on (JIT persisted in
`.data/cache/inductor` — each new output size compiles once per cache lifetime).

```
./run.sh flux2-dev        # quantizes the BF16 base on first ever run, then serves
```

## Measured (2026-08-30, GB10, 28 steps, avg of the 3 canonical seeds, warm)

| tier | size | seconds / image |
|---|---|---|
| HD | 1280×720 | **~43s** |
| FHD | 1920×1080 | **~108s** |
| QHD | 2560×1440 | **~389s** |

Compile-off (the old shipped default) was 1.8× slower across the board — the knob is gone now,
compile is unconditional. Protocol: `bench/README.md` (gauntlet, canonical seeds, QHD = showcase
tier — winners in `tests/` once picked). Memory: ~66GB vs ~112GB BF16.
