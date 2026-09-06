#!/usr/bin/env python3
"""Benchmark-card banner generator (myllmbox house style, Spark-Arena-inspired layout).

Pure PIL — deterministic typography, exact numbers, no diffusion-model spelling lottery.
Every number on a banner must be MEASURED (clean decode windows); see scripts/README.md.

Usage (defaults reproduce the Qwen3.8-Flash-Next banner):
  python3 scripts/benchmark-banner.py \
    --org Qwen --title QWEN3.8-FLASH-NEXT \
    --chip vLLM --chip "hibrid45 · NVFP4+bf16 attn" --chip "1 node" \
    --stat "55:tok/s:DECODE · CODE C1:green" \
    --stat "32.6:tok/s:DECODE · MIXED AVG:white" \
    --stat "3.0K:tok/s:INGEST · WARM PREFILL:orange" \
    --stat "800K:tok:KV POOL · 262K CTX:white" \
    --hardware "NVIDIA DGX Spark · GB10 × 1" \
    --foot "one command: ./run.sh qwen38-flash-next · clean 10s decode windows, single stream · code band 50–57 · quality-gated" \
    --out qwen38-benchmark-banner.png
"""
import argparse
import io
import os
import urllib.request

from PIL import Image, ImageDraw, ImageFont

BG, CARD, SUBTLE = (13, 15, 18), (20, 23, 28), (25, 29, 35)
BORDER, TEXT, SEC = (36, 42, 50), (233, 236, 239), (152, 161, 173)
COLORS = {"green": (195, 232, 141), "orange": (255, 147, 80), "white": TEXT}
LOGO_URL = "https://myllmbox.com/brand/logo-512.png"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans{}.ttf"


def font(sz: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT.format("-Bold" if bold else ""), sz)


def get_logo(path: str | None) -> Image.Image:
    if path and os.path.exists(path):
        return Image.open(path).convert("RGBA")
    with urllib.request.urlopen(LOGO_URL) as r:  # noqa: S310
        return Image.open(io.BytesIO(r.read())).convert("RGBA")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="Qwen")
    ap.add_argument("--title", default="QWEN3.8-FLASH-NEXT")
    ap.add_argument("--chip", action="append", default=None, help="repeatable")
    ap.add_argument("--stat", action="append", default=None,
                    help="repeatable, NUM:UNIT:LABEL:COLOR (color: green|orange|white); exactly 4 look best")
    ap.add_argument("--hardware", default="NVIDIA DGX Spark · GB10 × 1")
    ap.add_argument("--link", default="github.com/bilikaz/myllmbox-runner ↗")
    ap.add_argument("--subtitle", default="GB10 Spark · measured serve")
    ap.add_argument("--site", default="myllmbox.com")
    ap.add_argument("--foot", default="clean 10s decode windows, single stream · quality-gated")
    ap.add_argument("--logo", default=None, help="local logo path (default: fetch the brand logo)")
    ap.add_argument("--out", default="banner.png")
    a = ap.parse_args()
    chips = a.chip or ["vLLM", "1 node"]
    stats = [s.split(":") for s in (a.stat or ["?:tok/s:DECODE:green"])]

    W, H = 1800, 1000
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([40, 40, W - 40, H - 40], radius=28, fill=CARD, outline=BORDER, width=2)

    logo = get_logo(a.logo).resize((72, 72))
    img.paste(logo, (90, 84), logo)
    d.text((180, 96), "MY LLM BOX", font=font(34), fill=COLORS["green"])
    d.text((470, 104), a.subtitle, font=font(24, False), fill=SEC)
    d.text((W - 90 - d.textlength(a.site, font=font(24, False)), 104), a.site, font=font(24, False), fill=SEC)

    d.text((90, 200), f"{a.org} /", font=font(30, False), fill=SEC)
    d.text((90, 240), a.title, font=font(84), fill=TEXT)

    x = 90
    for c in chips:
        w = d.textlength(c, font=font(24)) + 44
        d.rounded_rectangle([x, 370, x + w, 424], radius=27, fill=SUBTLE, outline=BORDER, width=2)
        d.text((x + 22, 384), c, font=font(24), fill=TEXT)
        x += w + 18

    n = len(stats)
    bw, bh, gap, y0 = (W - 180 - (n - 1) * 24) // n, 220, 24, 470
    for i, (num, unit, label, col) in enumerate(stats):
        x0 = 90 + i * (bw + gap)
        d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=18, fill=SUBTLE, outline=BORDER, width=2)
        d.text((x0 + 30, y0 + 42), num, font=font(72), fill=COLORS.get(col, TEXT))
        d.text((x0 + 30 + d.textlength(num, font=font(72)) + 8, y0 + 76), f" {unit}", font=font(28, False), fill=SEC)
        d.text((x0 + 30, y0 + 150), label, font=font(22), fill=SEC)

    d.rounded_rectangle([90, 740, W - 90, 830], radius=18, fill=SUBTLE, outline=BORDER, width=2)
    d.text((120, 758), "HARDWARE", font=font(20), fill=SEC)
    d.text((120, 786), a.hardware, font=font(26), fill=TEXT)
    d.text((W - 120 - d.textlength(a.link, font=font(24)), 782), a.link, font=font(24), fill=COLORS["green"])

    d.text((90, 880), a.foot, font=font(22, False), fill=SEC)

    img.save(a.out)
    print(f"saved {a.out} ({W}x{H})")


if __name__ == "__main__":
    main()
