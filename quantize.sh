#!/usr/bin/env bash
# quantize.sh <recipe> [--to <format>]  —  offline-quantize a recipe's source model into
# models/myllmbox/<name>-<format> using the shared quantizer image (built once from quantizer/).
# Then `./run.sh <recipe>` serves that pre-quant INSTANTLY. The output is a shareable checkpoint.
#
#   ./quantize.sh flux2-dev              # uses the recipe's default format (quantize.format)
#   ./quantize.sh flux2-dev --to mxfp8   # override; must be in the recipe's quantize.formats
#
# All params live in the recipe's `quantize:` block (source, model_type, target, skip_modules, formats).
set -euo pipefail
cd "$(dirname "$0")"

R="${1:-}"
if [ -z "$R" ]; then echo "usage: ./quantize.sh <recipe> [--to nvfp4|mxfp8]"; echo "recipes: $(ls recipes 2>/dev/null | tr '\n' ' ')"; exit 1; fi
shift
FMT=""
while [ $# -gt 0 ]; do case "$1" in --to) FMT="${2:-}"; shift 2;; *) echo "unknown arg: $1"; exit 1;; esac; done
D="recipes/$R"
[ -f "$D/myllmbox.yaml" ] || { echo "no recipe: $D/myllmbox.yaml (see: ls recipes/)"; exit 1; }

# venv (shared with run.sh) just to read the recipe yaml
V=.venv
[ -x "$V/bin/python" ] || python3 -m venv "$V"
"$V/bin/python" -m pip install -q -U pip pyyaml >/dev/null

# pull the quantize: block; validate --to against the recipe's allowed formats
eval "$("$V/bin/python" - "$D/myllmbox.yaml" "$FMT" <<'PY'
import sys, yaml, shlex
cfg = yaml.safe_load(open(sys.argv[1])) or {}
q = cfg.get("quantize") or {}
req = sys.argv[2].strip().lower()
fmt = req or (q.get("format") or "nvfp4")
allowed = [str(x).lower() for x in (q.get("formats") or [])]
if allowed and fmt not in allowed:
    sys.exit(f"[quantize] format '{fmt}' not supported by this recipe (allowed: {', '.join(allowed)})")
vals = dict(
    SRC=q.get("source") or "", FMT=fmt, MTYPE=q.get("model_type") or "diffusion",
    TARGET=q.get("target") or "transformer", SKIP=q.get("skip_modules") or "",
    OUT=q.get("out") or "", MODELS=(cfg.get("server") or cfg.get("vllm") or {}).get("models_dir") or "models",
)
for k, v in vals.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)"
[ -n "${SRC:-}" ] || { echo "recipe '$R' has no quantize.source (BF16 model dir)"; exit 1; }

# ensure the BF16 source is here — quantize needs it; download it if missing (self-sufficient).
SRC_ID="${SRC#/models/}"
if [ ! -d "$MODELS/$SRC_ID" ] || [ -z "$(ls -A "$MODELS/$SRC_ID" 2>/dev/null)" ]; then
  echo "· source $SRC_ID not present → downloading"
  ./download.sh "$SRC_ID"
fi

IMG=mbx-quantizer
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "· building the quantizer image ($IMG) — first time only, nightly stack, slow"
  docker build -t "$IMG" quantizer
fi

MODELS_ABS="$(cd "$MODELS" && pwd)"
NAME="$(basename "$SRC")"
OUT="${OUT:-/models/myllmbox/${NAME}-${FMT}}"
echo "· quantize  $SRC  →  ${MODELS}/myllmbox/${NAME}-${FMT}   (format=$FMT type=$MTYPE target=$TARGET)"
# HOST_UID/HOST_GID: the container runs as root, so anything it writes to the /models mount lands root-owned
# and you'd need root to delete it. The quantizer chown's its output back to you at the end (see quantizer/__main__.py).
docker run --rm --gpus all --ipc=host -v "$MODELS_ABS":/models \
  -e MODEL_PATH="$SRC" -e QUANT_OUT="$OUT" -e QUANT_FORMAT="$FMT" \
  -e QUANT_MODEL_TYPE="$MTYPE" -e QUANT_TARGET="$TARGET" -e QUANT_SKIP_MODULES="$SKIP" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  "$IMG" --to "$FMT"
echo "· done  →  ${MODELS}/myllmbox/${NAME}-${FMT}    (serve it: ./run.sh $R)"
