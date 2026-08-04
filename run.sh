#!/usr/bin/env bash
# run.sh <recipe>  —  build the recipe's image, ensure its model exists, run it, front it with the
# keepalive proxy + cloudflared tunnel. Self-contained: bootstraps its own venv — you never activate anything.
#
# The model pipeline is AUTOMATIC:  present? serve it.  ·  a myllmbox/ quant that isn't built? quantize it
# (downloading its BF16 source first).  ·  a plain HF model that's missing? download it.  Then serve.
#
#   ./run.sh holo-3.1-35b     # plain vLLM model — downloads if missing
#   ./run.sh flux2-dev        # myllmbox quant — downloads FLUX.2-dev, quantizes to NVFP4, serves
#
set -euo pipefail
cd "$(dirname "$0")"

R="${1:-}"
if [ -z "$R" ]; then
  echo "usage: ./run.sh <recipe>"
  echo "recipes: $(ls recipes 2>/dev/null | tr '\n' ' ')"
  exit 1
fi
D="recipes/$R"
[ -f "$D/myllmbox.yaml" ] || { echo "no recipe: $D/myllmbox.yaml (see: ls recipes/)"; exit 1; }

# 1. venv — all the python/venv/pip nonsense lives in here, not on you. Created once, reused after.
V=.venv
[ -x "$V/bin/python" ] || python3 -m venv "$V"
"$V/bin/python" -m pip install -q -U pip aiohttp pyyaml click python-dotenv >/dev/null

# 2. build the base box once (root Dockerfile → mbx-base). Every recipe FROMs it — built once, shared.
if ! docker image inspect mbx-base:latest >/dev/null 2>&1; then
  echo "· building base box (./Dockerfile)  →  mbx-base"
  docker build -t mbx-base:latest -f Dockerfile .
fi

# 3. build THIS recipe on top of the base (recipes/<recipe>/Dockerfile → mbx-<recipe>)
if [ -f "$D/Dockerfile" ]; then
  echo "· building recipes/$R  →  mbx-$R"
  docker build -t "mbx-$R" "$D"
fi

# 3b. resolve the model — the pipeline is automatic:  have it? serve. missing myllmbox/ quant? quantize it
#     (downloading its BF16 source first). missing plain HF model? download it. THEN serve.
eval "$("$V/bin/python" - "$D/myllmbox.yaml" <<'PY'
import sys, os, yaml, shlex
cfg = yaml.safe_load(open(sys.argv[1])) or {}
s, q = cfg.get("server") or {}, cfg.get("quantize") or {}
c = cfg.get("cluster") or {}
# Resolve worker ssh-targets for the weight copy. Use the INTERCONNECT address (fast link) + ssh_user; node 0
# = head (this box) so it's dropped. `boxes:` resolves from cluster.yaml; legacy `nodes:` are interconnect IPs.
targets, boxes = [], c.get("boxes") or []
if boxes:
    spec = {}
    if os.path.exists("cluster.yaml"):
        spec = (yaml.safe_load(open("cluster.yaml")) or {}).get("boxes") or {}
    for name in boxes:
        b = spec.get(name) or {}
        addr, user = (b.get("interconnect") or b.get("host") or ""), (b.get("ssh_user") or "")
        targets.append(f"{user}@{addr}" if user else addr)
else:
    user = c.get("ssh_user") or ""
    targets = [f"{user}@{ip}" if user else str(ip) for ip in (c.get("nodes") or [])]
for k, val in dict(MODEL=s.get("model") or "", MODELS_DIR=s.get("models_dir") or "models",
                   QSRC=q.get("source") or "",
                   WORKERS=" ".join(targets[1:])).items():   # drop head; the rest get the weights
    print(f"{k}={shlex.quote(str(val))}")
PY
)"
strip() { echo "${1#/models/}"; }   # /models/org/name → org/name (the ./download.sh hf-id / host subpath)
if [ -n "$MODEL" ]; then
  HOSTPATH="$MODELS_DIR/$(strip "$MODEL")"
  if [ -d "$HOSTPATH" ] && [ -n "$(ls -A "$HOSTPATH" 2>/dev/null)" ]; then
    echo "· model present: $HOSTPATH"
  elif [[ "$MODEL" == */myllmbox/* ]]; then
    echo "· $MODEL is a myllmbox quant and isn't built yet → quantizing (quantize.sh fetches its source)"
    ./quantize.sh "$R"          # self-sufficient: downloads the BF16 source if missing, then quantizes
  else
    echo "· model not downloaded yet → fetching $(strip "$MODEL")"
    ./download.sh "$(strip "$MODEL")"
  fi

  # 3b-cluster: on a multi-node recipe the WORKERS need the same model at the same absolute path (they don't
  # download/quantize — only the head does, then we mirror it here). build-and-copy ships the IMAGE; this
  # ships the WEIGHTS. WORKERS = worker ssh-targets over the interconnect (head already dropped, see reader).
  if [ -n "${WORKERS:-}" ]; then
    MODELS_ABS="$(cd "$MODELS_DIR" 2>/dev/null && pwd || echo "$PWD/$MODELS_DIR")"
    SRC="$MODELS_ABS/$(strip "$MODEL")"                    # same absolute path the runner mounts (-v …:/models)
    for tgt in $WORKERS; do
        echo "· syncing model to worker $tgt  ($SRC)"
        # --size-only: model shards are immutable, so match by SIZE — never re-transfer 155GB just because the
        # worker's copy has a different mtime (independent downloads always do). No --partial: an interrupted
        # copy must NOT leave a half-written shard at the real path for the loader to choke on (rsync's temp
        # file is discarded on failure). Result: instant no-op when already present, clean re-copy when not.
        ssh -o BatchMode=yes "$tgt" "mkdir -p '$(dirname "$SRC")'" \
          && rsync -a --size-only --info=progress2 -e "ssh -o BatchMode=yes" "$SRC/" "$tgt:$SRC/" \
          || { echo "✗ could not sync model to worker $tgt — the cluster serve will fail without it"; exit 1; }
    done
  fi
fi

# 3c. pre-flight a myllmbox torchao quant against the serve image. torchao serializes FP4 weights as versioned
#     tensor subclasses — a torchao/torch mismatch between the box that quantized and this box's image fails
#     cryptically ("Unable to import torchao Tensor objects") and hangs. The quant carries myllmbox-quant.json;
#     we compare it to the image's actual versions and REFUSE to load on mismatch (don't even try — inform).
MANIFEST="$MODELS_DIR/$(strip "$MODEL")/myllmbox-quant.json"
if [ -n "$MODEL" ] && [ -f "$MANIFEST" ] && docker image inspect "mbx-$R" >/dev/null 2>&1; then
  read -r Q_TORCH Q_AO Q_BACKEND < <("$V/bin/python" - "$MANIFEST" <<'PY'
import sys, json
m = json.load(open(sys.argv[1]))
print(m.get("torch", ""), m.get("torchao", ""), m.get("backend", ""))
PY
)
  if [ "$Q_BACKEND" = "torchao" ]; then
    IMG_VERS="$(docker run --rm --entrypoint python3 "mbx-$R" -c \
      'import torch,torchao;print(torch.__version__,torchao.__version__)' 2>/dev/null || true)"
    I_TORCH="${IMG_VERS%% *}"; I_AO="${IMG_VERS##* }"
    if [ -z "$IMG_VERS" ]; then
      echo "· ⚠ could not read torch/torchao from mbx-$R (skipping quant pre-flight)"
    elif [ "$Q_TORCH" != "$I_TORCH" ] || [ "$Q_AO" != "$I_AO" ]; then
      echo "✗ QUANT ↔ IMAGE MISMATCH — refusing to load (torchao FP4 would fail to deserialize):"
      echo "    quant built with:  torch $Q_TORCH · torchao $Q_AO   ($MANIFEST)"
      echo "    serve image mbx-$R: torch $I_TORCH · torchao $I_AO"
      echo "  → rebuild mbx-$R to match (recipes/$R/Dockerfile pins), or re-quantize on this box."
      exit 1
    else
      echo "· quant ↔ image match (torch $I_TORCH · torchao $I_AO)"
    fi
  fi
fi

# 4. up: model container + keepalive proxy + cloudflared tunnel. Config = the recipe's myllmbox.yaml.
echo "· starting box (model → proxy → tunnel)"
exec "$V/bin/python" -m runner.cli up --config "$D/myllmbox.yaml"
