"""Entry point run inside mbx-quantizer:  python -m quantizer --to <format>

Everything comes from the recipe's `quantize:` block, passed as env by ./quantize.sh:
  MODEL_PATH (source BF16) · QUANT_OUT (dest) · QUANT_FORMAT · QUANT_MODEL_TYPE (diffusion|llm|video) ·
  QUANT_TARGET (submodule to quantize) · QUANT_SKIP_MODULES (comma-sep, kept BF16)
Writes a self-contained checkpoint to QUANT_OUT (default models/myllmbox/<name>-<format>).
"""
from __future__ import annotations

import argparse
import os
import sys

# format → backend module. torchao covers nvfp4/mxfp8 for diffusion/llm/video; gguf & w4a16 are
# separate backend images (added as they land) — an unknown --to fails loud with what IS available.
from . import torchao as torchao_backend

BACKENDS = {"nvfp4": torchao_backend, "mxfp8": torchao_backend}


def main() -> int:
    ap = argparse.ArgumentParser(prog="quantizer")
    ap.add_argument("--to", default=os.environ.get("QUANT_FORMAT", "nvfp4"))
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--out", default=os.environ.get("QUANT_OUT"))
    ap.add_argument("--type", default=os.environ.get("QUANT_MODEL_TYPE", "diffusion"))
    ap.add_argument("--target", default=os.environ.get("QUANT_TARGET", "transformer"))
    a = ap.parse_args()

    fmt = a.to.lower()
    if not a.model:
        raise SystemExit("[quantizer] MODEL_PATH/--model (source BF16 dir) is required")
    out = a.out or f"/models/myllmbox/{os.path.basename(a.model.rstrip('/'))}-{fmt}"
    backend = BACKENDS.get(fmt)
    if backend is None:
        raise SystemExit(
            f"[quantizer] format '{fmt}' has no backend in this image (have: {', '.join(BACKENDS)}). "
            "gguf (llama.cpp / ComfyUI-gguf) and w4a16 (llm-compressor) are separate quantizer images.")
    skip = [s.strip() for s in os.environ.get("QUANT_SKIP_MODULES", "").split(",") if s.strip()]

    print(f"[quantizer] {a.model}  --to {fmt}  --type {a.type}  --target {a.target}  "
          f"--out {out}  skip={skip or 'none (full)'}", flush=True)
    backend.quantize(model=a.model, out=out, fmt=fmt, model_type=a.type, target=a.target, skip=skip)

    # Build manifest — travels with the checkpoint (shareable, "move to another server"). Records the exact
    # stack that wrote it so a serve box can PRE-FLIGHT compatibility before loading: torchao serializes its
    # FP4 weights as versioned tensor subclasses, so a torchao/torch mismatch on load fails cryptically.
    import json
    import torch
    import torchao
    manifest = {
        "format": fmt, "backend": "torchao", "model_type": a.type, "target": a.target,
        "source": a.model, "skip_modules": skip,
        "torch": torch.__version__, "torchao": torchao.__version__,
    }
    with open(os.path.join(out, "myllmbox-quant.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # The container runs as root, so everything above landed root-owned on the /models mount — you'd need
    # root to delete it. Hand the output (and its myllmbox/ parent, so the dir entry itself is deletable)
    # back to the invoking host user. Best-effort: skip if HOST_UID unset or not running as root.
    huid, hgid = os.environ.get("HOST_UID"), os.environ.get("HOST_GID")
    if huid and hgid and os.geteuid() == 0:
        uid, gid = int(huid), int(hgid)
        os.chown(os.path.dirname(out.rstrip("/")), uid, gid)
        for root, dirs, files in os.walk(out):
            os.chown(root, uid, gid)
            for n in dirs + files:
                os.chown(os.path.join(root, n), uid, gid)
        print(f"[quantizer] chowned output to {uid}:{gid} (host user)", flush=True)

    print(f"[quantizer] DONE → {out}  (manifest: torchao {torchao.__version__} · torch {torch.__version__})", flush=True)
    print(f"[quantizer] serve with MODEL_PATH={out}, QUANT_PRELOADED=1", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
