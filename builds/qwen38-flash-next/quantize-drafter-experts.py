#!/usr/bin/env python3
"""quantize-drafter-experts.py — the MTP drafter's routed experts: fused bf16 (transformers ≥5.16 layout) → per-expert NVFP4
in compressed-tensors `nvfp4-pack-quantized` form, exactly the body's format, so vLLM's compressed-tensors MoE method loads
the drafter the same way it loads the body.

Why: orcarouter's checkpoint stores `mtp.layers.0.mlp.experts.gate_up_proj` [E, 2I, H] and `…experts.down_proj` [E, H, I]
as bf16 — 4.7 GiB, and a fused 3-D layout the vendor image's MoE loader has no code path for. Per-expert NVFP4 is
1.3 GiB and the layout the body already uses (`experts.{e}.{gate,up,down}_proj.{weight_packed,weight_scale,weight_global_scale}`).
Their config's `group_1` targets (`re:.*mlp\\.experts\\..*proj$`) already cover the drafter's experts, and `ignore` lists
none of them, so no config change: after this the checkpoint matches its own quantization_config.

Quantization = the compressed-tensors library's own functions (generate_gparam / calculate_qparams / quantize /
pack_fp4_to_uint8) with the body's QuantizationArgs (4-bit float, symmetric, tensor_group 16, fp8 block scales,
memoryless min-max) — data-free weight-only, the same recipe the body was made with.

The gate/up ORDER inside the fused tensor is not assumed: with --ref (hibrid47, whose drafter experts are the base
model's, per-expert modelopt NVFP4) expert 0's two halves are compared against the reference's dequantized gate_proj and
up_proj by cosine — the assignment must be unambiguous or the script refuses.

Never edits a shared file in place: writes `model-mtp.safetensors.tmp` next to the old one, then os.replace() swaps the
directory entry (the source checkpoint's inode is untouched), then rewrites model.safetensors.index.json.

  python3 builds/qwen38-flash-next/quantize-drafter-experts.py --ckpt models/myllmbox/Qwen3.8-Flash-Next-hibrid47-uncensored \
      --ref models/myllmbox/Qwen3.8-Flash-Next-hibrid47 [--check]
Run in the vendor image (torch + compressed_tensors): docker run --rm -i -v $PWD:/w --entrypoint python3 <image> /w/builds/… .
"""
import argparse, json, os, struct, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8, unpack_fp4_from_uint8

ARGS = QuantizationArgs(num_bits=4, type="float", symmetric=True, strategy="tensor_group", group_size=16,
                        scale_dtype=torch.float8_e4m3fn, observer="memoryless_minmax")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def ct_quantize(w: torch.Tensor):
    """bf16/float [out, in] → (weight_packed u8 [out, in/2], weight_scale fp8 [out, in/16], weight_global_scale f32 [1])."""
    w = w.to(torch.float32)
    out, inn = w.shape
    assert inn % 16 == 0
    g = generate_gparam(w.min().reshape(1), w.max().reshape(1))                 # FP8_MAX*FP4_MAX / amax(tensor)
    wg = w.view(out, inn // 16, 16)
    scale, zp = calculate_qparams(wg.amin(-1), wg.amax(-1), ARGS, global_scale=g)   # fp8, [out, in/16]
    q = quantize(w, scale, zp, ARGS, global_scale=g)                            # e2m1 values in [-6, 6]
    return pack_fp4_to_uint8(q), scale.to(torch.float8_e4m3fn), g.reshape(1)   # values are already fp8-rounded; store as fp8 like the body


def ct_dequant(packed, scale, g):
    out = packed.shape[0]
    vals = unpack_fp4_from_uint8(packed, out, packed.shape[1] * 2, dtype=torch.float32)
    return (vals.view(out, -1, 16) * (scale.to(torch.float32) / g.to(torch.float32)).unsqueeze(-1)).view(out, -1)


def modelopt_dequant(w_u8, scale_fp8, scale2, nibble_low_first=True):
    """our hibrid47 (modelopt) layout: code × block scale × global; used only to verify gate/up order."""
    lo, hi = (w_u8 & 0xF).long(), (w_u8 >> 4).long()
    idx = torch.stack([lo, hi] if nibble_low_first else [hi, lo], -1).view(w_u8.shape[0], -1)
    vals = E2M1[idx & 7] * torch.where(idx & 8 > 0, -1.0, 1.0)
    return (vals.view(vals.shape[0], -1, 16) * scale_fp8.to(torch.float32).unsqueeze(-1)).view(vals.shape[0], -1) * scale2.to(torch.float32)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def fix_config_ignore(ck):
    """compressed-tensors `ignore` rules that blanket the drafter (orcarouter ships `re:.*mtp\\..*` — that is WHY its drafter
    was bf16: vLLM builds every ignored module unquantized, so NVFP4 drafter experts would have no parameter to land in:
    'Layer mtp.layers.48.mlp.experts has no parameter w2_weight_global_scale'). Drop them; the experts regex target then
    covers the drafter's experts and the drafter's other linears stay bf16 (they match no explicit target)."""
    p = os.path.join(ck, "config.json"); cfg = json.load(open(p))
    q = cfg.get("quantization_config") or {}
    ig = q.get("ignore") or []
    drop = [x for x in ig if x.startswith("re:") and "mtp" in x]
    if not drop:
        print("  config.json: no drafter-wide ignore rule — nothing to change"); return
    q["ignore"] = [x for x in ig if x not in drop]
    json.dump(cfg, open(p, "w"), indent=2)
    print(f"  config.json: removed quantization_config.ignore {drop} ({len(ig)} → {len(q['ignore'])} entries) so the drafter's experts are quantized like the body's")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--ref"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--src-mtp", help="read the fused bf16 drafter from THIS file instead of the checkpoint's (re-run after the checkpoint's copy was already replaced)")
    ap.add_argument("--config-only", action="store_true", help="only fix config.json's quantization_config.ignore for the drafter (no tensor work)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ck = os.path.abspath(a.ckpt)
    if a.config_only:
        fix_config_ignore(ck); return
    ipath = os.path.join(ck, "model.safetensors.index.json"); index = json.load(open(ipath)); wm = index["weight_map"]
    if a.src_mtp:
        src_path = os.path.abspath(a.src_mtp); h = header(src_path)
        fused = sorted(k for k in h if k.startswith("mtp.") and k.endswith((".experts.gate_up_proj", ".experts.down_proj")))
        mtp_file = os.path.basename(src_path); path = os.path.join(ck, mtp_file)
    else:
        fused = sorted(k for k in wm if k.startswith("mtp.") and k.endswith((".experts.gate_up_proj", ".experts.down_proj")))
        if not fused:
            sys.exit("✓ nothing to do: no fused bf16 drafter experts in the index (use --src-mtp to redo from the source file)")
        files = {wm[k] for k in fused}; assert len(files) == 1, files
        mtp_file = files.pop(); path = src_path = os.path.join(ck, mtp_file)
        h = header(path)
    if not fused:
        sys.exit("✗ no fused drafter experts in the source file")
    gu_name = [k for k in fused if k.endswith("gate_up_proj")][0]; dn_name = [k for k in fused if k.endswith("down_proj")][0]
    layer_prefix = gu_name[: -len(".gate_up_proj")]                     # mtp.layers.0.mlp.experts
    E, twoI, H = h[gu_name]["shape"]; E2, H2, I = h[dn_name]["shape"]
    assert (E, H, twoI) == (E2, H2, 2 * I), (h[gu_name]["shape"], h[dn_name]["shape"])
    print(f"drafter experts: {E} × gate_up [{twoI},{H}] + down [{H},{I}] {h[gu_name]['dtype']} = "
          f"{(E*(twoI*H+H*I))*2/2**30:.2f} GiB bf16 → NVFP4 ≈ {(E*(twoI*H+H*I))*(0.5+1/16)/2**30:.2f} GiB", flush=True)
    st = safe_open(src_path, "pt")

    # ---- gate/up order inside the fused tensor, checked against the reference's per-expert drafter ---------------
    gate_first = True
    if a.ref:
        ridx = json.load(open(os.path.join(a.ref, "model.safetensors.index.json")))["weight_map"]
        def rt(n):
            with safe_open(os.path.join(a.ref, ridx[n]), "pt") as f: return f.get_tensor(n)
        rg = f"{layer_prefix}.0.gate_proj."; ru = f"{layer_prefix}.0.up_proj."
        if rg + "weight_scale_2" in ridx:                              # modelopt layout (hibrid47)
            best = {}
            for nib in (True, False):
                G = modelopt_dequant(rt(rg + "weight"), rt(rg + "weight_scale"), rt(rg + "weight_scale_2"), nib)
                U = modelopt_dequant(rt(ru + "weight"), rt(ru + "weight_scale"), rt(ru + "weight_scale_2"), nib)
                top, bot = st.get_slice(gu_name)[0][:I].float(), st.get_slice(gu_name)[0][I:].float()
                best[nib] = (cos(top, G), cos(bot, U), cos(top, U), cos(bot, G))
            nib, (tg, bu, tu, bg) = max(best.items(), key=lambda kv: max(kv[1]))
            print(f"  order check vs {os.path.basename(a.ref)} (nibble {'low' if nib else 'high'} first): top≈gate {tg:.3f}, bottom≈up {bu:.3f} | top≈up {tu:.3f}, bottom≈gate {bg:.3f}")
            if max(tg, bu) > 0.9 and max(tu, bg) < 0.5: gate_first = True
            elif max(tu, bg) > 0.9 and max(tg, bu) < 0.5: gate_first = False
            else: sys.exit("✗ gate/up order ambiguous — refusing to guess")
            D = modelopt_dequant(rt(f"{layer_prefix}.0.down_proj.weight"), rt(f"{layer_prefix}.0.down_proj.weight_scale"), rt(f"{layer_prefix}.0.down_proj.weight_scale_2"), nib)
            print(f"  down_proj expert 0 vs reference: cos {cos(st.get_slice(dn_name)[0].float(), D):.3f} (abliteration touched down_proj: expect <1, but close)")
        else:
            print("  reference has no modelopt drafter experts — assuming gate first")
    print(f"  fused layout: [{'gate; up' if gate_first else 'up; gate'}] along dim 1", flush=True)

    # ---- one-expert round trip -----------------------------------------------------------------------------------------
    w = st.get_slice(gu_name)[0][:I].float()
    p, s, g = ct_quantize(w); rel = ((ct_dequant(p, s, g) - w).norm() / w.norm()).item()
    print(f"  round trip expert 0 gate: packed {tuple(p.shape)} {p.dtype}, scale {tuple(s.shape)} {s.dtype}, global {g.item():.4g}; rel err {rel:.4f}", flush=True)
    if a.check:
        print("check only — nothing written"); return

    # ---- quantize all experts, keep every other drafter tensor ----------------------------------------------------------
    t0 = time.time(); out = {}
    for k in h:
        if k == "__metadata__" or k in (gu_name, dn_name): continue
        out[k] = st.get_tensor(k)
    gu_slice, dn_slice = st.get_slice(gu_name), st.get_slice(dn_name)
    errs = []
    for e in range(E):
        gu = gu_slice[e]; a_, b_ = gu[:I], gu[I:]
        gate, up = (a_, b_) if gate_first else (b_, a_)
        for proj, w in (("gate_proj", gate), ("up_proj", up), ("down_proj", dn_slice[e])):
            p, s, g = ct_quantize(w)
            out[f"{layer_prefix}.{e}.{proj}.weight_packed"] = p; out[f"{layer_prefix}.{e}.{proj}.weight_scale"] = s
            out[f"{layer_prefix}.{e}.{proj}.weight_global_scale"] = g
            if e % 64 == 0 and proj == "down_proj": errs.append(((ct_dequant(p, s, g) - w.float()).norm() / w.float().norm()).item())
        if e % 64 == 63: print(f"  {e+1}/{E} experts ({time.time()-t0:.0f}s)", flush=True)
    tmp = path + ".tmp"
    save_file(out, tmp, metadata={"format": "pt"})
    st = None
    os.replace(tmp, path)                                     # swaps the directory entry; the source's inode is untouched
    for k in [k for k, v in wm.items() if v == mtp_file]: del wm[k]      # everything the old drafter file held
    for k in out: wm[k] = mtp_file
    total = sum(os.path.getsize(os.path.join(ck, f)) for f in set(wm.values()))
    index["metadata"] = {**(index.get("metadata") or {}), "total_size": total}; index["weight_map"] = dict(sorted(wm.items()))
    json.dump(index, open(ipath, "w"), indent=1)
    # verify: header of the new file vs index
    hn = header(path); names = [k for k in hn if k != "__metadata__"]
    assert all(wm.get(k) == mtp_file for k in names) and sum(v == mtp_file for v in wm.values()) == len(names)
    fix_config_ignore(ck)
    print(f"✓ {path}: {len(names)} tensors, {os.path.getsize(path)/2**30:.2f} GiB (was {sum(v['data_offsets'][1]-v['data_offsets'][0] for k,v in h.items() if k!='__metadata__')/2**30:.2f}); "
          f"down_proj rel err sampled {min(errs):.4f}–{max(errs):.4f}; checkpoint total {total/2**30:.1f} GiB ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
