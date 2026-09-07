#!/usr/bin/env python3
"""MBX: DUAL-PATH demand-paged PLE gather with a runtime switch — "nvme" until boot is done, "mmap" after.

On top of 04-ple-mmap. The gather becomes a vLLM custom op that is ALSO a splitting op (like the attention / PLE-conv
ops), so it runs eagerly between the piecewise CUDA-graph pieces and can change behaviour at runtime:
  path A "nvme": rows are read with O_DIRECT preads (thread pool) straight from the shard files → NO page cache, NO
                 GPU page faults, memory flat. Slow (~ms per step) — meant for boot (profile, autotune, warm-ups).
                 NOTE: this vLLM runner records the op into the piecewise CUDA graph, so under capture the mapped
                 gather is recorded (nothing executes at capture) and graph replays always use path B.
  path B "mmap": the ATS gather from 04 (index_select on the mapped tensors) — device speed, hot rows stay cached.
Switch: MBX_PLE_MMAP_MODE = "auto" (default: nvme → mmap the moment the worker's compile_or_warm_up_model returns,
i.e. warmup + capture done) | "nvme" | "mmap". Manual override any time: touch /cache/mbx-ple-mmap or /cache/mbx-ple-nvme
(the watcher thread from 04 polls every 2 s; the op reads an in-memory flag, never the file).
MBX_PLE_MMAP_PREWARM = "auto" → populate the whole table right after the automatic flip (the "load it all once we
survived boot" fallback); "0" / "1" / "<seconds>" as in 04.
Also edits vllm/config/compilation.py: adds "vllm::mbx_ple_gather" to the default splitting ops (changes the compile
hash → first boot recompiles). `--check` = assert anchors only.
"""
import sys

CHECK = "--check" in sys.argv
P = "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
PC = "/usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py"
s = open(P).read()
c = open(PC).read()
assert "_mbx_mmap_gather" in s, "04-ple-mmap must be applied first"
assert "mbx_ple_gather" not in s and "mbx_ple_gather" not in c, "already patched"

# --- compilation.py: our op is a splitting op by default ------------------------------------------------------------
a = '        "vllm::qwen3_8_flash_next_ple_short_conv",\n'
assert c.count(a) == 1, "splitting-op anchor not found"
c = c.replace(a, a + '        "vllm::mbx_ple_gather",                                # MBX: eager PLE gather (runtime nvme/mmap switch)\n', 1)

# --- ple_layer.py ---------------------------------------------------------------------------------------------------
edits = []
# embedding(): route the mmap path through the custom op (eager, splitting)
edits.append(("embedding via op",
    """        if getattr(layer, "_mbx_mmap", False):                                            # MBX: demand-paged table
            pk, sc8 = _mbx_mmap_gather(layer, ids)
            n = pk.shape[0]""",
    """        if getattr(layer, "_mbx_mmap", False):                                            # MBX: demand-paged table
            n = ids.shape[0]
            pk = torch.empty((n, layer._mbx_mm_cols_p), dtype=torch.uint8, device=ids.device)
            sc8 = torch.empty((n, layer._mbx_mm_cols_s), dtype=torch.float8_e4m3fn, device=ids.device)
            torch.ops.vllm.mbx_ple_gather(ids, pk, sc8, layer._mbx_name)                  # eager splitting op"""))
# attach: register the layer, keep file metadata for the nvme path, start auto-flip / prewarm plumbing
edits.append(("attach registry",
    """    packed, scales, keep = _mbx_mmap_table(model_dir, cfg, cuda=True, device_id=dev)
    layer._mbx_mm_packed = packed; layer._mbx_mm_scales = scales; layer._mbx_mm_keep = keep
    layer._mbx_mm_S = int(cfg["shard_rows"]); layer._mbx_mm_n = len(packed)""",
    """    packed, scales, keep, meta = _mbx_mmap_table(model_dir, cfg, cuda=True, device_id=dev, want_meta=True)
    layer._mbx_mm_packed = packed; layer._mbx_mm_scales = scales; layer._mbx_mm_keep = keep
    layer._mbx_mm_S = int(cfg["shard_rows"]); layer._mbx_mm_n = len(packed)
    layer._mbx_mm_meta = meta
    layer._mbx_mm_cols_p = int(packed[0].shape[1]); layer._mbx_mm_cols_s = int(scales[0].shape[1])
    layer._mbx_name = f"ple_mmap_{len(_MBX_PLE_LAYERS)}"
    _MBX_PLE_LAYERS[layer._mbx_name] = layer
    _mbx_dual_setup(model_dir)"""))
# mmap_table: optionally return per-tensor file metadata
edits.append(("table meta sig",
    "def _mbx_mmap_table(model_dir: str, cfg: dict, cuda: bool = True, device_id: int = 0):",
    "def _mbx_mmap_table(model_dir: str, cfg: dict, cuda: bool = True, device_id: int = 0, want_meta: bool = False):"))
edits.append(("table meta collect",
    """        ten = dl.uint8_2d(data0 + a, rows, cols, cuda=cuda, device_id=device_id)""",
    """        meta[(k, kind)] = (path, 8 + hl + a, cols, _os.path.getsize(path))       # MBX: for the O_DIRECT path
        ten = dl.uint8_2d(data0 + a, rows, cols, cuda=cuda, device_id=device_id)"""))
edits.append(("table meta init",
    "    files, packed, scales, keep = {}, [None] * nsh, [None] * nsh, []",
    "    files, packed, scales, keep, meta = {}, [None] * nsh, [None] * nsh, [], {}"))
edits.append(("table meta return",
    "    return packed, scales, keep\n\n\ndef _mbx_mmap_attach",
    "    return (packed, scales, keep, meta) if want_meta else (packed, scales, keep)\n\n\ndef _mbx_mmap_attach"))
# the watcher from 04: also honour the mode flags
edits.append(("watcher flags",
    """            while not done:
                _time.sleep(2.0)
                if _os.path.exists(trigger) or (delay is not None and _time.time() - t_start >= delay):""",
    """            while True:
                _time.sleep(2.0)
                for flag, mode_ in ((_MBX_FLAG_DIR + "/mbx-ple-mmap", "mmap"), (_MBX_FLAG_DIR + "/mbx-ple-nvme", "nvme")):
                    if _os.path.exists(flag):
                        _mbx_set_mode(mode_, "flag file")
                        try:
                            _os.remove(flag)
                        except OSError:
                            pass
                # the trigger file works EVERY time (a populate is repeatable); the delay fires once
                if _os.path.exists(trigger) or (not done and delay is not None and _time.time() - t_start >= delay):"""))

for label, a_, b_ in edits:
    assert s.count(a_) == 1, f"anchor '{label}' not found/unique ({s.count(a_)})"
    s = s.replace(a_, b_, 1)

s += '''

# ---- MBX: dual-path gather (nvme O_DIRECT until boot is done, then the ATS mmap gather) ---------------------------
_MBX_PLE_LAYERS: dict = {}
_MBX_PLE_STATE = {"mode": "nvme", "steps": 0, "nvme_ms": 0.0}
_MBX_FLAG_DIR = "/cache"
_MBX_NVME_POOL = None
_MBX_DUAL_DONE = False


def _mbx_set_mode(mode: str, why: str) -> None:
    if mode not in ("nvme", "mmap") or _MBX_PLE_STATE["mode"] == mode:
        return
    _MBX_PLE_STATE["mode"] = mode
    print(f"PLE MMAP: gather path → {mode.upper()} ({why}; {_MBX_PLE_STATE['steps']} steps so far, "
          f"{_MBX_PLE_STATE['nvme_ms']:.0f} ms spent in nvme reads)", flush=True)


def _mbx_dual_setup(model_dir: str) -> None:
    """Once per process: initial mode, the auto-flip hook on the worker's warm-up, PREWARM=auto chaining."""
    global _MBX_DUAL_DONE
    if _MBX_DUAL_DONE:
        return
    _MBX_DUAL_DONE = True
    import os as _os
    mode = _os.environ.get("MBX_PLE_MMAP_MODE", "auto").lower()
    prewarm = _os.environ.get("MBX_PLE_MMAP_PREWARM", "0")
    if mode in ("nvme", "mmap"):
        _MBX_PLE_STATE["mode"] = mode
        print(f"PLE MMAP: gather path fixed to {mode.upper()} (MBX_PLE_MMAP_MODE)", flush=True)
        return
    _MBX_PLE_STATE["mode"] = "nvme"
    try:
        import vllm.v1.worker.gpu_worker as _gw
        _orig = _gw.Worker.compile_or_warm_up_model

        def _wrapped(self, *a, **k):
            r = _orig(self, *a, **k)
            _mbx_set_mode("mmap", "auto: warm-up + graph capture finished")
            _mbx_trim_heap("after warm-up")                     # hand the loader's retained heap back before populating
            if prewarm == "auto":
                keep = []
                for lay in _MBX_PLE_LAYERS.values():
                    keep.extend(lay._mbx_mm_keep)
                print("PLE MMAP: PREWARM=auto → populating the whole table now", flush=True)
                _mbx_mmap_populate(keep)
            return r
        _gw.Worker.compile_or_warm_up_model = _wrapped
        print("PLE MMAP: gather path NVME (O_DIRECT) during boot; flips to MMAP when the worker finishes warm-up/capture",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"PLE MMAP: could not hook the worker warm-up ({e}); staying on MMAP", flush=True)
        _MBX_PLE_STATE["mode"] = "mmap"


def _mbx_trim_heap(why: str) -> None:
    """glibc keeps the loader's freed blocks in the worker heap (~3 GB seen on the solo box). Return them to the
    kernel: that is memory the populated table can live in. Also releases the CUDA caching allocator's idle blocks."""
    def _rss():
        try:
            return int([l for l in open("/proc/self/status") if l.startswith("VmRSS:")][0].split()[1]) / 2**20
        except Exception:  # noqa: BLE001
            return float("nan")
    before = _rss()
    try:
        import ctypes as _c
        _c.CDLL("libc.so.6").malloc_trim(0)
    except Exception as e:  # noqa: BLE001
        print(f"PLE MMAP: malloc_trim unavailable ({e})", flush=True)
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    print(f"PLE MMAP: heap trim {why}: worker RSS {before:.2f} → {_rss():.2f} GiB", flush=True)


def _mbx_nvme_fds(layer):
    """Per (shard, kind): O_DIRECT fd (falls back to buffered if the filesystem refuses O_DIRECT)."""
    import os as _os
    fds = getattr(layer, "_mbx_nvme_fds", None)
    if fds is None:
        fds = {}
        for key, (path, off0, cols, fsize) in layer._mbx_mm_meta.items():
            try:
                fd = _os.open(path, _os.O_RDONLY | _os.O_DIRECT); direct = True
            except OSError:
                fd = _os.open(path, _os.O_RDONLY); direct = False
            fds[key] = (fd, off0, cols, fsize, direct)
        layer._mbx_nvme_fds = fds
        if not all(v[4] for v in fds.values()):
            print("PLE MMAP: O_DIRECT unavailable on this filesystem — nvme path uses buffered preads", flush=True)
    return fds


def _mbx_nvme_gather(layer, ids: torch.Tensor):
    """Path A: read the rows of `ids` from the shard files with aligned O_DIRECT preads on a thread pool.
    Returns CPU tensors (packed u8 [n, D/2], scales fp8 [n, D/16])."""
    global _MBX_NVME_POOL
    import mmap as _mmap, os as _os, time as _time
    import numpy as _np
    from concurrent.futures import ThreadPoolExecutor
    if _MBX_NVME_POOL is None:
        _MBX_NVME_POOL = ThreadPoolExecutor(max_workers=int(_os.environ.get("MBX_PLE_NVME_THREADS", "32")))
    t0 = _time.time()
    fds = _mbx_nvme_fds(layer)
    S = layer._mbx_mm_S
    ids_np = ids.detach().cpu().numpy().astype(_np.int64)
    n = ids_np.shape[0]
    uniq, inv = _np.unique(ids_np, return_inverse=True)
    PG = 4096
    out = {}
    for kind in ("packed", "scales"):
        cols = layer._mbx_mm_cols_p if kind == "packed" else layer._mbx_mm_cols_s
        buf = _mmap.mmap(-1, max(1, len(uniq)) * 2 * PG)          # page-aligned scratch, 2 pages per row (may straddle)
        rows = _np.empty((len(uniq), cols), dtype=_np.uint8)

        def one(i, row=None):
            k, r = divmod(int(uniq[i]), S)
            fd, off0, c, fsize, direct = fds[(k, kind)]
            off = off0 + r * c
            a0 = off & ~(PG - 1)
            ln = 2 * PG if (off - a0 + c) > PG else PG
            if a0 + ln > fsize:
                ln = ((fsize - a0 + PG - 1) // PG) * PG if direct else fsize - a0
            mv = memoryview(buf)[i * 2 * PG: i * 2 * PG + ln]
            got = _os.preadv(fd, [mv], a0)
            rows[i] = _np.frombuffer(mv, dtype=_np.uint8, count=c, offset=off - a0)
            return got
        list(_MBX_NVME_POOL.map(one, range(len(uniq))))
        out[kind] = torch.from_numpy(rows[inv]) if len(uniq) else torch.empty((0, cols), dtype=torch.uint8)
        buf.close()
    _MBX_PLE_STATE["nvme_ms"] += (_time.time() - t0) * 1e3
    return out["packed"], out["scales"].view(torch.float8_e4m3fn)


def mbx_ple_gather(ids: torch.Tensor, packed_out: torch.Tensor, scales_out: torch.Tensor, layer_name: str) -> None:
    """The eager splitting op: fills packed_out/scales_out for `ids` via the current path."""
    layer = _MBX_PLE_LAYERS[layer_name]
    _MBX_PLE_STATE["steps"] += 1
    # During piecewise CUDA-graph CAPTURE the piece that computes the ids is recorded, not run: its output buffer is
    # uninitialized memory, so this eager op sees junk ids (seen: 2.3e11 as a shard index). Clamp to the table range —
    # junk becomes a harmless read of a valid row; real ids are already in range. Also guards the mmap path from an
    # out-of-bounds read through the host pointer.
    ids = torch.clamp(ids, 0, layer._mbx_mm_S * layer._mbx_mm_n - 1)
    # This runner records the op INTO the piecewise graph (the stream is capturing when we run — a CPU sync here
    # raised "Cannot copy between CPU and CUDA tensors during CUDA graph capture"). Under capture only GPU kernels are
    # legal, so the mapped gather is what gets recorded; captured kernels do not execute at capture time, so no table
    # page is touched during boot. Graph REPLAYS (decode) therefore always take the mmap path; the nvme/mmap mode
    # applies to eager execution: the profile run, warm-ups, prefill and any batch size without a graph.
    if _MBX_PLE_STATE["mode"] == "mmap" or (ids.is_cuda and torch.cuda.is_current_stream_capturing()):
        pk, sc = _mbx_mmap_gather(layer, ids)
        packed_out.copy_(pk)
        scales_out.copy_(sc)
    else:
        pk, sc = _mbx_nvme_gather(layer, ids)
        packed_out.copy_(pk)
        scales_out.copy_(sc)


def mbx_ple_gather_fake(ids: torch.Tensor, packed_out: torch.Tensor, scales_out: torch.Tensor, layer_name: str) -> None:
    return


direct_register_custom_op(
    op_name="mbx_ple_gather",
    op_func=mbx_ple_gather,
    mutates_args=["packed_out", "scales_out"],
    fake_impl=mbx_ple_gather_fake,
)
'''

if not CHECK:
    open(P, "w").write(s)
    open(PC, "w").write(c)
print("dual-path PLE gather patch:", "anchors OK" if CHECK else "applied (MBX_PLE_MMAP_MODE=auto|nvme|mmap; vllm::mbx_ple_gather is a splitting op)")
