#!/usr/bin/env python3
"""MBX: DEMAND-PAGED NVFP4 PLE table — the shard files stay on NVMe, mapped read-only; the GPU gathers straight
out of the mapping (GB10 ATS: `pageableMemoryAccess=1`). Nothing is allocated for the table at boot.

Activates on top of the GPU-resident patch (02/03-ple-gpu-nvfp4) when MBX_PLE_MMAP=1 (single rank / TP=1 or
MBX_PLE_REPLICATE=1 — every rank maps the whole table). Then:
  1. `create_weights` registers the packed/scales parameters with ZERO rows (no device allocation).
  2. `_mbx_nvfp4_load` skips the copy for shard tensors (the loader hands us the safetensors mmap; untouched = unread),
     only counting rows.
  3. `process_weights_after_loading` mmaps the 8 `ple-nvfp4-*.safetensors` files (via model.safetensors.index.json +
     each file's header) and wraps the tensor bytes as CUDA tensors through a hand-built DLPack capsule
     (device kDLCUDA over the host pointer — torch does not check residency; ATS makes it legal).
  4. `embedding()` gathers per shard from the mappings (index_select on the DLPack tensors), then the same
     nibble-unpack/LUT/scale math as the resident path.
First touch of a page = one NVMe read + a GPU-side page fault (~80 µs + ~200 µs, builds/qwen38-flash-next/ple-mmap-probe);
after that the row is a page-cache read at device speed. Pages are clean file pages: the kernel may reclaim them under
pressure (then they cost a re-read, not an OOM). MBX_PLE_MMAP_PREWARM=1 = MADV_POPULATE_READ the whole table at boot
(the fully-resident baseline, same bytes as the resident path but evictable).
`--check` = assert anchors only.
"""
import sys

CHECK = "--check" in sys.argv
P = "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
s = open(P).read()
assert "_MbxNvfp4EmbeddingMethod" in s, "GPU-resident NVFP4 patch must be applied first"
assert "_mbx_ple_mmap" not in s, "already patched"

edits = []

# A. create_weights: zero-row parameters in mmap mode
edits.append(("create_weights alloc",
    """        n = int(sum(output_partition_sizes)); D = int(input_size_per_partition)
        assert D % self.BLOCK == 0 and D % 2 == 0, f"PLE head_dim {D} not NVFP4-blockable"
        layer.register_parameter("nvfp4_packed", nn.Parameter(torch.empty((n, D // 2), dtype=torch.uint8), requires_grad=False))
        layer.register_parameter("nvfp4_scales", nn.Parameter(torch.empty((n, D // self.BLOCK), dtype=torch.float8_e4m3fn), requires_grad=False))""",
    """        n = int(sum(output_partition_sizes)); D = int(input_size_per_partition)
        assert D % self.BLOCK == 0 and D % 2 == 0, f"PLE head_dim {D} not NVFP4-blockable"
        layer._mbx_mmap = _mbx_ple_mmap()                    # MBX: demand-paged table → allocate nothing
        layer._mbx_mmap_rows = n
        n_alloc = 0 if layer._mbx_mmap else n
        layer.register_parameter("nvfp4_packed", nn.Parameter(torch.empty((n_alloc, D // 2), dtype=torch.uint8), requires_grad=False))
        layer.register_parameter("nvfp4_scales", nn.Parameter(torch.empty((n_alloc, D // self.BLOCK), dtype=torch.float8_e4m3fn), requires_grad=False))"""))

# B. process_weights_after_loading: build the mappings instead of checking the copy
edits.append(("pwal",
    """    def process_weights_after_loading(self, layer) -> None:
        exp = int(layer.nvfp4_packed.shape[0])
        got = int(getattr(layer, "_mbx_nvfp4_rows_loaded", 0))""",
    """    def process_weights_after_loading(self, layer) -> None:
        if getattr(layer, "_mbx_mmap", False):                # MBX: demand-paged table
            _mbx_mmap_attach(layer)
            return
        exp = int(layer.nvfp4_packed.shape[0])
        got = int(getattr(layer, "_mbx_nvfp4_rows_loaded", 0))"""))

# C. embedding(): per-shard gather from the mappings
edits.append(("embedding gather",
    """        shape = input_.shape
        ids = input_.reshape(-1)
        pk = layer.nvfp4_packed.index_select(0, ids)                                     # [n, D/2] u8
        n = pk.shape[0]""",
    """        shape = input_.shape
        ids = input_.reshape(-1)
        if getattr(layer, "_mbx_mmap", False):                                            # MBX: demand-paged table
            pk, sc8 = _mbx_mmap_gather(layer, ids)
            n = pk.shape[0]
            codes = torch.stack(((pk & 0x0F), (pk >> 4)), dim=-1).view(n, -1).long()
            x = layer.nvfp4_lut[codes]
            sc = sc8.to(torch.float32) * layer.nvfp4_global
            x = x.view(n, -1, self.BLOCK) * sc.unsqueeze(-1)
            return x.view(*shape, -1).to(layer.params_dtype)
        pk = layer.nvfp4_packed.index_select(0, ids)                                     # [n, D/2] u8
        n = pk.shape[0]"""))

# D. loader: count rows, copy nothing
edits.append(("load skip",
    """    param = emb.nvfp4_packed if kind == "packed" else emb.nvfp4_scales
    if hi > lo:""",
    """    if getattr(emb, "_mbx_mmap", False):                     # MBX: demand-paged — the bytes stay in the file
        if kind == "packed" and hi > lo:
            emb._mbx_nvfp4_rows_loaded = getattr(emb, "_mbx_nvfp4_rows_loaded", 0) + (hi - lo)
        return {f"ngram_embedding.nvfp4_{kind}"}
    param = emb.nvfp4_packed if kind == "packed" else emb.nvfp4_scales
    if hi > lo:"""))

for label, a, b in edits:
    assert s.count(a) == 1, f"anchor '{label}' not found/unique"
    s = s.replace(a, b, 1)

s += '''

# ---- MBX: demand-paged NVFP4 PLE table (mmap + DLPack-over-host-pointer, GB10 ATS) ----------------------------
def _mbx_ple_mmap() -> bool:
    import os as _os
    return _os.environ.get("MBX_PLE_MMAP", "0") == "1"


import ctypes as _ct


class _MbxDLDevice(_ct.Structure):
    _fields_ = [("device_type", _ct.c_int32), ("device_id", _ct.c_int32)]


class _MbxDLDataType(_ct.Structure):
    _fields_ = [("code", _ct.c_uint8), ("bits", _ct.c_uint8), ("lanes", _ct.c_uint16)]


class _MbxDLTensor(_ct.Structure):
    _fields_ = [("data", _ct.c_void_p), ("device", _MbxDLDevice), ("ndim", _ct.c_int32), ("dtype", _MbxDLDataType),
                ("shape", _ct.POINTER(_ct.c_int64)), ("strides", _ct.POINTER(_ct.c_int64)), ("byte_offset", _ct.c_uint64)]


class _MbxDLManagedTensor(_ct.Structure):
    pass


_MbxDLDeleter = _ct.CFUNCTYPE(None, _ct.POINTER(_MbxDLManagedTensor))
_MbxDLManagedTensor._fields_ = [("dl_tensor", _MbxDLTensor), ("manager_ctx", _ct.c_void_p), ("deleter", _MbxDLDeleter)]


class _MbxDLPack:
    """Wrap a raw host pointer as a (CUDA or CPU) tensor through a hand-built DLPack capsule. The structs must
    outlive the tensors — keep this object alive alongside them."""
    kDLCPU, kDLCUDA, kDLUInt = 1, 2, 1

    def __init__(self):
        self.keep = []

    def uint8_2d(self, ptr: int, rows: int, cols: int, cuda: bool = True, device_id: int = 0):
        shape = (_ct.c_int64 * 2)(rows, cols)
        m = _MbxDLManagedTensor()
        m.dl_tensor.data = _ct.c_void_p(ptr)
        m.dl_tensor.device = _MbxDLDevice(self.kDLCUDA if cuda else self.kDLCPU, device_id)
        m.dl_tensor.ndim = 2
        m.dl_tensor.dtype = _MbxDLDataType(self.kDLUInt, 8, 1)
        m.dl_tensor.shape = shape
        m.dl_tensor.strides = None
        m.dl_tensor.byte_offset = 0
        m.manager_ctx = None
        m.deleter = _MbxDLDeleter()        # NULL deleter (DLPack-legal; torch checks for it): nothing to free — the mmap
                                           # and this struct are owned here, and a Python callback would be torn down
                                           # before the tensors at interpreter exit (segfault in Py_Finalize)
        self.keep.extend((shape, m))
        _ct.pythonapi.PyCapsule_New.restype = _ct.py_object
        _ct.pythonapi.PyCapsule_New.argtypes = [_ct.c_void_p, _ct.c_char_p, _ct.c_void_p]
        cap = _ct.pythonapi.PyCapsule_New(_ct.addressof(m), b"dltensor", None)
        return torch.from_dlpack(cap)


def _mbx_mmap_table(model_dir: str, cfg: dict, cuda: bool = True, device_id: int = 0):
    """Map every `ngram_embedding.nvfp4_shard_k.{packed,scales}` tensor of the checkpoint; return
    (packed[k] u8 [S, D/2], scales[k] fp8 [S, D/16], keepalive). Reads only the safetensors headers."""
    import json as _json, mmap as _mmap, os as _os, re as _re, struct as _struct
    idx = _json.load(open(_os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    pat = _re.compile(r"ngram_embedding\\.nvfp4_shard_(\\d+)\\.(packed|scales)$")
    want = {}
    for name, f in idx.items():
        m = pat.search(name)
        if m:
            want[(int(m.group(1)), m.group(2))] = (name, f)
    nsh = int(cfg["shards"]); S = int(cfg["shard_rows"])
    assert len(want) == 2 * nsh, f"index lists {len(want)} table tensors, expected {2 * nsh}"
    files, packed, scales, keep = {}, [None] * nsh, [None] * nsh, []
    dl = _mbx_ple_mmap_dlpack = _MbxDLPack(); keep.append(dl)
    for (k, kind), (name, f) in sorted(want.items()):
        path = _os.path.join(model_dir, f)
        if path not in files:
            fd = _os.open(path, _os.O_RDONLY)
            mm = _mmap.mmap(fd, 0, prot=_mmap.PROT_READ, flags=_mmap.MAP_SHARED)
            _os.close(fd)
            hl = _struct.unpack("<Q", mm[:8])[0]
            hdr = _json.loads(mm[8:8 + hl])
            import numpy as _np                                    # ctypes.from_buffer needs a WRITABLE buffer;
            base = int(_np.frombuffer(mm, dtype=_np.uint8).ctypes.data)  # numpy takes the read-only mmap as-is
            files[path] = (mm, hdr, base + 8 + hl); keep.append(mm)
        mm, hdr, data0 = files[path]
        t = hdr[name]; a, b = t["data_offsets"]; rows, cols = t["shape"]
        assert rows == S, f"{name}: {rows} rows, config says {S}"
        assert b - a == rows * cols, f"{name}: byte size mismatch"
        ten = dl.uint8_2d(data0 + a, rows, cols, cuda=cuda, device_id=device_id)
        if kind == "packed":
            packed[k] = ten
        else:
            scales[k] = ten.view(torch.float8_e4m3fn)
    return packed, scales, keep


def _mbx_mmap_attach(layer) -> None:
    import os as _os, time as _time
    cfg = _mbx_ple_std_cfg()
    emb_rows = int(layer._mbx_mmap_rows)
    total = int(cfg["rows"])
    if emb_rows < total:
        raise RuntimeError(f"MBX PLE MMAP needs the whole table on this rank ({emb_rows} < {total} rows): "
                           "run TP=1 or set MBX_PLE_REPLICATE=1")
    model_dir = _os.environ.get("MBX_PLE_MMAP_DIR") or get_current_vllm_config().model_config.model
    dev = torch.cuda.current_device()
    t0 = _time.time()
    packed, scales, keep = _mbx_mmap_table(model_dir, cfg, cuda=True, device_id=dev)
    layer._mbx_mm_packed = packed; layer._mbx_mm_scales = scales; layer._mbx_mm_keep = keep
    layer._mbx_mm_S = int(cfg["shard_rows"]); layer._mbx_mm_n = len(packed)
    gib = sum(t.numel() for t in packed) / 2**30 + sum(t.numel() for t in scales) / 2**30
    # MBX_PLE_MMAP_PREWARM: "0" = pure demand paging; "1" = populate the whole table NOW (at load, before autotune);
    # "<N>" = populate N seconds after load from a background thread (i.e. after autotune/graph capture have had the
    # room — the "load it all once we survived boot" fallback). Any mode: touching the trigger file
    # $MBX_PLE_MMAP_TRIGGER (default /cache/mbx-ple-prewarm) populates on demand.
    mode = _os.environ.get("MBX_PLE_MMAP_PREWARM", "0")
    trigger = _os.environ.get("MBX_PLE_MMAP_TRIGGER", "/cache/mbx-ple-prewarm")
    if mode == "1":
        _mbx_mmap_populate(keep)
    else:
        import threading as _th
        delay = int(mode) if mode.isdigit() and int(mode) > 1 else None
        def _watch():
            t_start = _time.time(); done = False
            while not done:
                _time.sleep(2.0)
                if _os.path.exists(trigger) or (delay is not None and _time.time() - t_start >= delay):
                    why = "trigger file" if _os.path.exists(trigger) else f"{delay}s after load"
                    print(f"PLE MMAP: populating the whole table ({why})", flush=True)
                    _mbx_mmap_populate(keep); done = True
                    try:
                        _os.path.exists(trigger) and _os.remove(trigger)
                    except OSError:
                        pass
        _th.Thread(target=_watch, name="mbx-ple-prewarm", daemon=True).start()
    print(f"PLE: NVFP4 table DEMAND-PAGED from {model_dir} ({layer._mbx_mm_n} shards, {gib:.2f} GiB mapped, "
          f"0 allocated, prewarm={mode}, trigger={trigger}, {_time.time() - t0:.1f}s)", flush=True)


def _mbx_mmap_populate(keep) -> None:
    """Pull every page of the mapped table into the page cache AND this process's page tables (MADV_POPULATE_READ,
    Linux ≥ 5.14; fallback = touch one byte per page). Pages stay clean file pages: evictable under pressure."""
    import mmap as _mmap, time as _time
    t0 = _time.time(); tot = 0
    POPULATE_READ = getattr(_mmap, "MADV_POPULATE_READ", 22)
    for mm in keep:
        if not isinstance(mm, _mmap.mmap):
            continue
        tot += len(mm)
        try:
            mm.madvise(POPULATE_READ)
        except Exception as e:  # noqa: BLE001
            print(f"PLE MMAP: madvise(POPULATE_READ) failed ({e}); touching pages instead", flush=True)
            for off in range(0, len(mm), _mmap.PAGESIZE):
                mm[off]
    print(f"PLE MMAP: populated {tot / 2**30:.2f} GiB in {_time.time() - t0:.1f}s", flush=True)


def _mbx_mmap_gather(layer, ids: torch.Tensor):
    """Gather rows `ids` (global row index) from the per-shard mappings → (packed u8 [n, D/2], scales fp8 [n, D/16])."""
    S = layer._mbx_mm_S
    k = torch.div(ids, S, rounding_mode="floor")
    r = ids - k * S
    pk_all, sc_all = [], []
    for j in range(layer._mbx_mm_n):
        rj = torch.where(k == j, r, torch.zeros_like(r))          # rows of other shards → row 0 (cheap, always mapped)
        pk_all.append(layer._mbx_mm_packed[j].index_select(0, rj))
        sc_all.append(layer._mbx_mm_scales[j].index_select(0, rj))
    pk = torch.stack(pk_all, 0)                                    # [nsh, n, D/2]
    sc = torch.stack(sc_all, 0)                                    # [nsh, n, D/16]
    n = ids.shape[0]
    ar = torch.arange(n, device=ids.device)
    return pk[k, ar], sc[k, ar]
'''

if not CHECK:
    open(P, "w").write(s)
print("demand-paged NVFP4 PLE patch:", "anchors OK" if CHECK else "applied (activates with MBX_PLE_MMAP=1)")
