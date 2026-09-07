#!/usr/bin/env python3
"""MBX: multi-node PLE offload (gated by env MBX_PLE_MULTINODE=1; absent -> stock behaviour, byte-identical).

Upstream fences the PLE CPU-offload worker to nnodes=1, but the worker is already node-local by design (its own
gloo world_size=1, full table, IPC over /tmp). Three gaps keep it single-node; all lifted under the gate:
  1. gpu_worker._validate_ple_offload_config refuses nnodes != 1          -> allowed
  2. the spawn gate is GLOBAL rank 0 (docstring says node-local)          -> per-node local_rank 0
  3. the worker expects dp*tp GLOBAL registrations + TP0 buffers          -> collapse to the LOCAL view: each
     node's worker serves its local rank; the sole registration is remapped to tp_rank 0 so the stock
     request-source and target-count checks pass unchanged.
  4. (first multi-node boot deadlocked in warmup) the CONNECTOR gates runtime gather requests on GLOBAL tp_rank 0,
     so box2 never fed its own worker, its PLE semaphore never fired, box1 spun in an allreduce forever.
     Under the gate every rank is its node's request source.
Proven 2026-08-29 (experiments/qwen38-flash-next-cluster-vllm, TP=2, first healthy boot). Anchor-asserted:
refuses to apply twice; fails the build loudly if upstream moved an anchor. `--check` = assert anchors only.
"""
import sys

CHECK = "--check" in sys.argv
SP = "/usr/local/lib/python3.12/dist-packages/vllm/"


def patch(path, edits):
    p = SP + path
    s = open(p).read()
    assert "MBX_PLE_MULTINODE" not in s, f"{path}: already patched"
    for name, a, b in edits:
        assert s.count(a) == 1, f"{path}: anchor '{name}' not found/unique — upstream changed"
        s = s.replace(a, b, 1)
    if not CHECK:
        open(p, "w").write(s)
    print(f"{'checked' if CHECK else 'patched'} {path} ({len(edits)} sites)")


# --- gpu_worker.py: validation, spawn gate, local worker count ---
patch("v1/worker/gpu_worker.py", [
    ("nnodes-check",
     """        if parallel_config.nnodes != 1:
            unsupported.append(f"nnodes={parallel_config.nnodes}")""",
     """        import os as _os
        if parallel_config.nnodes != 1 and _os.environ.get(
            "MBX_PLE_MULTINODE", "0"
        ) != "1":
            unsupported.append(f"nnodes={parallel_config.nnodes}")"""),
    ("spawn-gate",
     """        if (
            not self._ple_offload_enabled
            or self.rank != 0
            or self.parallel_config.data_parallel_rank != 0
        ):
            return""",
     """        import os as _os
        _mbx_mn = _os.environ.get("MBX_PLE_MULTINODE", "0") == "1"
        _gate_rank = self.local_rank if _mbx_mn else self.rank
        if (
            not self._ple_offload_enabled
            or _gate_rank != 0
            or self.parallel_config.data_parallel_rank != 0
        ):
            return"""),
    ("num_workers",
     """        dp_size = self.parallel_config.data_parallel_size
        tp_size = self.parallel_config.tensor_parallel_size
        num_workers = dp_size * tp_size""",
     """        dp_size = self.parallel_config.data_parallel_size
        tp_size = self.parallel_config.tensor_parallel_size
        num_workers = dp_size * tp_size
        if _mbx_mn:
            # MBX: per-node worker — expect only this node's local rank(s)
            num_workers = max(1, num_workers // max(1, self.parallel_config.nnodes))"""),
])

# --- ple_offload/worker.py: collapse topology checks to the local view ---
patch("v1/ple_offload/worker.py", [
    ("worker-topology",
     """        dp_size = self.vllm_config.parallel_config.data_parallel_size
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if num_workers != dp_size * tp_size:""",
     """        dp_size = self.vllm_config.parallel_config.data_parallel_size
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        import os as _os
        if _os.environ.get("MBX_PLE_MULTINODE", "0") == "1":
            # MBX: per-node worker — the local registrations ARE the whole world.
            # Remap local tp ranks to 0..n-1 so the stock request-source (tp_rank 0)
            # and per-layer target-count checks hold without further changes.
            dp_size = 1
            tp_size = num_workers
            for _i, _r in enumerate(sorted(registrations, key=lambda r: r.tp_rank)):
                try:
                    _r.tp_rank = _i
                    _r.dp_rank = 0
                except Exception:
                    object.__setattr__(_r, "tp_rank", _i)
                    object.__setattr__(_r, "dp_rank", 0)
        if num_workers != dp_size * tp_size:"""),
])

# --- ple_offload/connector.py: every rank is its node's request source ---
patch("v1/ple_offload/connector.py", [
    ("connector-pin-gate",
     """            if self.tp_rank == 0:
                # ForkingPickler may replace CPU storage while converting its
                # sharing strategy, so register only the final addresses.""",
     """            import os as _os
            _mbx_mn = _os.environ.get("MBX_PLE_MULTINODE", "0") == "1"
            if self.tp_rank == 0 or _mbx_mn:
                # ForkingPickler may replace CPU storage while converting its
                # sharing strategy, so register only the final addresses."""),
    ("connector-request-gate",
     """        # Inputs are replicated across TP ranks. One request per DP rank drives
        # the CPU result fan-out to every registered TP output buffer.
        if self.tp_rank != 0:
            return""",
     """        # Inputs are replicated across TP ranks. One request per DP rank drives
        # the CPU result fan-out to every registered TP output buffer.
        # MBX multi-node: every rank drives its own node-local worker.
        import os as _os
        if self.tp_rank != 0 and _os.environ.get("MBX_PLE_MULTINODE", "0") != "1":
            return"""),
])
print("multi-node PLE offload patch:", "anchors OK" if CHECK else "applied (gated by MBX_PLE_MULTINODE=1)")
