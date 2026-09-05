#!/usr/bin/env python3
"""MBX: drop each checkpoint shard from the page cache as soon as the loader has consumed it (default ON).

Why: on a DGX Spark the GPU driver and the page cache share one unified-memory pool, and the driver wants pages
that are FREE, not merely reclaimable. vLLM's safetensors loader mmap-reads the whole checkpoint (91 GB here) while
it allocates ~40 GB of parameters into GPU memory; the page cache balloons to 60-70 GB, MemFree falls to ~1 GB, and
the driver can stall on one host→GPU copy that never completes (100 % CPU, UVM thread busy, no progress — seen
twice on 2026-09-05, both times inside the fused-MoE expert copy). Root would `echo 3 > drop_caches`; a shipped
kit never asks for root. So the loader itself tells the kernel POSIX_FADV_DONTNEED for every shard file right
after its tensors were yielded and copied — the cache stays ~one shard deep for the whole load.

Applies to `safetensors_weights_iterator` (the default and the "eager" strategy), which both the GPU workers and the
PLE offload worker use. Gate: env MBX_LOAD_DROP_CACHE=0 disables. Anchor-asserted; `--check` = assert only.
"""
import sys

CHECK = "--check" in sys.argv
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py"
s = open(P).read()
assert "_mbx_fadvise_dontneed" not in s, "already patched"

edits = [
 ("helper next to _mbx_stale",
  """    def _mbx_stale(name: str, st_file: str) -> bool:""",
  """    def _mbx_fadvise_dontneed(path: str) -> None:
        # MBX: page-cache hygiene on unified memory — drop this shard's pages now that its tensors are consumed.
        if os.environ.get("MBX_LOAD_DROP_CACHE", "1") != "1":
            return
        try:
            _fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(_fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(_fd)
        except OSError:
            pass

    def _mbx_stale(name: str, st_file: str) -> bool:"""),
 ("eager branch",
  """        if safetensors_load_strategy == "eager":
            with open(st_file, "rb") as f:
                state_dict = load(f.read())
            for name, param in state_dict.items():
                if not should_skip_weight(name, local_expert_ids) and not _mbx_stale(
                    name, st_file
                ):
                    yield name, param""",
  """        if safetensors_load_strategy == "eager":
            with open(st_file, "rb") as f:
                state_dict = load(f.read())
            for name, param in state_dict.items():
                if not should_skip_weight(name, local_expert_ids) and not _mbx_stale(
                    name, st_file
                ):
                    yield name, param
            _mbx_fadvise_dontneed(st_file)"""),
 ("default branch",
  """        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids) or _mbx_stale(
                        name, st_file
                    ):
                        continue
                    param = f.get_tensor(name)
                    yield name, param


def multi_thread_safetensors_weights_iterator(""",
  """        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids) or _mbx_stale(
                        name, st_file
                    ):
                        continue
                    param = f.get_tensor(name)
                    yield name, param
            _mbx_fadvise_dontneed(st_file)


def multi_thread_safetensors_weights_iterator("""),
]
for label, a, b in edits:
    assert s.count(a) == 1, f"anchor '{label}' not found/unique — upstream changed"
    s = s.replace(a, b, 1)
assert "\nimport os\n" in s or "import os" in s, "weight_utils.py must import os"
if not CHECK:
    open(P, "w").write(s)
print("loader page-cache patch:", "anchors OK" if CHECK else "applied (MBX_LOAD_DROP_CACHE=1 default; set 0 to disable)")
