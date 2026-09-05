# docker.io/myllmbox/qwen38-flash-next-cluster-vllm:v2

**Published:** 2026-09-05 (pushed from ai1; only the patch layers uploaded).
**Digest:** `sha256:21634ef576ac5a185327d548c410931a79e64f217074421e28b569d76108c1b2` (from the `docker push` output)
**Image id (local, = mbx-qwen38-flash-next-cluster):** `sha256:25219fc0c42580b15ee7772707f775f3d29d3a6f55d8091437fd448cba48d713`
**Reason for a new tag:** adds patch 02 (loader page-cache hygiene). v1 (cluster-v1.md) stays as published.

## What changed vs v1
`cluster/docker/patches/02-loader-drop-page-cache.py`: in `safetensors_weights_iterator` (default + eager
strategies, used by GPU workers AND the PLE offload worker) call `posix_fadvise(DONTNEED)` on each shard file
right after its tensors were yielded. Motive: two kit boots on 2026-09-05 livelocked during load with page cache at
61-68 GB and MemFree ~1 GB (UVM stall on one expert copy, see recipes/qwen38-flash-next-cluster/reports.md and the
uma-relaunch memory); on a Spark the driver wants FREE pages, and a shipped kit may not ask for root to drop caches.
Gate: `MBX_LOAD_DROP_CACHE=0` disables. Default ON.

## Validation
- Build: both patches applied + asserted (Dockerfile sanity step), vllm imports.
- Runtime with v2: **validated 2026-09-05 20:03** — kit boot on 2× Spark reached healthy (head load 412 s, worker
  243 s, KV 2,805,498 tokens). Page cache stayed at 16–22 GB THROUGH the load and 10 GB after (every earlier boot
  climbed to 60–70 GB); MemFree held ≥ 50 GB during the shard copies. No stall. (The kit reached healthy on 2026-09-05 19:32
  with v1 + the kit-side fixes: plain models/ mount with a real dir, head-first launch, memory gate, unprivileged
  `dd iflag=nocache` eviction of the checkpoint before launch; head load 351 s, worker 180 s, KV 2,805,498 tokens.)

## Pairs with
- Weights: `hf.co/myllmbox/Qwen3.8-Flash-Next-hibrid46` (unchanged)
- Kit: `github.com/bilikaz/qwen38-flash-next-cluster-recipe` (recipe.yaml → v2)
