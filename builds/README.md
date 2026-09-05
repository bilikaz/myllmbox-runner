# builds/ — the release ledger + reproducibility bundles

One FOLDER per published model (e.g. `qwen38-flash-next/`) holding everything that PRODUCES its
public artifacts: the image Dockerfile + `docker/` payloads, the quant converter it bakes, and
one ledger entry per pushed artifact (`vllm-v1.md`, …). recipes/ CONSUME published artifacts
(pinned image tag + downloaded weights); builds/ is where they come from. This is the paper trail
the public claims stand on: what bytes were shipped, what produced them, and what validation they
passed before the push. Committed to git (unlike solo/ kits, which are their own repos).

Every entry records:
- **artifact**: the public address (docker tag / HF repo) and its immutable digest/revision
- **source**: the Dockerfile / converter / export invocation and the git commit that held them
- **base**: pinned upstream digests it was built FROM
- **validation**: the exact evidence gathered before publishing (boot, bands, output checks)
- **config**: the shipping serve parameters at release time

Rules: entries are append-only history — never rewrite a published entry; a re-push under the
same tag gets a NEW entry (and a reason). Fill digests from the actual push output, never from
memory. House rule applies: public references are versioned TAGS with the digest documented
beside them, never bare @sha256 pulls.

## Solo kits (the distribution repos)

Each published model gets a standalone serving kit at `github.com/bilikaz/<model>-recipe`,
developed under `solo/<model>-recipe/` (fully gitignored here — each kit is its OWN git repo).
The kit contract: **one `recipe.yaml`** (the only config: image, weights repo, port, every vLLM
flag) + `run.sh` (download if needed → serve → wait healthy) + `stop.sh` + `view.sh` (live
stats) + `README.md` (quickstart, measured numbers WITH conditions, tuning, image digest,
license). Kits are consumers only: no Dockerfile, no build machinery — the README links back to
this folder for reproduction. Flags in a kit's recipe.yaml must equal the validated recipe lane's
set at export time; no myllmbox internals may leak in; the kit must work on a fresh box with
nothing but docker (+ optionally the hf CLI).

**Cluster kits** (a model served across N boxes, e.g. `qwen38-flash-next-cluster-recipe`) extend the
contract with `setup.sh` + `lib.sh`: `run.sh` runs `setup.sh` when no `cluster.env` exists — it
installs an ssh key on the worker, probes both boxes, discovers the interconnect (bound pings across
every non-management interface pair), probes the firewall root-free (listener + connect over the
interconnect; a `ufw allow` is offered, consent-gated, only where a box blocks its peer), and writes
`cluster.env` (machine-specific, gitignored). `recipe.yaml` stays model-only; the cluster flags and
per-box NCCL/gloo pins are composed by `run.sh`. Containers get `--device /dev/infiniband
--cap-add IPC_LOCK --ulimit memlock=-1:-1` or NCCL silently runs TCP; `view.sh` proves RDMA by counters.
