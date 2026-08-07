---
name: add-dashboard
description: Scaffold a myllmbox dashboard — a web UI the proxy fronts at the box's public URL. Writes dashboards/<name>/ (dashboard.yaml + up.sh + down.sh) to the house contract. Use when the user wants to add/port a monitoring or control UI for the box.
---

# Add a dashboard — a web UI at your box's public URL

A dashboard is a folder under `dashboards/<name>/` that a recipe opts into with `dashboard: <name>`.
The proxy then routes `/v1/*` to the model and **every other path** (incl. WebSockets) to the UI at
`127.0.0.1:<port>`, HTTP-Basic-gated by `DASHBOARD_PASSWORD` from `.env`. It's a **pattern, not a
product** — any web UI plugs in. Reference implementation: `dashboards/sparkdash/`. The living
contract doc is `dashboards/README.md` — keep it and this skill in sync.

## The contract — three files

| File | Job |
|---|---|
| `dashboard.yaml` | `port:` (what the proxy forwards to; default 5555) + optional `description:`, `env: {}` |
| `up.sh` | bring the UI up on `$PORT` — do WHATEVER it needs (clone, build, generate config, `docker run`). Must leave something listening on **`127.0.0.1:$PORT`**. Runs on the head. |
| `down.sh` | tear it ALL down — the head container AND anything `up.sh` started on other boxes. Omit it only if `docker rm -f mbx-dashboard` is enough. |

`up.sh` receives in the env: **`PORT`**, **`DASHBOARD_PASSWORD`** (may be empty — the proxy enforces
it, not the UI), and **`MBX_BOXES`** — a JSON array of this recipe's boxes
(`[{"name","role":"head|worker","host","interconnect","ssh_user"}, …]`; 1 entry single-node, N for a
cluster). A monitoring UI turns `MBX_BOXES` into its own config; a plain page ignores it.

## House rules (each one earned the hard way — don't relax them)

1. **Container name `mbx-dashboard`** — makes the default teardown work and never collides with the
   model container (`mbx-vllm`).
2. **Bind `127.0.0.1`, never `0.0.0.0`** — the proxy is the door; a LAN-open UI bypasses the
   password gate. If the UI needs host networking (e.g. to probe `127.0.0.1:8000`), set its own
   bind-host env to `127.0.0.1`.
3. **All runtime junk in `dashboards/<name>/.data/`** — clones, builds, generated config, keys. It's
   the one gitignored dir per dashboard; nothing machine-specific is ever committed.
4. **Regenerate machine config from `$MBX_BOXES` on EVERY run** (the box set changes run to run) —
   but skip regeneration when `MBX_BOXES` is empty/`[]` (a manual re-run must not wipe config).
5. **SSH to other boxes uses the dedicated, disposable key** (`~/.ssh/id_myllmbox`, created by
   `cluster/setup.sh`) — NEVER the admin key, and never mount host `~/.ssh` into a container
   (uid mismatch → ssh "bad owner"; copy the key in root-owned instead, see sparkdash's `up.sh`).
6. **A dashboard failing must never block the model serve** — the runner already treats it as
   optional (logs a warning and serves without it); keep `up.sh` fail-fast and side-effect-free on
   failure.
7. If the UI has **power/destructive actions** (shutdown buttons…), say so in `dashboard.yaml`'s
   comment and strongly advise setting `DASHBOARD_PASSWORD`.

## Scaffold

```
dashboards/<name>/
  dashboard.yaml      # port: 5555  + description (+ what it does, auth posture, data it can touch)
  up.sh               # set -euo pipefail; build/pull once (cache under .data/); docker rm -f first;
                      #   docker run -d --name mbx-dashboard … listening on 127.0.0.1:$PORT
  down.sh             # docker rm -f mbx-dashboard (+ anything started on other boxes)
```

Then the recipe opts in: `dashboard: <name>` in its `myllmbox.yaml`. The runner records the active
dashboard in `.mbx/.running` so `down` calls the right `down.sh` — no registration step.

## Verify

1. Standalone: `PORT=5555 MBX_BOXES='[...]' bash dashboards/<name>/up.sh` → `curl -s 127.0.0.1:5555`
   answers; `bash down.sh` removes everything (`docker ps` clean).
2. Through the front: run a recipe with `dashboard: <name>` and `DASHBOARD_PASSWORD` set → the public
   URL serves the UI behind Basic auth, `/v1/*` still hits the model, and (if the UI streams) its
   WebSocket path works through the proxy.
3. `stop.sh` / `runner.cli down` tears the dashboard down with the box.
