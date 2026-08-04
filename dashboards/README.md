# dashboards/

Optional web UIs the proxy fronts at your public URL. A recipe opts in with `dashboard: <name>`, pointing here
at `dashboards/<name>/`. The proxy then routes `/v1/*` to the model and **every other path** to the dashboard
(see `runner/proxy.py`); `DASHBOARD_PASSWORD` in `.env` HTTP-Basic-gates it (unset → served ungated — the UI's
own guard rails, or your explicit public choice). No `dashboard:` in a recipe → the box is model-only.

It's a **pattern, not a product**: you run mia (sparkDash), someone else a "robot", someone else their own — the
runner is UI-agnostic. What loads is the recipe's taste.

## The contract — each `dashboards/<name>/` folder has:

- **`dashboard.yaml`** — `port:` (what the proxy forwards to; default 5555), optional `description:` and `env: {}`.
- **`up.sh`** — brings the UI up on `$PORT`. Do WHATEVER it needs: build an image, generate config, `docker run`
  a web container. Must leave something listening on `127.0.0.1:$PORT`. Runs on the head.
- **`down.sh`** — tears it ALL down (the head container **and** anything `up.sh` started on other boxes).
  Omit it and the default is `docker rm -f mbx-dashboard`.

## What the runner passes `up.sh` (env):

- `PORT` — the port the proxy forwards to (from `dashboard.yaml`).
- `DASHBOARD_PASSWORD` — the `.env` password (may be empty; the proxy, not the UI, enforces it).
- `MBX_BOXES` — JSON array of **this recipe's** boxes (1 for single-node, N for a cluster):
  `[{"name","role":"head|worker","host","interconnect","ssh_user"}, …]`. A UI that monitors the cluster (mia)
  turns this into its own config; a plain page ignores it.

The runner records the active dashboard in `.mbx/.running`, so `down` calls the right `down.sh`. **Name your
container `mbx-dashboard`** so the default teardown works and it never collides with the model container.
