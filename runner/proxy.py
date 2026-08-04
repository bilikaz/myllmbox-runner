"""The keepalive auth proxy — the box's public face.

/v1/models and /health are always public (the model card the platform's connection check probes).
With a BINDING_TOKEN configured, everything else requires `Authorization: Bearer <token>`; with none,
the box runs all-public — the user's explicit choice, made in .env.

SSE responses get `: ping` comments whenever the upstream is quiet, so Cloudflare's ~100s idle cap never
resets a long prefill or a mid-stream stall. Headers go out before the first upstream byte for exactly
that reason — the pings must be able to start during prefill.
"""
from __future__ import annotations

import asyncio
import logging

from aiohttp import ClientSession, ClientTimeout, web

log = logging.getLogger("mbx.proxy")

PUBLIC = {"/v1/models", "/health"}
CHUNK = 8192
# hop-by-hop headers never cross a proxy
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}
# CORS: browser apps send a credential-less OPTIONS preflight before the POST — it must NOT be gated by the
# bearer token, or the preflight 401s and the browser blocks the real request. Answer preflights + echo CORS.
_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type",
    "Access-Control-Max-Age": "86400",
}


def make_app(upstream: str, binding_token: str = "", ping_secs: float = 30.0) -> web.Application:
    app = web.Application()
    app["upstream"] = upstream.rstrip("/")
    app["token"] = binding_token
    app["ping_secs"] = ping_secs

    async def on_startup(a: web.Application) -> None:
        a["session"] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10))

    async def on_cleanup(a: web.Application) -> None:
        await a["session"].close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{tail:.*}", handle)
    return app


async def handle(request: web.Request) -> web.StreamResponse:
    app = request.app
    path = request.rel_url.path
    # CORS preflight: browsers send it without the Authorization header — answer it directly, never gate it.
    if request.method == "OPTIONS":
        return web.Response(status=204, headers=_CORS)
    # no token configured = the user chose an all-public box — no gate anywhere
    if app["token"] and path not in PUBLIC:
        if request.headers.get("Authorization") != f"Bearer {app['token']}":
            return web.Response(status=401, text="Unauthorized", headers=_CORS)

    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
    upstream = await app["session"].request(
        request.method,
        app["upstream"] + str(request.rel_url),
        headers=headers,
        data=request.content if request.body_exists else None,
        allow_redirects=False,
    )

    resp = web.StreamResponse(status=upstream.status)
    for k, v in upstream.headers.items():
        if k.lower() not in _HOP:
            resp.headers[k] = v
    resp.headers.update(_CORS)  # let browser apps read the cross-origin response
    is_sse = "text/event-stream" in upstream.headers.get("Content-Type", "").lower()
    await resp.prepare(request)  # headers out now — pings can start during prefill

    try:
        if not is_sse:
            async for chunk in upstream.content.iter_any():
                await resp.write(chunk)
        else:
            ping_secs = app["ping_secs"]
            while True:
                read_task = asyncio.ensure_future(upstream.content.read(CHUNK))
                while True:
                    done, _ = await asyncio.wait({read_task}, timeout=ping_secs)
                    if read_task in done:
                        break
                    await resp.write(b": ping\n\n")  # upstream quiet — keep CF's idle timer alive
                chunk = read_task.result()
                if not chunk:
                    break
                await resp.write(chunk)
    finally:
        upstream.release()
    await resp.write_eof()
    return resp


def main() -> None:  # `python -m runner.proxy` — how the supervisor spawns it
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    upstream = os.environ.get("MBX_UPSTREAM", "http://127.0.0.1:8000")
    port = int(os.environ.get("MBX_PROXY_PORT", "8011"))
    token = os.environ.get("BINDING_TOKEN", "")
    ping = float(os.environ.get("MBX_PING_SECS", "30"))
    if not token:
        log.warning("no BINDING_TOKEN — serving FULLY PUBLIC, generation included")
    web.run_app(make_app(upstream, token, ping), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
