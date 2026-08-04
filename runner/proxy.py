"""The keepalive auth proxy — the box's public face.

Routing:
  · /v1/* (and /v1/models, /health) → the MODEL (vLLM / generic server). Bearer BINDING_TOKEN gates it
    (except the public probes); SSE gets keepalive pings so Cloudflare's ~100s idle cap never cuts a stream.
  · everything else → the DASHBOARD (sparkDash on :5555), IF a DASHBOARD_PASSWORD is set — behind HTTP Basic
    auth (any username, that password). This is what makes the dashboard's otherwise-unauthenticated UI +
    power actions safe to reach through the tunnel. WebSockets (sparkDash's /ws) are proxied too.
  With no dashboard configured, the proxy behaves exactly as before: everything → the model.

Headers go out before the first upstream byte so the SSE pings can start during prefill.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import logging

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

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


def _to_model(path: str) -> bool:
    """The model owns the OpenAI API surface (/v1/* + the public probes); everything else is the dashboard's."""
    return path.startswith("/v1") or path in PUBLIC


def make_app(upstream: str, binding_token: str = "", ping_secs: float = 30.0,
             dashboard: str = "", dashboard_password: str = "") -> web.Application:
    app = web.Application()
    app["upstream"] = upstream.rstrip("/")
    app["token"] = binding_token
    app["ping_secs"] = ping_secs
    app["dashboard"] = dashboard.rstrip("/")          # "" → no dashboard, everything goes to the model
    app["dash_pw"] = dashboard_password

    async def on_startup(a: web.Application) -> None:
        a["session"] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10))

    async def on_cleanup(a: web.Application) -> None:
        await a["session"].close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{tail:.*}", handle)
    return app


def _basic_ok(request: web.Request, password: str) -> bool:
    h = request.headers.get("Authorization", "")
    if not h.startswith("Basic "):
        return False
    try:
        _user, _, passwd = base64.b64decode(h[6:]).decode("utf-8", "replace").partition(":")
    except (binascii.Error, ValueError):
        return False
    return passwd == password   # any username; the password is the secret


async def _ws_proxy(request: web.Request, base: str) -> web.StreamResponse:
    """Relay a WebSocket both ways (sparkDash streams metrics over /ws)."""
    ws_server = web.WebSocketResponse()
    await ws_server.prepare(request)
    url = ("wss" if base.startswith("https") else "ws") + base[base.index(":"):] + str(request.rel_url)
    session = request.app["session"]
    async with session.ws_connect(url) as ws_client:
        async def c2s() -> None:
            async for m in ws_client:
                if m.type == WSMsgType.TEXT:
                    await ws_server.send_str(m.data)
                elif m.type == WSMsgType.BINARY:
                    await ws_server.send_bytes(m.data)
                else:
                    break

        async def s2c() -> None:
            async for m in ws_server:
                if m.type == WSMsgType.TEXT:
                    await ws_client.send_str(m.data)
                elif m.type == WSMsgType.BINARY:
                    await ws_client.send_bytes(m.data)
                else:
                    break

        await asyncio.gather(c2s(), s2c(), return_exceptions=True)
    return ws_server


async def _dashboard(request: web.Request) -> web.StreamResponse:
    app = request.app
    if app["dash_pw"] and not _basic_ok(request, app["dash_pw"]):
        return web.Response(status=401, text="dashboard login required",
                            headers={"WWW-Authenticate": 'Basic realm="myllmbox dashboard"'})
    base = app["dashboard"]
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return await _ws_proxy(request, base)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
    up = await app["session"].request(
        request.method, base + str(request.rel_url), headers=headers,
        data=request.content if request.body_exists else None, allow_redirects=False)
    resp = web.StreamResponse(status=up.status)
    for k, v in up.headers.items():
        if k.lower() not in _HOP:
            resp.headers[k] = v
    await resp.prepare(request)
    try:
        async for chunk in up.content.iter_any():
            await resp.write(chunk)
    finally:
        up.release()
    await resp.write_eof()
    return resp


async def handle(request: web.Request) -> web.StreamResponse:
    app = request.app
    path = request.rel_url.path
    # dashboard owns everything that isn't the model's OpenAI API surface (only when one is configured)
    if app["dashboard"] and not _to_model(path):
        return await _dashboard(request)

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
    dashboard = os.environ.get("MBX_DASHBOARD", "")          # e.g. http://127.0.0.1:5555 (sparkDash)
    dash_pw = os.environ.get("DASHBOARD_PASSWORD", "")
    if not token:
        log.warning("no BINDING_TOKEN — serving FULLY PUBLIC, generation included")
    if dashboard and not dash_pw:
        log.warning("MBX_DASHBOARD set without DASHBOARD_PASSWORD — dashboard would be PUBLIC; not routing it")
        dashboard = ""
    web.run_app(make_app(upstream, token, ping, dashboard, dash_pw), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
