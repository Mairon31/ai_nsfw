"""Cookie-authenticated proxy: no email codes; HTTP and WebSocket support."""
from __future__ import annotations
import asyncio
import hmac
import os
import secrets
import time
from collections import defaultdict
from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

COOKIE = "anime_studio_session"
HOP = {"connection", "keep-alive", "proxy-connection", "proxy-authenticate",
       "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"}

def setup_app(target_port: int, password: str, cookie_token: str | None = None) -> web.Application:
    if len(password) < 10:
        raise ValueError("La contraseña de acceso debe tener al menos 10 caracteres.")
    if not 0 < int(target_port) <= 65535:
        raise ValueError("Puerto interno inválido.")
    token = cookie_token or secrets.token_urlsafe(36)
    attempts = defaultdict(list)
    app = web.Application(client_max_size=128 * 1024 ** 2)

    async def client_context(app):
        app["client"] = ClientSession(timeout=ClientTimeout(total=None), auto_decompress=False)
        yield
        await app["client"].close()
    app.cleanup_ctx.append(client_context)

    def permitted(request):
        return hmac.compare_digest(request.cookies.get(COOKIE, ""), token)

    async def health(request):
        return web.json_response({"service": "anime-studio-proxy", "status": "ok"})

    async def login_page(request):
        if permitted(request):
            raise web.HTTPFound("/")
        page = """<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Anime Colab Studio</title>
<style>html{color-scheme:dark}body{background:#0d1117;color:#e6edf3;font-family:system-ui;display:grid;place-items:center;min-height:95vh}
main{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:30px;width:min(380px,85vw)}
input,button{font:inherit;box-sizing:border-box;width:100%;padding:12px;border-radius:8px}input{background:#0d1117;color:#fff;border:1px solid #555}
button{background:#2979ff;color:#fff;border:0;margin-top:18px;cursor:pointer}small{color:#aaa}</style></head>
<body><main><h2>Anime Colab Studio</h2><p>Introduce la contraseña de acceso que aparece en tu notebook de Colab.</p>
<form action="/__studio/login" method="post"><label for="password">Contraseña</label>
<input name="password" id="password" type="password" minlength="10" required autocomplete="current-password">
<button type="submit">Entrar</button></form>
<p><small>Acceso una vez por navegador. Sin códigos por correo.</small></p></main></body></html>"""
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def login(request):
        who = (request.headers.get("CF-Connecting-IP") or request.remote or "unknown")[:100]
        attempts[who] = [t for t in attempts[who] if time.monotonic() - t < 120]
        if len(attempts[who]) >= 12:
            return web.Response(text="Demasiados intentos.", status=429)
        try:
            body = await request.post()
            entered = str(body.get("password", ""))
        except (ValueError, UnicodeError):
            entered = ""
        if not hmac.compare_digest(entered, password):
            attempts[who].append(time.monotonic())
            return web.Response(text="Contraseña incorrecta.", status=403)
        attempts.pop(who, None)
        result = web.HTTPFound("/")
        result.set_cookie(COOKIE, token, httponly=True, secure=True, samesite="Lax",
                          path="/", max_age=43200)
        raise result

    async def logout(request):
        result = web.HTTPFound("/__studio/login")
        result.del_cookie(COOKIE, path="/")
        raise result

    async def proxy(request):
        if not permitted(request):
            if request.method == "GET" and "text/html" in request.headers.get("Accept", "") and request.headers.get("Upgrade", "").lower() != "websocket":
                raise web.HTTPFound("/__studio/login")
            return web.json_response({"error": "authentication_required"}, status=401)

        url = f"http://127.0.0.1:{target_port}" + str(request.rel_url)
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in HOP
                   and k.lower() not in {"host", "cookie", "authorization", "content-length", "accept-encoding"}
                   and not k.lower().startswith("sec-websocket-")}
        headers["Accept-Encoding"] = "identity"
        headers["X-Forwarded-Proto"] = "https"
        headers["X-Forwarded-Host"] = request.host

        if request.headers.get("Upgrade", "").lower() == "websocket":
            offered = [p.strip() for p in request.headers.get("Sec-WebSocket-Protocol", "").split(",") if p.strip()]
            async with app["client"].ws_connect(url, protocols=offered, headers=headers, heartbeat=30) as upstream:
                downstream = web.WebSocketResponse(protocols=offered, heartbeat=30, max_msg_size=32 * 1024 ** 2)
                await downstream.prepare(request)
                async def forward_to_upstream():
                    async for message in downstream:
                        if message.type == WSMsgType.TEXT:
                            await upstream.send_str(message.data)
                        elif message.type == WSMsgType.BINARY:
                            await upstream.send_bytes(message.data)
                        elif message.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                            break
                    await upstream.close()
                async def forward_to_client():
                    async for message in upstream:
                        if message.type == WSMsgType.TEXT:
                            await downstream.send_str(message.data)
                        elif message.type == WSMsgType.BINARY:
                            await downstream.send_bytes(message.data)
                        elif message.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                            break
                    await downstream.close()
                tasks = [asyncio.create_task(forward_to_upstream()), asyncio.create_task(forward_to_client())]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                return downstream

        async with app["client"].request(request.method, url, headers=headers,
                                         data=request.content if request.can_read_body else None,
                                         allow_redirects=False) as upstream:
            copied_headers = {k: v for k, v in upstream.headers.items()
                              if k.lower() not in HOP and k.lower() not in {"content-length"}}
            response = web.StreamResponse(status=upstream.status, headers=copied_headers)
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(256 * 1024):
                await response.write(chunk)
            await response.write_eof()
            return response

    app.router.add_get("/__studio/health", health)
    app.router.add_get("/__studio/login", login_page)
    app.router.add_post("/__studio/login", login)
    app.router.add_get("/__studio/logout", logout)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app

def main():
    target = int(os.environ["ANIME_STUDIO_ORIGIN_PORT"])
    port = int(os.environ.get("ANIME_STUDIO_PROXY_PORT", "8787"))
    password = os.environ["ANIME_STUDIO_ACCESS_PASSWORD"]
    app = setup_app(target, password)
    print(f"[ACCESS PROXY] Listo en 127.0.0.1:{port} (origen local {target})", flush=True)
    web.run_app(app, host="127.0.0.1", port=port, access_log=None)

if __name__ == "__main__":
    main()
