"""Integration tests for the authenticated HTTP and WebSocket proxy."""
import asyncio
import json
import unittest

from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer
from studio.access_proxy import setup_app


class ProxyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def homepage(request):
            return web.Response(text="<html>generator working</html>", content_type="text/html")
        async def api(request):
            return web.json_response({"status": "ready"})
        async def websocket(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    await ws.send_str("echo:" + message.data)
                elif message.type == WSMsgType.BINARY:
                    await ws.send_bytes(message.data)
            return ws

        upstream = web.Application()
        upstream.router.add_get("/", homepage)
        upstream.router.add_get("/sdapi/v1/options", api)
        upstream.router.add_get("/ws", websocket)
        self.upstream = TestServer(upstream)
        await self.upstream.start_server()
        port = self.upstream.port

        self.proxy = TestServer(setup_app(port, "a-test-password-with-16-chars", cookie_token="test-session-token"))
        self.client = TestClient(self.proxy)
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.upstream.close()

    def headers(self):
        return {"Cookie": "anime_studio_session=test-session-token"}

    async def test_no_plaintext_unauthorized_response_to_api(self):
        response = await self.client.get("/sdapi/v1/options", allow_redirects=False)
        self.assertEqual(response.status, 401)
        self.assertEqual(await response.json(), {"error": "authentication_required"})
        await response.release()

    async def test_login_once_then_http_api(self):
        form = await self.client.get("/__studio/login")
        self.assertEqual(form.status, 200)
        self.assertIn("Anime Colab Studio", await form.text())
        wrong = await self.client.post("/__studio/login", data={"password": "wrong"},
                                       allow_redirects=False)
        self.assertEqual(wrong.status, 403)
        correct = await self.client.post("/__studio/login",
                                         data={"password": "a-test-password-with-16-chars"},
                                         allow_redirects=False)
        self.assertEqual(correct.status, 302)
        self.assertIn("anime_studio_session", correct.headers.get("Set-Cookie", ""))
        self.assertIn("Secure", correct.headers.get("Set-Cookie", ""))
        self.assertIn("HttpOnly", correct.headers.get("Set-Cookie", ""))
        api = await self.client.get("/sdapi/v1/options", headers=self.headers())
        self.assertEqual(api.status, 200)
        self.assertEqual(await api.json(), {"status": "ready"})
        home = await self.client.get("/", headers=self.headers())
        self.assertIn("generator working", await home.text())

    async def test_websocket_round_trip_through_proxy(self):
        ws = await self.client.ws_connect("/ws", headers=self.headers())
        await ws.send_str("hello")
        answer = await asyncio.wait_for(ws.receive(), timeout=5)
        self.assertEqual(answer.type, WSMsgType.TEXT)
        self.assertEqual(answer.data, "echo:hello")
        await ws.close()

    async def test_proxy_sends_browser_to_login(self):
        response = await self.client.get(
            "/", headers={"Accept": "text/html"}, allow_redirects=False,
        )
        self.assertEqual(response.status, 302)
        self.assertEqual(response.headers["Location"], "/__studio/login")

    async def test_proxy_health_does_not_leak_password(self):
        response = await self.client.get("/__studio/health")
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.json(),
                         {"service": "anime-studio-proxy", "status": "ok"})
