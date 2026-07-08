import importlib
import sys
import types
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from httpx import ASGITransport


class _TestFastAPI:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self._routes: dict[tuple[str, str], Callable[..., Any]] = {}
        self._middleware: Callable[..., Any] | None = None

    async def __call__(
        self, scope: dict[str, Any], receive: object, send: Callable[..., Any]
    ) -> None:
        method = scope.get("method", "GET")
        path = scope.get("path", "")

        class Request:
            def __init__(self) -> None:
                self.method = method
                self.headers = {
                    key.decode().title(): value.decode() for key, value in scope.get("headers", [])
                }

            async def is_disconnected(self) -> bool:
                return True

        async def call_next(_: Request) -> _TestJSONResponse:
            status = 200 if (method, path) in self._routes else 404
            return _TestJSONResponse({}, status_code=status)

        request = Request()
        response = (
            await self._middleware(request, call_next)
            if self._middleware
            else await call_next(request)
        )
        await send({"type": "http.response.start", "status": response.status_code, "headers": []})
        await send({"type": "http.response.body", "body": response.body, "more_body": False})

    def get(
        self, path: str, *args: object, **kwargs: object
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._route("GET", path)

    def post(
        self, path: str, *args: object, **kwargs: object
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._route("POST", path)

    def websocket(
        self, path: str, *args: object, **kwargs: object
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._route("WEBSOCKET", path)

    def middleware(
        self, *args: object, **kwargs: object
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._middleware = fn
            return fn

        return decorator

    def _route(self, method: str, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._routes[(method, path)] = fn
            return fn

        return decorator


class _TestJSONResponse:
    def __init__(self, content: object = b"", *args: object, **kwargs: object) -> None:
        self.status_code = int(kwargs.get("status_code", 200))
        self.body = b"{}"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_documented_dashboard_entrypoint_requires_token_for_post(monkeypatch):
    """The documented dashboard app entrypoint must configure token middleware."""
    fastapi_mod = types.ModuleType("fastapi")
    fastapi_mod.FastAPI = _TestFastAPI
    fastapi_mod.Request = object
    fastapi_mod.Response = object
    fastapi_mod.WebSocket = object
    fastapi_mod.WebSocketDisconnect = Exception
    responses_mod = types.ModuleType("fastapi.responses")
    responses_mod.JSONResponse = _TestJSONResponse
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_mod)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses_mod)
    monkeypatch.setenv("DASHBOARD_API_TOKEN", "secret")
    for mod in ["src.http_app", "src.interfaces.dashboard_backend"]:
        if mod in sys.modules:
            del sys.modules[mod]
    http_app = importlib.import_module("src.http_app")
    transport = ASGITransport(app=http_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/governance/propose",
            json={"proposer_id": "a1", "text": "law"},
        )
        assert resp.status_code == 401
        resp = await client.post(
            "/api/governance/propose",
            json={"proposer_id": "a1", "text": "law"},
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 200
