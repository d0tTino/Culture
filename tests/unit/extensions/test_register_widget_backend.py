from __future__ import annotations

import types

import httpx
import pytest

pytest.importorskip("fastapi")

from src.extensions import register_widget_backend


class DummyClient:
    async def __aenter__(self) -> DummyClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        pass

    async def post(self, *args: object, **kwargs: object) -> None:
        raise httpx.HTTPError("boom")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_widget_backend_handles_http_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: DummyClient())
    with caplog.at_level("ERROR"):
        await register_widget_backend("Widget", "http://x/y.js")
    assert "Widget registration failed" in caplog.text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_widget_backend_persists_widget(monkeypatch: pytest.MonkeyPatch) -> None:
    from httpx import ASGITransport

    from src import http_app
    from src.interfaces import dashboard_backend as db

    db.WIDGET_REGISTRY._widgets.clear()
    transport = ASGITransport(app=http_app.app)

    orig_client = httpx.AsyncClient

    class ClientWrapper:
        def __init__(self) -> None:
            self.client = orig_client(transport=transport, base_url="http://backend")

        async def __aenter__(self) -> httpx.AsyncClient:
            return self.client

        async def __aexit__(self, exc_type, exc, tb) -> None:
            await self.client.aclose()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: ClientWrapper())

    await register_widget_backend(
        "ExtraWidget",
        "http://x/y.js",
        backend_url="http://backend",
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://backend") as client:
        resp = await client.post("/api/register_widget", json={})
    data = resp.json()

    assert db.WIDGET_REGISTRY.get("ExtraWidget") == {"script_url": "http://x/y.js"}
    assert any(w.get("name") == "ExtraWidget" for w in data.get("widgets", []))
