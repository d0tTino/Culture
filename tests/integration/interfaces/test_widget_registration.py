import httpx
import pytest
from httpx import ASGITransport

pytest.importorskip("fastapi")

from src import http_app
from src.extensions import register_widget_backend
from src.interfaces import dashboard_backend as db


@pytest.mark.integration
@pytest.mark.asyncio
async def test_post_widget_adds_to_registry() -> None:
    db.WIDGET_REGISTRY._widgets.clear()
    transport = ASGITransport(app=http_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/register_widget",
            json={"name": "extra", "script_url": "s.js"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert db.WIDGET_REGISTRY.get("extra") == {"script_url": "s.js"}
    assert any(w.get("name") == "extra" for w in data.get("widgets", []))


class DummyClient:
    async def __aenter__(self) -> "DummyClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - interface
        pass

    async def post(self, *args, **kwargs) -> None:
        raise httpx.HTTPError("boom")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_register_widget_backend_network_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: DummyClient())
    with caplog.at_level("ERROR"):
        await register_widget_backend("Widget", "http://x/y.js")
    assert "Widget registration failed" in caplog.text
