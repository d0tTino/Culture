import importlib
import sys

import httpx
import pytest
from httpx import ASGITransport

pytest.importorskip("fastapi")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_token_required(monkeypatch):
    monkeypatch.setenv("DASHBOARD_API_TOKEN", "secret")
    for mod in ["src.http_app", "src.interfaces.dashboard_backend"]:
        if mod in sys.modules:
            del sys.modules[mod]
    http_app = importlib.import_module("src.http_app")
    transport = ASGITransport(app=http_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/propose", json={"proposer_id": "a1", "text": "law"})
        assert resp.status_code == 401
        resp = await client.post(
            "/api/propose",
            json={"proposer_id": "a1", "text": "law"},
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 200
