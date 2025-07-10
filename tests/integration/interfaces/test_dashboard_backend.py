import httpx
import pytest
from httpx import ASGITransport

pytest.importorskip("fastapi")

from src import http_app
from src.interfaces import dashboard_backend as db


class FailingManager:
    def get_semantic_summaries(self, agent_id: str, limit: int = 3) -> list[str]:
        raise RuntimeError("boom")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_semantic_summaries_error_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(db.SIM_STATE, "semantic_manager", FailingManager())
    transport = ASGITransport(app=http_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/agents/agent-1/semantic_summaries")
    assert resp.status_code == 500
    data = resp.json()
    assert data["error"] == "summary retrieval failed"
