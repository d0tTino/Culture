import httpx
import pytest
from httpx import ASGITransport

pytest.importorskip("fastapi")

from src import http_app
from src.governance.law_board import LawBoard
from src.infra.ledger import Ledger
from src.interfaces import dashboard_backend as db


@pytest.mark.asyncio
@pytest.mark.integration
async def test_laws_and_votes_endpoints(monkeypatch: pytest.MonkeyPatch, tmp_path):
    board = LawBoard(tmp_path / "laws.sqlite")
    board.add_law("be nice")

    ledger = Ledger(tmp_path / "ledger.sqlite")
    ledger.record_law_proposal("a1", "be nice", True, 1.0, 0.0, 0.0)

    monkeypatch.setattr(db, "law_board", board)
    monkeypatch.setattr(db, "ledger", ledger)

    transport = ASGITransport(app=http_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/laws")
        assert resp.status_code == 200
        assert resp.json()["laws"] == ["be nice"]

        resp = await client.get("/api/votes")
        assert resp.status_code == 200
        votes = resp.json().get("votes")
        assert votes and votes[0]["text"] == "be nice"
