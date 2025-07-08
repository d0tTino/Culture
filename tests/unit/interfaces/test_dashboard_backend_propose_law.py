import json

import pytest

pytest.importorskip("pytest_asyncio")

from src.interfaces import dashboard_backend as db


class DummySim:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def propose_law(self, proposer_id: str, text: str) -> bool:
        self.calls.append((proposer_id, text))
        return True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_propose_law(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = DummySim()
    monkeypatch.setitem(db.SIM_STATE, "simulation", sim)
    resp = await db.api_propose_law(db.LawProposal(proposer_id="a1", text="t"))
    data = json.loads(resp.body)
    assert data == {"approved": True}
    assert sim.calls == [("a1", "t")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_propose_law_no_sim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(db.SIM_STATE, "simulation", None)
    resp = await db.api_propose_law(db.LawProposal(proposer_id="a1", text="t"))
    data = json.loads(resp.body)
    assert data == {"approved": False}
