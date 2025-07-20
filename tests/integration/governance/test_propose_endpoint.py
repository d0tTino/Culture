import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

pytest.importorskip("fastapi")

from src.governance.law_board import LawBoard
from src.infra.ledger import Ledger
from src.interfaces import dashboard_backend as db


class DummyState(SimpleNamespace):
    ip: float = 0.0
    du: float = 0.0
    age: int = 0
    is_alive: bool = True
    inheritance: float = 0.0
    short_term_memory: ClassVar[list] = []
    messages_sent_count: int = 0
    last_message_step: int = 0
    relationships: ClassVar[dict] = {}
    current_role: str = "dummy"
    steps_in_current_role: int = 0

    def update_collective_metrics(self, ip: float, du: float) -> None:
        pass


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = DummyState()

    def get_id(self) -> str:
        return self.agent_id

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
    ) -> dict:
        return {}

    def update_state(self, new_state: DummyState) -> None:
        self.state = new_state


@pytest.mark.asyncio
@pytest.mark.integration
async def test_propose_endpoint_records_spent_ip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    board = LawBoard(tmp_path / "laws.sqlite")

    gservice = importlib.import_module("src.governance.service")
    monkeypatch.setattr(gservice, "ledger", ledger)
    monkeypatch.setattr(gservice, "law_board", board)
    monkeypatch.setattr(db, "ledger", ledger)
    monkeypatch.setattr(db, "law_board", board)
    monkeypatch.setattr(db, "governance", gservice.governance)

    async def allow(_: str) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr(gservice, "evaluate_with_opa", allow)

    votes = [True, True, False]

    async def fake_vote(_agent: DummyAgent, _text: str) -> bool:
        return votes.pop(0)

    monkeypatch.setattr(gservice.governance, "vote", fake_vote)

    from src.sim.simulation import Simulation

    agents = [DummyAgent("a1"), DummyAgent("a2"), DummyAgent("a3")]
    sim = Simulation(agents=agents)
    monkeypatch.setitem(db.SIM_STATE, "simulation", sim)

    for a in agents:
        ledger.log_change(a.agent_id, 5.0, 0.0, "fund")

    weights = {"a1": 3, "a2": 1, "a3": 1}

    resp = await db.api_propose(db.Proposal(proposer_id="a1", text="law", vote_weights=weights))
    data = json.loads(resp.body)
    assert data["approved"] is True

    assert ledger.get_balance("a1")[0] == pytest.approx(0.0)
    assert ledger.get_balance("a2")[0] == pytest.approx(4.0)
    assert ledger.get_balance("a3")[0] == pytest.approx(4.0)

    proposals = ledger.get_law_proposals()
    assert proposals[0]["ip_spent"] == pytest.approx(7.0)
    assert proposals[0]["yes_weight"] == pytest.approx(4.0)
    assert proposals[0]["no_weight"] == pytest.approx(1.0)
