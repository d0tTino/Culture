import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from src.governance.law_board import LawBoard
from src.infra import config
from src.infra.ledger import Ledger
from src.interfaces import dashboard_backend as db


class DummyState(SimpleNamespace):
    ip: float = 1.0
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
async def test_weights_include_staked_ip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    board = LawBoard(tmp_path / "laws.sqlite")

    voting_mod = importlib.import_module("src.governance.voting")
    gservice = importlib.import_module("src.governance.service")
    monkeypatch.setattr(gservice, "law_board", board)
    monkeypatch.setattr(gservice, "ledger", ledger)
    monkeypatch.setattr(voting_mod, "governance", gservice.governance)
    monkeypatch.setattr(db, "governance", gservice.governance)

    monkeypatch.setitem(config._CONFIG, "OPA_URL", "http://opa")

    async def allow_policy(_p: str) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr(gservice, "evaluate_with_opa", allow_policy)

    votes = [True, False]

    async def fake_vote(_a: DummyAgent, _t: str) -> bool:
        return votes.pop(0)

    monkeypatch.setattr(gservice.governance, "vote", fake_vote)

    ledger.log_change("a1", 10.0, 0.0, "fund")
    ledger.log_change("a2", 10.0, 0.0, "fund")
    ledger.stake_ip("a1", 9.0)

    agents = [DummyAgent("a1"), DummyAgent("a2")]

    approved = await voting_mod.propose_law(agents[0], "law", agents)
    assert approved is True

    proposals = ledger.get_law_proposals()
    assert proposals and proposals[0]["approved"] is True
    assert proposals[0]["ip_spent"] == pytest.approx(0.0)

    resp = await db.api_get_proposals(limit=1)
    data = json.loads(resp.body)
    assert data["proposals"][0]["approved"] is True
