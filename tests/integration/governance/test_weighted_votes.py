import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from src.governance.law_board import LawBoard
from src.infra.ledger import Ledger


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
async def test_weighted_votes_deduct_ip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    board = LawBoard(tmp_path / "laws.sqlite")

    gservice = importlib.import_module("src.governance.service")
    voting_mod = importlib.import_module("src.governance.voting")
    monkeypatch.setattr(gservice, "law_board", board)
    monkeypatch.setattr(gservice, "ledger", ledger)
    monkeypatch.setattr(voting_mod, "governance", gservice.governance)

    async def allow(_: str) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr(gservice, "evaluate_with_opa", allow)

    votes = [True, False, False]

    async def fake_vote(_agent: DummyAgent, _text: str) -> bool:
        return votes.pop(0)

    monkeypatch.setattr(gservice.governance, "vote", fake_vote)

    for a in ("a1", "a2", "a3"):
        ledger.log_change(a, 10.0, 0.0, "fund")

    agents = [DummyAgent("a1"), DummyAgent("a2"), DummyAgent("a3")]
    weights = {"a1": 3, "a2": 1, "a3": 1}

    approved = await gservice.governance.propose_law(
        agents[0], "law", agents, vote_weights=weights
    )
    assert approved is True

    assert ledger.get_balance("a1")[0] == pytest.approx(1.0)
    assert ledger.get_balance("a2")[0] == pytest.approx(9.0)
    assert ledger.get_balance("a3")[0] == pytest.approx(9.0)

    proposals = ledger.get_law_proposals()
    assert proposals[0]["yes_weight"] == pytest.approx(3.0)
    assert proposals[0]["no_weight"] == pytest.approx(2.0)
