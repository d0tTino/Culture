import asyncio
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
async def test_weighted_vote_records_spend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    ledger._hooks = [ledger._db_hook]  # keep DB hook only for thread safety
    orig_spend = ledger.spend
    lock = asyncio.Lock()

    async def locked_spend(*args, **kwargs):
        async with lock:
            return await orig_spend(*args, **kwargs)

    monkeypatch.setattr(ledger, "spend", locked_spend)
    board = LawBoard(tmp_path / "laws.sqlite")

    gservice = importlib.import_module("src.governance.service")
    voting_mod = importlib.import_module("src.governance.voting")
    monkeypatch.setattr(gservice, "law_board", board)
    monkeypatch.setattr(gservice, "ledger", ledger)
    monkeypatch.setattr(voting_mod, "governance", gservice.governance)

    async def allow(_: str) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr(gservice, "evaluate_with_opa", allow)

    vote_order = [True, False]

    async def fake_vote(_agent: DummyAgent, _text: str) -> bool:
        return vote_order.pop(0)

    monkeypatch.setattr(gservice.governance, "vote", fake_vote)

    # seed the ledger with different IP balances for each agent
    await ledger.reward("a1", ip=12.0, reason="fund")
    await ledger.reward("a2", ip=5.0, reason="fund")

    agents = [DummyAgent("a1"), DummyAgent("a2")]
    weights = {"a1": 2, "a2": 1}

    result = await gservice.governance.propose_law(agents[0], "law", agents, vote_weights=weights)
    assert result["approved"] is True

    assert ledger.get_balance("a1")[0] == pytest.approx(8.0)
    assert ledger.get_balance("a2")[0] == pytest.approx(4.0)

    proposals = ledger.get_law_proposals()
    assert proposals[0]["ip_spent"] == pytest.approx(5.0)
    assert proposals[0]["yes_weight"] == pytest.approx(2.0)
    assert proposals[0]["no_weight"] == pytest.approx(1.0)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_weighted_vote_overdraw(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    ledger._hooks = [ledger._db_hook]  # keep DB hook only for thread safety
    orig_spend = ledger.spend
    lock = asyncio.Lock()

    async def locked_spend(*args, **kwargs):
        async with lock:
            return await orig_spend(*args, **kwargs)

    monkeypatch.setattr(ledger, "spend", locked_spend)
    board = LawBoard(tmp_path / "laws.sqlite")

    gservice = importlib.import_module("src.governance.service")
    voting_mod = importlib.import_module("src.governance.voting")
    monkeypatch.setattr(gservice, "law_board", board)
    monkeypatch.setattr(gservice, "ledger", ledger)
    monkeypatch.setattr(voting_mod, "governance", gservice.governance)

    async def allow(_: str) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr(gservice, "evaluate_with_opa", allow)

    vote_order = [True, False]

    async def fake_vote(_agent: DummyAgent, _text: str) -> bool:
        return vote_order.pop(0)

    monkeypatch.setattr(gservice.governance, "vote", fake_vote)

    # each agent has less IP than their weight squared
    await ledger.reward("a1", ip=3.0, reason="fund")
    await ledger.reward("a2", ip=1.0, reason="fund")

    agents = [DummyAgent("a1"), DummyAgent("a2")]
    weights = {"a1": 3, "a2": 1}

    result = await gservice.governance.propose_law(agents[0], "law", agents, vote_weights=weights)
    assert result["approved"] is True

    # IP cannot go negative, so both balances hit zero
    assert ledger.get_balance("a1")[0] == pytest.approx(0.0)
    assert ledger.get_balance("a2")[0] == pytest.approx(0.0)

    proposals = ledger.get_law_proposals()
    # only the available IP is deducted and persisted
    assert proposals[0]["ip_spent"] == pytest.approx(4.0)
    assert proposals[0]["yes_weight"] == pytest.approx(3.0)
    assert proposals[0]["no_weight"] == pytest.approx(1.0)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_quadratic_weighting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    vote_order = [True, False]

    async def fake_vote(_agent: DummyAgent, _text: str) -> bool:
        return vote_order.pop(0)

    monkeypatch.setattr(gservice.governance, "vote", fake_vote)

    # quadratic weights derived from agent state + ledger balance
    await ledger.reward("a1", ip=16.0, reason="fund")
    await ledger.reward("a2", ip=1.0, reason="fund")

    agents = [DummyAgent("a1"), DummyAgent("a2")]
    agents[0].state.ip = 16.0
    agents[1].state.ip = 1.0

    result = await gservice.governance.propose_law(agents[0], "law", agents)
    assert result["approved"] is True

    # no IP is spent when vote_weights is None
    assert ledger.get_balance("a1")[0] == pytest.approx(16.0)
    assert ledger.get_balance("a2")[0] == pytest.approx(1.0)

    proposals = ledger.get_law_proposals()
    assert proposals[0]["ip_spent"] == pytest.approx(0.0)
    assert proposals[0]["yes_weight"] == pytest.approx(4.0)
    assert proposals[0]["no_weight"] == pytest.approx(1.0)
