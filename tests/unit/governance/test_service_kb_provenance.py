from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.governance.service import GovernanceService

pytestmark = pytest.mark.unit


class DummyBoard:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def add_entry(self, entry, agent_id: str, step: int, vector=None):
        self.entries.append({"entry": entry, "agent_id": agent_id, "step": step})
        return True


def _agent(agent_id: str) -> SimpleNamespace:
    return SimpleNamespace(agent_id=agent_id, state=SimpleNamespace(ip=9.0))


@pytest.mark.asyncio
async def test_propose_law_routes_writes_to_knowledge_board(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = GovernanceService()
    board = DummyBoard()
    service.attach_knowledge_board(board, step_provider=lambda: 42)

    async def _allow(_text: str):
        return True, {}

    monkeypatch.setattr("src.governance.service.evaluate_with_opa", _allow)
    monkeypatch.setattr("src.governance.service.ledger.get_staked_ip", lambda _a: 0.0)
    monkeypatch.setattr(
        "src.governance.service.governance_rules_engine.materialize_from_proposal",
        lambda *a, **k: {"rule_id": "rule-1", "decision": "accepted", "rule_ids": ["rule-1"]},
    )
    monkeypatch.setattr("src.governance.service.ledger.record_law_proposal", lambda *a, **k: None)

    result = await service.propose_law(_agent("A"), "adopt policy", [_agent("A"), _agent("B")])

    assert isinstance(result, dict)
    assert result["proposal_entry_id"]
    entry_types = [item["entry"].entry_type.value for item in board.entries]
    assert "proposal" in entry_types
    assert "vote" in entry_types
    assert "law" in entry_types
    assert all(item["step"] == 42 for item in board.entries)
