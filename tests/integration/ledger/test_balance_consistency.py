from pathlib import Path

import pytest

from src.agents.core.agent_state import AgentState
from src.infra import llm_client
from src.infra.ledger import Ledger
from tests.utils.mock_llm import MockLLM


@pytest.mark.integration
def test_balance_consistency_after_multiple_actions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr(llm_client, "ledger", ledger)

    state = AgentState(agent_id="A", name="Agent", ip=0.0, du=0.0)
    # Fund the agent
    state.ip += 5.0
    state.du += 10.0
    ledger.log_change(state.agent_id, 5.0, 10.0, "fund")

    with MockLLM({"default": "ok"}):
        llm_client.generate_text("hi", agent_state=state)
        llm_client.generate_text("bye", agent_state=state)

    # Additional manual adjustment
    state.ip -= 1.5
    state.du += 2.0
    ledger.log_change(state.agent_id, -1.5, 2.0, "manual")

    ip, du = ledger.get_balance(state.agent_id)
    assert ip == pytest.approx(state.ip)
    assert du == pytest.approx(state.du)
