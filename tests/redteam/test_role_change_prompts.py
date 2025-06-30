import pytest

from src.agents.core.agent_controller import AgentController
from src.agents.core.agent_state import AgentState


@pytest.mark.redteam
def test_prompt_updates_after_role_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "ROLE_DU_GENERATION", '{"Facilitator": {"base": 1.0}, "Innovator": {"base": 1.0}}'
    )
    state = AgentState(agent_id="a1", name="A", current_role="Facilitator", ip=10.0)
    controller = AgentController(state)
    before = state.role_prompt
    assert controller.change_role("Innovator", current_step=1)
    after = state.role_prompt
    assert state.current_role == "Innovator"
    assert "Embedding:" in after and "reputation:" in after
    assert after != before
