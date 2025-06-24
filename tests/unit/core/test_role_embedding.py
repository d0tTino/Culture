import pytest

from src.agents.core.agent_controller import AgentController
from src.agents.core.agent_state import AgentState


@pytest.mark.unit
def test_role_embedding_initialized() -> None:
    s1 = AgentState(agent_id="a1", name="A")
    s2 = AgentState(agent_id="a2", name="B")
    assert len(s1.role_embedding) == 8
    assert len(s2.role_embedding) == 8
    assert s1.role_embedding != s2.role_embedding


@pytest.mark.unit
def test_gossip_update_changes_embedding() -> None:
    state = AgentState(agent_id="a", name="A")
    controller = AgentController(state)
    original = list(state.role_embedding)
    other = [v + 0.5 for v in original]
    controller.gossip_update(other, 1.0)
    assert state.role_embedding != original


@pytest.mark.unit
def test_role_prompt_contains_embedding() -> None:
    state = AgentState(agent_id="a", name="A")
    state.reputation["b"] = 0.5
    prompt = state.role_prompt
    assert isinstance(prompt, str)
    assert f"{state.role_embedding[0]:.2f}" in prompt
