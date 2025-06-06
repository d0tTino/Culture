import pytest

from src.agents.core.agent_controller import AgentController
from src.agents.core.agent_state import AgentState


@pytest.mark.unit
def test_agent_state_serialization_roundtrip() -> None:
    state = AgentState(agent_id="agent1", name="TestAgent")
    controller = AgentController(state)
    controller.update_mood(0.4)
    controller.update_relationship("agent2", sentiment_score=0.2)

    serialized = state.to_dict()
    restored = AgentState.from_dict(serialized)

    assert serialized == restored.to_dict()
    assert restored.agent_id == "agent1"
    assert restored.name == "TestAgent"
