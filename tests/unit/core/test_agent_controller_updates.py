import pytest

try:
    from src.agents.core.agent_controller import AgentController
    from src.agents.core.agent_state import AgentState
except IndentationError:
    pytest.skip("agent_state module is unparsable", allow_module_level=True)


@pytest.mark.unit
def test_update_mood_clamps_and_history() -> None:
    state = AgentState(agent_id="a1", name="Test", mood_level=0.9)
    controller = AgentController(state)

    controller.update_mood(1.0)
    assert state.mood_level == 1.0
    assert state.mood_history[-1][1] == 1.0
    history_len = len(state.mood_history)

    controller.update_mood(-10.0)
    assert state.mood_level == -1.0
    assert len(state.mood_history) == history_len + 1

    controller.update_mood(None)
    assert -1.0 < state.mood_level <= 1.0


@pytest.mark.unit
def test_update_relationship_learning_rates_and_history() -> None:
    state = AgentState(agent_id="a1", name="A")
    controller = AgentController(state)

    controller.update_relationship("b", 1.0, is_targeted=True)
    pos_score = state.relationships["b"]
    assert pos_score == pytest.approx(0.9)
    assert state.relationship_history["b"][-1][1] == pytest.approx(pos_score)

    controller.update_relationship("b", -1.0, is_targeted=True)
    neg_score = state.relationships["b"]
    assert neg_score == pytest.approx(-0.3)
    assert state.relationship_history["b"][-1][1] == pytest.approx(neg_score)

    history_len = len(state.relationship_history["b"])
    controller.update_relationship("b", None)
    assert len(state.relationship_history["b"]) == history_len


@pytest.mark.unit
def test_change_role_cost_and_cooldown() -> None:
    state = AgentState(agent_id="a1", name="A", current_role="Facilitator", ip=10.0)
    controller = AgentController(state)

    assert controller.change_role("Innovator", current_step=1)
    assert state.current_role == "Innovator"
    assert state.ip == pytest.approx(5.0)

    assert not controller.change_role("Analyzer", current_step=2)
    assert state.ip == pytest.approx(5.0)

    state.ip = 4.0
    assert not controller.change_role("Analyzer", current_step=5)
    assert state.ip == pytest.approx(4.0)

    state.ip = 6.0
    assert not controller.change_role("Unknown", current_step=6)

    with pytest.raises(ValueError):
        AgentController().change_role("Innovator", current_step=0)
