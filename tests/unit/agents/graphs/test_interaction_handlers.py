# ruff: noqa: E402
import pytest

pytest.importorskip("langgraph")

from src.agents.core.agent_state import AgentState
from src.agents.graphs.basic_agent_graph import AgentTurnState
from src.agents.graphs.interaction_handlers import (
    handle_deep_analysis_node,
    handle_propose_idea_node,
)


def test_handlers_modify_state() -> None:
    state = AgentState(agent_id="a1", name="A1")
    state.ip = 10
    state.du = 5
    turn_state: AgentTurnState = {"state": state}
    out = handle_propose_idea_node(turn_state)
    assert out["state"].ip < 10
    out2 = handle_deep_analysis_node(turn_state)
    assert out2["state"].du < 5
