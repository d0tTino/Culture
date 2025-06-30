# ruff: noqa: E402
import pytest

pytest.importorskip("langgraph")

from src.agents.core.agent_state import AgentState
from src.agents.graphs.basic_agent_graph import AgentTurnState
from src.agents.graphs.interaction_handlers import (
    handle_deep_analysis_node,
    handle_propose_idea_node,
)
from src.infra.config import (
    DU_COST_DEEP_ANALYSIS,
    IP_AWARD_FOR_PROPOSAL,
    IP_COST_TO_POST_IDEA,
)


def test_handlers_modify_state() -> None:
    initial_ip = 10.0
    initial_du = 5.0
    state = AgentState(agent_id="a1", name="A1", ip=initial_ip, du=initial_du)
    turn_state: AgentTurnState = {"agent_state": state, "structured_output": None}

    # Test handle_propose_idea_node
    out = handle_propose_idea_node(turn_state)
    expected_ip = initial_ip - IP_COST_TO_POST_IDEA + IP_AWARD_FOR_PROPOSAL
    assert out["agent_state"].ip == expected_ip

    # Test handle_deep_analysis_node
    out2 = handle_deep_analysis_node(turn_state)
    expected_du = initial_du - DU_COST_DEEP_ANALYSIS
    assert out2["agent_state"].du == expected_du
