import pytest

from src.agents.core.agent_attributes import AgentAttributes
from src.agents.graphs.interaction_handlers import (
    handle_propose_idea,
    handle_propose_idea_node,
    handle_continue_collaboration_node,
    handle_idle_node,
    handle_ask_clarification_node,
    handle_deep_analysis_node,
    handle_retrieve_and_update,
)
from src.agents.graphs.interaction_handlers import _UNLOCKED_CAPABILITY
from src.infra.config import (
    DU_AWARD_FOR_PROPOSAL,
    DU_COST_DEEP_ANALYSIS,
    IP_AWARD_FOR_PROPOSAL,
    IP_COST_TO_POST_IDEA,
)
from src.shared.memory_store import ChromaMemoryStore


@pytest.mark.unit
def test_handle_propose_idea_node() -> None:
    state_obj = AgentAttributes(
        id="a",
        mood=0.0,
        goals=[],
        resources={},
        relationships={},
    )
    state_obj.ip = 10
    turn_state = {"state": state_obj}
    out = handle_propose_idea_node(turn_state)
    expected = 10 - IP_COST_TO_POST_IDEA + IP_AWARD_FOR_PROPOSAL
    assert out["state"].ip == expected


@pytest.mark.unit
def test_handle_deep_analysis_node() -> None:
    state_obj = AgentAttributes(
        id="a",
        mood=0.0,
        goals=[],
        resources={},
        relationships={},
    )
    state_obj.du = 5
    turn_state = {"state": state_obj}
    out = handle_deep_analysis_node(turn_state)
    assert out["state"].du == 5 - DU_COST_DEEP_ANALYSIS


@pytest.mark.unit
def test_handle_propose_idea() -> None:
    agent = AgentAttributes(
        id="a",
        mood=0.0,
        goals=[],
        resources={},
        relationships={},
    )
    board = ChromaMemoryStore()
    memory = ChromaMemoryStore()
    agent.ip = 8
    agent.du = 0
    handle_propose_idea(agent, memory, board)
    docs = board.query("", top_k=1)
    assert docs[0]["metadata"]["author"] == "a"
    assert agent.ip == 8 - IP_COST_TO_POST_IDEA
    assert agent.du == DU_AWARD_FOR_PROPOSAL


@pytest.mark.unit
def test_handle_retrieve_and_update() -> None:
    agent = AgentAttributes(
        id="b",
        mood=0.0,
        goals=[],
        resources={},
        relationships={"a": 0.4},
    )
    store = ChromaMemoryStore()
    store.add_documents(["idea"], [{"author": "a"}])
    handle_retrieve_and_update(agent, store)
    assert agent.relationships["a"] >= 0.5
    assert _UNLOCKED_CAPABILITY in agent.unlocked_capabilities
    assert agent.relationship_momentum["a"] > 0


@pytest.mark.unit
def test_simple_passthrough_handlers() -> None:
    agent = AgentAttributes(
        id="x",
        mood=0.0,
        goals=[],
        resources={},
        relationships={},
    )
    ts = {"state": agent}
    assert handle_continue_collaboration_node(ts)["state"] is agent
    assert handle_idle_node(ts)["state"] is agent
    assert handle_ask_clarification_node(ts)["state"] is agent


@pytest.mark.unit
def test_handle_retrieve_and_update_edge_cases() -> None:
    agent = AgentAttributes(
        id="c",
        mood=0.0,
        goals=[],
        resources={},
        relationships={},
    )
    store = ChromaMemoryStore()
    # No documents -> early return
    handle_retrieve_and_update(agent, store)
    assert agent.relationships == {}
    store.add_documents(["idea"], [{"metadata_only": True}])
    handle_retrieve_and_update(agent, store)
    assert agent.relationships == {}
