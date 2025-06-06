# ruff: noqa: E402
import pytest

pytest.importorskip("langgraph")

from src.agents.core.agent_state import AgentState
from src.agents.graphs.basic_agent_graph import AgentTurnState
from src.agents.graphs.graph_nodes import (
    analyze_perception_sentiment_node,
    finalize_message_agent_node,
    generate_thought_and_message_node,
    prepare_relationship_prompt_node,
    retrieve_and_summarize_memories_node,
)


@pytest.mark.asyncio
async def test_nodes_flow() -> None:
    state = AgentState(agent_id="a1", name="A1")
    turn_state: AgentTurnState = {
        "agent_id": "a1",
        "simulation_step": 1,
        "current_state": {},
        "perceived_messages": [{"sender_id": "b", "content": "hi"}],
        "state": state,
        "environment_perception": {},
    }
    out1 = analyze_perception_sentiment_node(turn_state)
    assert "turn_sentiment_score" in out1
    out2 = prepare_relationship_prompt_node(turn_state)
    assert "prompt_modifier" in out2
    turn_state.update(out1)
    turn_state.update(out2)
    turn_state["vector_store_manager"] = None
    turn_state["agent_instance"] = None
    out3 = await retrieve_and_summarize_memories_node(turn_state)
    assert "rag_summary" in out3
    turn_state.update(out3)
    out4 = await generate_thought_and_message_node(turn_state)
    assert "structured_output" in out4
    turn_state.update(out4)
    out5 = await finalize_message_agent_node(turn_state)
    assert "message_content" in out5
