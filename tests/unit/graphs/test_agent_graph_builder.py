import pytest

pytest.importorskip("langgraph")

from langgraph.graph.graph import END, START, Graph

from src.agents.graphs.agent_graph_builder import build_graph


@pytest.mark.unit
def test_build_graph_structure() -> None:
    graph = build_graph()
    assert isinstance(graph, Graph)

    expected_nodes = {
        "analyze_perception_sentiment",
        "prepare_relationship_prompt",
        "retrieve_and_summarize_memories",
        "generate_thought_and_message",
        "handle_propose_idea",
        "handle_ask_clarification",
        "handle_continue_collaboration",
        "handle_idle",
        "handle_deep_analysis",
        "finalize_message_agent",
    }
    assert expected_nodes.issubset(graph.nodes.keys())

    expected_edges = {
        (START, "analyze_perception_sentiment"),
        ("analyze_perception_sentiment", "prepare_relationship_prompt"),
        ("prepare_relationship_prompt", "retrieve_and_summarize_memories"),
        ("retrieve_and_summarize_memories", "generate_thought_and_message"),
        ("generate_thought_and_message", "handle_idle"),
        ("handle_idle", "finalize_message_agent"),
        ("finalize_message_agent", END),
    }
    assert expected_edges.issubset(graph.edges)
