import pytest

# ruff: noqa: E402

# - allow runtime dependency checks before imports

pytest.importorskip("langgraph")

from langgraph.graph.graph import END, START

from src.agents.graphs.agent_graph_builder import build_graph


@pytest.mark.unit
def test_build_graph_structure() -> None:
    graph = build_graph()
    base = graph.builder if hasattr(graph, "builder") else graph
    assert hasattr(base, "nodes")

    expected_nodes = {
        "analyze_perception_sentiment",
        "prepare_relationship_prompt",
        "retrieve_and_summarize_memories",
        "generate_thought_and_message",
        "route_action_intent",
        "handle_propose_idea",
        "handle_ask_clarification",
        "handle_continue_collaboration",
        "handle_idle",
        "handle_deep_analysis",
        "finalize_message_agent",
    }
    assert expected_nodes.issubset(base.nodes.keys())

    expected_edges = {
        (START, "analyze_perception_sentiment"),
        ("analyze_perception_sentiment", "prepare_relationship_prompt"),
        ("prepare_relationship_prompt", "retrieve_and_summarize_memories"),
        ("retrieve_and_summarize_memories", "retrieve_semantic_context"),
        ("retrieve_semantic_context", "generate_thought_and_message"),
        ("generate_thought_and_message", "route_action_intent"),
        ("handle_idle", "finalize_message_agent"),
        ("finalize_message_agent", END),
    }
    edges = set(getattr(base, "edges", set()))
    branches = getattr(base, "branches", {})
    for start, branch_dict in branches.items():
        for branch in branch_dict.values():
            if hasattr(branch, "ends"):
                for dest in branch.ends.values():
                    edges.add((start, dest))
    assert expected_edges.issubset(edges)
