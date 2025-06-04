import pytest

pytest.importorskip("langgraph")
from langgraph.graph import StateGraph

from src.agents.graphs.agent_graph_builder import build_graph


def test_build_graph_nodes() -> None:
    graph = build_graph()
    assert isinstance(graph, StateGraph)
    assert "analyze_perception_sentiment" in graph.nodes
    assert "finalize_message_agent" in graph.nodes
