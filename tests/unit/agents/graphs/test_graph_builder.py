import pytest

pytest.importorskip("langgraph")
from src.agents.graphs.agent_graph_builder import build_graph


def test_build_graph_nodes() -> None:
    graph = build_graph()
    assert hasattr(graph, "ainvoke")
