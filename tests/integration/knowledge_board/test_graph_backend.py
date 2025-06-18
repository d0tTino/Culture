from collections.abc import Iterable
from typing import Any

import pytest

from src.infra import config
from src.sim.graph_knowledge_board import GraphKnowledgeBoard
from src.sim.simulation import Simulation


class DummyResult(list):
    def single(self) -> Any:
        return self[0]

class DummySession:
    def __init__(self, store: list[dict[str, Any]]) -> None:
        self.store = store
    def run(self, query: str, **params: Any) -> Iterable[Any]:
        if query.startswith("CREATE"):
            self.store.append(params.get("props", params))
            return []
        if "count" in query:
            return DummyResult([{"cnt": len(self.store)}])
        if "DELETE" in query:
            self.store.clear()
            return []
        if "ORDER BY e.step DESC" in query:
            limit = params["limit"]
            entries = sorted(self.store, key=lambda x: x["step"], reverse=True)[:limit]
            return DummyResult([{"e": e} for e in entries])
        if "ORDER BY e.step ASC" in query:
            entries = sorted(self.store, key=lambda x: x["step"])
            return DummyResult([{"e": e} for e in entries])
        return []
    def __enter__(self) -> "DummySession":
        return self
    def __exit__(self, exc_type, exc, tb) -> None:
        pass

class DummyDriver:
    def __init__(self) -> None:
        self.store: list[dict[str, Any]] = []
    def session(self) -> DummySession:
        return DummySession(self.store)

@pytest.mark.integration
def test_simulation_uses_graph_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_BOARD_BACKEND", "graph")
    config.load_config()
    monkeypatch.setattr(
        "neo4j.GraphDatabase.driver", lambda *a, **k: DummyDriver(), raising=False
    )
    sim = Simulation(agents=[])
    assert isinstance(sim.knowledge_board, GraphKnowledgeBoard)

@pytest.mark.integration
def test_graph_board_add_and_retrieve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_BOARD_BACKEND", "graph")
    config.load_config()
    dummy_driver = DummyDriver()
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *a, **k: dummy_driver)
    board = GraphKnowledgeBoard()
    board.clear_board()
    assert board.get_state() == []
    board.add_entry("idea", "agent", 1)
    assert board.get_state() == ["Step 1 (Agent: agent): idea"]

