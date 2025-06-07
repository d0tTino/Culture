import sys
import types

import pytest
from pytest import MonkeyPatch


def _ensure_stubs() -> None:
    """Insert lightweight stubs for optional dependencies."""
    if "chromadb" not in sys.modules:
        chromadb = types.ModuleType("chromadb")
        chromadb.__version__ = "0.0"

        class _DummyClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

        chromadb.PersistentClient = _DummyClient
        sys.modules["chromadb"] = chromadb
        utils_mod = types.ModuleType("chromadb.utils.embedding_functions")
        utils_mod.SentenceTransformerEmbeddingFunction = object  # type: ignore[attr-defined]
        utils_pkg = types.ModuleType("chromadb.utils")
        utils_pkg.embedding_functions = utils_mod
        sys.modules["chromadb.utils"] = utils_pkg
        sys.modules["chromadb.utils.embedding_functions"] = utils_mod
    if "weaviate" not in sys.modules:
        weaviate = types.ModuleType("weaviate")
        weaviate.__path__ = []  # treat stub as a package
        weaviate.Client = object  # type: ignore[attr-defined]
        sys.modules["weaviate"] = weaviate
        classes_mod = types.ModuleType("weaviate.classes")
        sys.modules["weaviate.classes"] = classes_mod
        weaviate.classes = classes_mod  # type: ignore[attr-defined]
    if "sse_starlette.sse" not in sys.modules:
        sse_mod = types.ModuleType("sse_starlette.sse")
        sse_mod.EventSourceResponse = object  # type: ignore[attr-defined]
        sys.modules["sse_starlette.sse"] = sse_mod


@pytest.mark.unit
def test_update_mood_delegates(monkeypatch: MonkeyPatch) -> None:
    pytest.importorskip("weaviate.classes")
    pytest.importorskip("chromadb")
    _ensure_stubs()
    from src.agents.core.agent_controller import AgentController
    from src.agents.core.base_agent import Agent
    from src.agents.graphs import basic_agent_graph

    monkeypatch.setattr(basic_agent_graph, "compile_agent_graph", lambda: None)

    agent = Agent(agent_id="a3", vector_store_manager=object())
    called: dict[str, float] = {}

    def fake_update(self: AgentController, score: float) -> None:
        called["score"] = score
        self.state.mood_level = 0.42
        self.state.mood_history.append((0, 0.42))

    monkeypatch.setattr(AgentController, "update_mood", fake_update)

    agent.update_mood(0.7)

    assert called["score"] == 0.7
    assert agent.state.mood_level == 0.42
