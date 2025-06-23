import pytest

from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaVectorStoreManager
from tests.utils.dummy_chromadb import setup_dummy_chromadb


class DummySession:
    def __init__(self, store: list[dict[str, str]]) -> None:
        self.store = store

    def run(self, query: str, **params: object):
        if query.strip().startswith("MERGE"):
            self.store.append(
                {
                    "agent": params["agent_id"],
                    "summary": params["summary"],
                    "created_at": params["now"],
                }
            )
            return []
        if query.strip().startswith("MATCH"):
            limit = params.get("limit", 3)
            items = list(reversed(self.store))[: int(limit)]
            return [{"summary": item["summary"]} for item in items]
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyDriver:
    def __init__(self) -> None:
        self.store: list[dict[str, str]] = []

    def session(self) -> DummySession:
        return DummySession(self.store)


@pytest.fixture(autouse=True)
def _dummy_chroma() -> None:
    setup_dummy_chromadb()


@pytest.mark.unit
def test_consolidation_and_retrieval(tmp_path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=str(tmp_path),
        embedding_function=lambda texts: [[0.0] for _ in texts],
    )
    driver = DummyDriver()
    manager = SemanticMemoryManager(vector, driver)

    vector.add_memory("agent", 1, "thought", "first")
    vector.add_memory("agent", 2, "thought", "second")

    summary = manager.consolidate_memories("agent")
    assert "first" in summary and "second" in summary
    assert driver.store[0]["agent"] == "agent"

    recent = manager.get_recent_summaries("agent", limit=1)
    assert recent == [summary]
