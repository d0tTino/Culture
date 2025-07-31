import asyncio

import pytest

pytest.importorskip("sklearn")

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

    asyncio.run(manager.run_nightly_job("agent"))
    assert driver.store[0]["agent"] == "agent"

    recent = manager.get_recent_summaries("agent", limit=1)
    assert recent == [summary]


@pytest.mark.unit
def test_group_memories_by_topic_zero_embeddings(tmp_path) -> None:
    """Memories should still be grouped when embeddings are all zeros."""
    vector = ChromaVectorStoreManager(
        persist_directory=str(tmp_path), embedding_function=lambda texts: [[0.0] for _ in texts]
    )
    manager = SemanticMemoryManager(vector, driver=None)

    for i, text in enumerate(["cat", "dog", "bird"]):
        vector.add_memory("agent", i, "thought", text)

    groups = manager.group_memories_by_topic("agent", num_topics=2)

    assert groups
    assert sum(len(g) for g in groups.values()) == 3
    assert len(groups) <= 2
    # When embeddings provide no signal we fall back to TF-IDF clustering
    assert manager.topic_centroids["agent"].shape[0] <= 2


@pytest.mark.unit
def test_retrieve_context_with_scores(tmp_path) -> None:
    """Results should include relevance scores sorted in descending order."""

    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0 if "cat" in t else 0.0, 1.0 if "dog" in t else 0.0] for t in texts]

    vector = ChromaVectorStoreManager(persist_directory=str(tmp_path), embedding_function=embed)
    manager = SemanticMemoryManager(vector, driver=None)

    vector.add_memory("agent", 1, "thought", "cat memory")
    vector.add_memory("agent", 2, "thought", "dog memory")
    vector.add_memory("agent", 3, "thought", "cat and dog")

    manager.group_memories_by_topic("agent", num_topics=2)
    results = manager.retrieve_context_with_scores("agent", "cat", k=3)

    assert results
    # "dog memory" may not be returned if it belongs to a different topic
    assert [r["content"] for r in results][:2] == ["cat memory", "cat and dog"]
    scores = [r["relevance_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
