import sys
import time
import types

import pytest

from tests.utils.dummy_chromadb import setup_dummy_chromadb


@pytest.fixture(autouse=True)
def _dummy_chroma() -> None:
    setup_dummy_chromadb()

    # Provide a lightweight neo4j stub so optional imports succeed
    neo4j_stub = types.ModuleType("neo4j")
    neo4j_stub.GraphDatabase = object  # type: ignore[attr-defined]
    neo4j_stub.Driver = object  # type: ignore[attr-defined]
    sys.modules.setdefault("neo4j", neo4j_stub)
    sys.modules.setdefault("neo4j.exceptions", types.ModuleType("neo4j.exceptions"))


@pytest.mark.unit
def test_nightly_consolidation_groups_memories(tmp_path) -> None:
    from src.agents.memory.vector_store import ChromaVectorStoreManager

    manager = ChromaVectorStoreManager(
        persist_directory=str(tmp_path), embedding_function=lambda texts: [[0.0] for _ in texts]
    )

    events = [
        (1, "thought", "m1"),
        (2, "action", "m2"),
        (3, "thought", "m3"),
        (4, "action", "m4"),
    ]
    for step, etype, content in events:
        manager.add_memory("agent", step, etype, content)

    manager.consolidate_daily_memories("agent", 1, 4)

    summaries = manager.retrieve_filtered_memories(
        "agent", filters={"memory_type": "consolidated_summary"}, limit=None
    )
    assert len(summaries) == 2
    texts = [s["content"] for s in summaries]
    assert any("m1" in t and "m3" in t for t in texts)
    assert any("m2" in t and "m4" in t for t in texts)


@pytest.mark.unit
def test_consolidation_improves_retrieval_speed(tmp_path, monkeypatch) -> None:
    from src.agents.memory.vector_store import ChromaVectorStoreManager

    manager = ChromaVectorStoreManager(
        persist_directory=str(tmp_path), embedding_function=lambda texts: [[0.0] for _ in texts]
    )

    for i in range(20):
        manager.add_memory("agent", i + 1, "thought", f"mem {i}")

    collection = manager.collection
    original_query = collection.query

    def slow_query(*args, **kwargs):
        time.sleep(len(collection.docs) / 1000.0)
        return original_query(*args, **kwargs)

    monkeypatch.setattr(collection, "query", slow_query)

    manager.retrieve_relevant_memories("agent", "mem")
    before = manager.retrieval_times[-1]

    manager.consolidate_daily_memories("agent", 1, 20)

    manager.retrieve_relevant_memories("agent", "mem")
    after = manager.retrieval_times[-1]

    assert after < before
