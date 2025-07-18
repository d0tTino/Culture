from datetime import datetime, timedelta

import pytest

from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.infra import config

pytest.importorskip("chromadb")


@pytest.mark.unit
@pytest.mark.memory
@pytest.mark.usefixtures("chroma_test_dir")
def test_prune_memories_hybrid(monkeypatch: pytest.MonkeyPatch, chroma_test_dir):
    store = ChromaVectorStoreManager(
        persist_directory=chroma_test_dir,
        embedding_function=lambda texts: [[0.0] for _ in texts],
    )

    now = datetime.utcnow()
    older = (now - timedelta(days=10)).isoformat()
    very_old = (now - timedelta(days=60)).isoformat()

    l1_low = store.add_memory(
        "agent",
        1,
        "summary",
        "low l1",
        memory_type="consolidated_summary",
        metadata={"simulation_step_timestamp": older, "mus_value": 0.1},
    )
    l1_high = store.add_memory(
        "agent",
        2,
        "summary",
        "high l1",
        memory_type="consolidated_summary",
        metadata={"simulation_step_timestamp": older, "mus_value": 0.8},
    )
    l2_low = store.add_memory(
        "agent",
        3,
        "summary",
        "low l2",
        memory_type="chapter_summary",
        metadata={"simulation_step_end_timestamp": older, "mus_value": 0.2},
    )
    l2_new_high = store.add_memory(
        "agent",
        4,
        "summary",
        "new high",
        memory_type="chapter_summary",
        metadata={"simulation_step_end_timestamp": now.isoformat(), "mus_value": 0.6},
    )
    l2_very_old = store.add_memory(
        "agent",
        5,
        "summary",
        "very old",
        memory_type="chapter_summary",
        metadata={"simulation_step_end_timestamp": very_old, "mus_value": 0.5},
    )

    monkeypatch.setattr(store, "_calculate_mus", lambda m: float(m.get("mus_value", 0.0)))
    monkeypatch.setitem(config.CONFIG_OVERRIDES, "MEMORY_PRUNING_USAGE_COUNT_THRESHOLD", 5)

    pruned = store.prune_memories_hybrid(
        l1_mus_threshold=0.5,
        l2_mus_threshold=0.3,
        l2_age_days=30,
        l1_min_age_days=0,
        l2_min_age_days=0,
    )

    remaining = [d["id"] for d in store.collection.docs]

    assert pruned == 3
    assert l1_low not in remaining
    assert l2_low not in remaining
    assert l2_very_old not in remaining
    assert l1_high in remaining
    assert l2_new_high in remaining
