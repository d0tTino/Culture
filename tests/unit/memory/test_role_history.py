import pytest

from tests.utils.dummy_chromadb import setup_dummy_chromadb


@pytest.fixture(autouse=True)
def _install_dummy_chromadb() -> None:
    setup_dummy_chromadb()


@pytest.mark.unit
def test_get_role_history_builds_periods(tmp_path) -> None:
    from src.agents.memory.vector_store import ChromaVectorStoreManager

    manager = ChromaVectorStoreManager(persist_directory=str(tmp_path), embedding_function=lambda x: [[0.0]])

    manager.record_role_change("agent", 1, "R0", "R1")
    manager.record_role_change("agent", 5, "R1", "R2")
    manager.record_role_change("agent", 10, "R2", "R3")

    history = manager.get_role_history("agent")
    assert history == [
        {"role": "R1", "start_step": 1, "end_step": 4},
        {"role": "R2", "start_step": 5, "end_step": 9},
        {"role": "R3", "start_step": 10, "end_step": None},
    ]


@pytest.mark.unit
def test_retrieve_role_specific_memories_without_query(monkeypatch, tmp_path) -> None:
    from src.agents.memory.vector_store import ChromaVectorStoreManager

    manager = ChromaVectorStoreManager(persist_directory=str(tmp_path), embedding_function=lambda x: [[0.0]])

    monkeypatch.setattr(
        manager,
        "get_role_history",
        lambda agent_id: [
            {"role": "Analyst", "start_step": 0, "end_step": 2},
            {"role": "Analyst", "start_step": 5, "end_step": 6},
        ],
    )

    calls = []

    def fake_retrieve(agent_id, filters=None, limit=None, include_usage_stats=False):
        calls.append(filters)
        return [{"id": len(calls)}]

    monkeypatch.setattr(manager, "retrieve_filtered_memories", fake_retrieve)

    memories = manager.retrieve_role_specific_memories("agent", role="Analyst", query=None, k=3)

    assert len(memories) == 2
    assert calls[0]["step"] == {"$gte": 0, "$lte": 2}
    assert calls[1]["step"] == {"$gte": 5, "$lte": 6}

