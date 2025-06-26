import pytest

pytest.importorskip("chromadb")

from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.infra import config
from tests.utils.mock_llm import MockLLM


@pytest.mark.integration
@pytest.mark.memory
@pytest.mark.usefixtures("chroma_test_dir")
def test_frequency_pruning(monkeypatch: pytest.MonkeyPatch, chroma_test_dir: str) -> None:
    mock_llm_cm = MockLLM({"default": "resp"})
    mock_llm_cm.__enter__()
    try:
        store = ChromaVectorStoreManager(persist_directory=chroma_test_dir)
        agent = "agent_prune"
        low_id = store.add_memory(
            agent_id=agent,
            step=1,
            event_type="summary",
            content="low",
            memory_type="consolidated_summary",
            metadata={
                "retrieval_count": 1,
                "usage_count": 1,
                "retrieval_relevance_count": 1,
                "accumulated_relevance_score": 0.1,
                "last_retrieved_timestamp": "2024-01-01T00:00:00",
            },
        )
        high_id = store.add_memory(
            agent_id=agent,
            step=2,
            event_type="summary",
            content="high",
            memory_type="consolidated_summary",
            metadata={
                "retrieval_count": 10,
                "usage_count": 10,
                "retrieval_relevance_count": 10,
                "accumulated_relevance_score": 5.0,
                "last_retrieved_timestamp": "2024-01-01T00:00:00",
            },
        )
        monkeypatch.setitem(config.CONFIG_OVERRIDES, "MEMORY_PRUNING_USAGE_COUNT_THRESHOLD", 5)
        monkeypatch.setitem(config.CONFIG_OVERRIDES, "MEMORY_PRUNING_L1_MUS_THRESHOLD", 1.0)
        monkeypatch.setattr(store, "_calculate_mus", lambda m: 0.1)
        ids = store.get_l1_memories_for_mus_pruning(1.0, 0)
        assert low_id in ids
        assert high_id not in ids
    finally:
        mock_llm_cm.__exit__(None, None, None)
