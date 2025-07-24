import pytest

from src.agents.memory.level3_summary_manager import Level3SummaryManager
from src.agents.memory.memory_service import MemoryService
from src.agents.memory.vector_store import ChromaVectorStoreManager
from tests.utils.dummy_chromadb import setup_dummy_chromadb


@pytest.fixture(autouse=True)
def _dummy_chroma() -> None:
    setup_dummy_chromadb()


@pytest.mark.unit
def test_l3_consolidation_and_retrieval(tmp_path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=str(tmp_path), embedding_function=lambda texts: [[0.0] for _ in texts]
    )
    l3_manager = Level3SummaryManager(vector)
    service = MemoryService(vector_store=vector, semantic_manager=None, level3_manager=l3_manager)

    vector.add_memory("agent", 1, "thought", "s1", memory_type="chapter_summary")
    vector.add_memory("agent", 2, "thought", "s2", memory_type="chapter_summary")

    summary = l3_manager.consolidate_summaries("agent", 1, 2)
    assert "s1" in summary and "s2" in summary

    retrieved = service.get_long_term_summaries("agent", limit=1)
    assert retrieved == [summary]
