from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("chromadb")

from src.agents.memory.memory_service import MemoryService
from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaVectorStoreManager
from tests.unit.memory.test_semantic_memory_manager import DummyDriver


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.memory
@pytest.mark.usefixtures("chroma_test_dir")
async def test_semantic_summary_updates_on_retrieval(chroma_test_dir: Path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=chroma_test_dir, embedding_function=lambda t: [[0.0] for _ in t]
    )
    driver = DummyDriver()
    manager = SemanticMemoryManager(vector, driver)
    service = MemoryService(vector_store=vector, semantic_manager=manager)

    vector.add_memory("agent", 1, "thought", "one", memory_type="raw")
    await service.get_context_pipeline("agent", semantic_limit=1)
    first = manager.get_recent_summaries("agent", limit=1)
    assert first and "one" in first[0]

    vector.add_memory("agent", 2, "thought", "two", memory_type="raw")
    await service.get_context_pipeline("agent", semantic_limit=1)
    second = manager.get_recent_summaries("agent", limit=1)
    assert second and "two" in second[0]
