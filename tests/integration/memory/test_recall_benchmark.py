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
async def test_recall_benchmark(recall_benchmark, chroma_test_dir: Path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=chroma_test_dir,
        embedding_function=lambda t: [[1.0 if "cat" in x else 0.0] for x in t],
    )
    driver = DummyDriver()
    semantic = SemanticMemoryManager(vector, driver)
    service = MemoryService(vector, semantic)

    cat_id = vector.add_memory("agent", 1, "thought", "cat", memory_type="raw")
    await service.run_semantic_job("agent")

    async def retrieval():
        return await service.retrieve_relevant_memories("agent", "cat", k=1)

    metrics = await recall_benchmark(retrieval, {cat_id}, 1)
    assert 0.0 <= metrics["p_at_k"] <= 1.0
    assert metrics["latency"] >= 0.0
