from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from src.agents.memory.multi_layer_retriever import MultiLayerRetriever
from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaVectorStoreManager
from tests.unit.memory.test_semantic_memory_manager import DummyDriver


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.memory
@pytest.mark.usefixtures("chroma_test_dir")
async def test_multilayer_retrieval_combines_layers(chroma_test_dir: Path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=chroma_test_dir, embedding_function=lambda t: [[0.0] for _ in t]
    )
    driver = DummyDriver()
    semantic = SemanticMemoryManager(vector, driver)
    retriever = MultiLayerRetriever(vector, semantic)

    vector.add_memory("agent", 1, "thought", "cat", memory_type="raw")
    vector.add_memory("agent", 2, "thought", "dog", memory_type="raw")

    await retriever.retrieve_and_update_semantic("agent", k=2)

    vector.add_memory("agent", 3, "thought", "bird", memory_type="raw")
    semantic.group_memories_by_topic("agent", num_topics=2)

    episodic_only = await vector.aretrieve_relevant_memories("agent", "cat", k=5)
    combined = await retriever.retrieve("agent", "cat", k=5)
    assert len(combined) >= len(episodic_only)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.memory
@pytest.mark.usefixtures("chroma_test_dir")
async def test_blend_with_recent_semantic(chroma_test_dir: Path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=chroma_test_dir, embedding_function=lambda t: [[0.0] for _ in t]
    )
    driver = DummyDriver()
    semantic = SemanticMemoryManager(vector, driver)
    retriever = MultiLayerRetriever(vector, semantic)

    vector.add_memory("agent", 1, "thought", "cat", memory_type="raw")
    vector.add_memory("agent", 2, "thought", "dog", memory_type="raw")

    await retriever.run_semantic_job("agent")

    blended = retriever.blend_with_recent_semantic("agent", "bird", limit=1)
    assert "bird" in blended
    assert "cat" in blended or "dog" in blended
