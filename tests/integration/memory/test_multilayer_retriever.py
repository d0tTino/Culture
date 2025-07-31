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
async def test_multilayer_retrieval_combines_layers(chroma_test_dir: Path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=chroma_test_dir, embedding_function=lambda t: [[0.0] for _ in t]
    )
    driver = DummyDriver()
    semantic = SemanticMemoryManager(vector, driver)
    service = MemoryService(vector, semantic)

    vector.add_memory("agent", 1, "thought", "cat", memory_type="raw")
    vector.add_memory("agent", 2, "thought", "dog", memory_type="raw")

    await service.retrieve_episodic_and_update_semantic("agent", k=2)

    vector.add_memory("agent", 3, "thought", "bird", memory_type="raw")
    semantic.group_memories_by_topic("agent", num_topics=2)

    episodic_only = await vector.aretrieve_relevant_memories("agent", "cat", k=5)
    episodic, semantic_res = await service.get_context_pipeline(
        "agent", query="cat", k=5, semantic_limit=5
    )
    combined_len = len(episodic) + len(semantic_res)
    assert combined_len >= len(episodic_only)


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
    service = MemoryService(vector, semantic)

    vector.add_memory("agent", 1, "thought", "cat", memory_type="raw")
    vector.add_memory("agent", 2, "thought", "dog", memory_type="raw")

    await service.run_semantic_job("agent")

    blended = service.blend_with_recent_semantic("agent", "bird", limit=1)
    assert "bird" in blended
    assert "cat" in blended or "dog" in blended


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.memory
@pytest.mark.usefixtures("chroma_test_dir")
async def test_hit_rate_with_semantic_summary(chroma_test_dir: Path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=chroma_test_dir,
        embedding_function=lambda t: [
            [1.0 if "cat" in x else 0.0, 1.0 if "dog" in x else 0.0] for x in t
        ],
    )
    driver = DummyDriver()
    semantic = SemanticMemoryManager(vector, driver)
    service = MemoryService(vector, semantic)

    for i in range(8):
        vector.add_memory("agent", i, "thought", f"cat memory {i}", memory_type="raw")
    for i in range(2):
        vector.add_memory("agent", 8 + i, "thought", f"dog memory {i}", memory_type="raw")

    await service.run_semantic_job("agent")
    semantic.group_memories_by_topic("agent", num_topics=2)

    episodic, summaries = await service.get_context_pipeline(
        "agent", query="cat", k=5, semantic_limit=1
    )

    total = len(episodic) + len(summaries)
    relevant = sum("cat" in m.get("content", "") for m in episodic)
    relevant += sum("cat" in s for s in summaries)
    hit_rate = relevant / total if total else 0.0

    assert hit_rate > 0.7
