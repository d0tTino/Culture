import pytest

pytest.importorskip("chromadb")

from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaVectorStoreManager
from tests.unit.memory.test_semantic_memory_manager import DummyDriver


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.memory
@pytest.mark.usefixtures("chroma_test_dir")
async def test_nightly_job_and_retrieval(chroma_test_dir: str) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=chroma_test_dir,
        embedding_function=lambda texts: [[0.0] for _ in texts],
    )
    driver = DummyDriver()
    manager = SemanticMemoryManager(vector, driver)

    vector.add_memory("agent", 1, "thought", "first")
    vector.add_memory("agent", 2, "thought", "second")

    await manager.run_nightly_job("agent")

    assert driver.store[0]["agent"] == "agent"
    recent = manager.get_recent_summaries("agent", limit=1)
    assert recent == ["first\nsecond"]
