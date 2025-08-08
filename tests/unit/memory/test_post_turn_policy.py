import pytest

pytest.importorskip("sklearn")
pytest.importorskip("chromadb")

from pathlib import Path  # noqa: E402

from src.agents.memory.memory_service import MemoryService  # noqa: E402
from src.agents.memory.vector_store import ChromaVectorStoreManager  # noqa: E402


@pytest.mark.unit
@pytest.mark.memory
@pytest.mark.usefixtures("chroma_test_dir")
def test_post_turn_write_policy(chroma_test_dir: Path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=chroma_test_dir,
        embedding_function=lambda t: [[0.0] for _ in t],
    )
    service = MemoryService(vector)

    mem_id = service.store_post_turn_memory("agent", 1, "thought", "hello", write=True)
    assert mem_id

    no_mem = service.store_post_turn_memory("agent", 2, "thought", "skip", write=False)
    assert no_mem == ""
