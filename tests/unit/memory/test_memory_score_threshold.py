from pathlib import Path

import pytest

from src.agents.memory.memory_service import MemoryService
from src.agents.memory.vector_store import ChromaVectorStoreManager

pytest.importorskip("sklearn")
pytest.importorskip("chromadb")


@pytest.mark.unit
@pytest.mark.memory
@pytest.mark.usefixtures("chroma_test_dir")
def test_memory_score_threshold(chroma_test_dir: Path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=chroma_test_dir,
        embedding_function=lambda t: [[0.0] for _ in t],
    )

    def scorer(content: str, metadata: dict | None = None) -> int:
        return len(content)

    service = MemoryService(
        vector_store=vector,
        memory_score_threshold=5,
        memory_scorer=scorer,
    )

    low = service.add_memory("agent", 1, "thought", "hi")
    assert low == ""
    assert vector.collection.count() == 0

    high = service.add_memory("agent", 2, "thought", "hello world")
    assert high
    assert vector.collection.count() == 1
