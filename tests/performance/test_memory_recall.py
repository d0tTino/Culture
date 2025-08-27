import json
import time
from pathlib import Path

import pytest

from src.agents.memory.memory_service import MemoryService
from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.infra import metrics as infra_metrics
from src.utils import retrieval_metrics
from tests.unit.memory.test_semantic_memory_manager import DummyDriver

pytest.importorskip("chromadb")
pytest.importorskip("sklearn")


@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.memory
async def test_memory_recall(tmp_path) -> None:
    path = Path(__file__).parents[2] / "data" / "recall_fixture.json"
    with path.open() as f:
        recall_fixture = json.load(f)

    class _Embed:
        def __call__(self, input: list[str]) -> list[list[float]]:
            return [[1.0 if "cat" in x else 0.0] for x in input]

        def name(self) -> str:  # pragma: no cover - simple attribute
            return "dummy"

    vector = ChromaVectorStoreManager(persist_directory=tmp_path, embedding_function=_Embed())
    driver = DummyDriver()
    semantic = SemanticMemoryManager(vector, driver)
    service = MemoryService(vector, semantic)

    def _add_memory(mid: str, content: str, step: int) -> None:
        embedding = vector.get_embedding(content)
        metadata = {
            "agent_id": "agent",
            "step": step,
            "event_type": "thought",
            "memory_type": "raw",
            "timestamp": "",
            "retrieval_count": 0,
            "usage_count": 0,
            "last_retrieved_timestamp": "",
            "accumulated_relevance_score": 0.0,
            "retrieval_relevance_count": 0,
        }
        vector.collection.add(
            ids=[mid],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )

    step = 0
    for query, ids in recall_fixture.items():
        for i, mid in enumerate(ids):
            _add_memory(mid, f"{query} {i}", step)
            step += 1
    for i in range(5):
        _add_memory(f"dog_{i}", f"dog {i}", step + i)

    await service.run_semantic_job("agent")

    query = next(iter(recall_fixture.keys()))
    start = time.perf_counter()
    results = await service.retrieve_relevant_memories("agent", query, k=5)
    latency_ms = (time.perf_counter() - start) * 1000.0
    ids = [m.get("memory_id") or m.get("id") for m in results]
    p_at_5 = retrieval_metrics.precision_at_k(ids, set(recall_fixture[query]), 5)

    infra_metrics.record_retrieval_latency(latency_ms)
    infra_metrics.record_recall_p5(p_at_5)

    assert p_at_5 >= 0.7, f"p@5={p_at_5:.3f} below threshold"
    assert latency_ms <= 50, f"latency={latency_ms:.2f}ms above threshold"
