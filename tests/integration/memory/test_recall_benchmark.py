from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("chromadb")

import asyncio

from src.agents.memory.memory_service import MemoryService
from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.utils.retrieval_metrics import p95
from tests.unit.memory.test_semantic_memory_manager import DummyDriver


@pytest.fixture
def benchmark_cases(chroma_test_dir: Path, event_loop: asyncio.AbstractEventLoop):
    """Prepare 30 retrieval cases for benchmarking."""
    class _Embed:
        def __call__(self, input: list[str]) -> list[list[float]]:
            return [[1.0 if "cat" in x else 0.0] for x in input]

        def name(self) -> str:  # pragma: no cover - simple attribute
            return "dummy"

    vector = ChromaVectorStoreManager(
        persist_directory=chroma_test_dir,
        embedding_function=_Embed(),
    )
    driver = DummyDriver()
    semantic = SemanticMemoryManager(vector, driver)
    service = MemoryService(vector, semantic)

    relevant_ids = set()
    for i in range(5):
        mid = vector.add_memory("agent", i, "thought", f"cat {i}", memory_type="raw")
        relevant_ids.add(mid)
    for i in range(5):
        vector.add_memory("agent", i + 5, "thought", f"dog {i}", memory_type="raw")

    event_loop.run_until_complete(service.run_semantic_job("agent"))

    async def retrieval():
        return await service.retrieve_relevant_memories("agent", "cat", k=5)

    return [(retrieval, relevant_ids) for _ in range(30)]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.memory
async def test_recall_benchmark(recall_benchmark, benchmark_cases) -> None:
    latencies = []
    precisions = []
    for retrieval, relevant_ids in benchmark_cases:
        metrics = await recall_benchmark(retrieval, relevant_ids, 5)
        precisions.append(metrics["p_at_k"])
        latencies.append(metrics["latency"])
    avg_p5 = sum(precisions) / len(precisions)
    latency_p95 = p95(latencies) * 1000.0  # convert to milliseconds
    assert avg_p5 >= 0.7
    assert latency_p95 <= 50
