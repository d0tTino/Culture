import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("chromadb")

from src.agents.memory.memory_service import MemoryService
from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.infra import metrics as infra_metrics
from src.interfaces import metrics
from src.utils import retrieval_metrics
from src.utils.retrieval_metrics import p95
from tests.unit.memory.test_semantic_memory_manager import DummyDriver

BenchFunc = Callable[[], Awaitable[Sequence[Any]]]
BenchMetrics = dict[str, float]
BenchFixture = Callable[[BenchFunc, Iterable[str], int], Awaitable[BenchMetrics]]


@pytest.fixture
def recall_benchmark() -> BenchFixture:
    """Return helper to record precision@k and latency for a retrieval coroutine."""

    logger = logging.getLogger(__name__)

    async def _bench(
        retrieval_coro: BenchFunc, relevant_ids: Iterable[str], k: int
    ) -> BenchMetrics:
        results, latency = await retrieval_metrics.time_call(retrieval_coro)
        ids = [m.get("memory_id") or m.get("id") for m in results]
        p_at_k = retrieval_metrics.precision_at_k(ids, set(relevant_ids), k)
        latency_ms = latency * 1000.0
        logger.info("p@%d=%.3f latency=%.2fms", k, p_at_k, latency_ms)
        metrics.P_AT_K.set(p_at_k)
        metrics.RETRIEVAL_LATENCY_MS.set(latency_ms)
        infra_metrics.record_recall_p5(p_at_k)
        infra_metrics.record_retrieval_latency(latency_ms)
        return {"p_at_k": p_at_k, "latency": latency}

    return _bench


@pytest.fixture
def benchmark_cases(
    chroma_test_dir: Path, event_loop: asyncio.AbstractEventLoop
) -> list[tuple[BenchFunc, set[str]]]:
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

    async def retrieval() -> Sequence[Any]:
        return cast(
            Sequence[Any],
            await service.retrieve_relevant_memories("agent", "cat", k=5),
        )

    return [(retrieval, relevant_ids) for _ in range(30)]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.memory
async def test_recall_benchmark(
    recall_benchmark: BenchFixture, benchmark_cases: list[tuple[BenchFunc, set[str]]]
) -> None:
    latencies: list[float] = []
    precisions: list[float] = []
    for retrieval, relevant_ids in benchmark_cases:
        stats = await recall_benchmark(retrieval, relevant_ids, 5)
        precisions.append(stats["p_at_k"])
        latencies.append(stats["latency"])
    avg_p5 = sum(precisions) / len(precisions)
    latency_p95 = p95(latencies) * 1000.0  # convert to milliseconds
    assert avg_p5 >= 0.7
    assert latency_p95 <= 50
