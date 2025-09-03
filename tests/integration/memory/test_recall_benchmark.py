import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest
import pytest_asyncio

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
def recall_fixture() -> dict[str, list[str]]:
    """Load mapping of query to expected memory IDs."""
    path = Path(__file__).parents[2] / "data" / "recall_fixture.json"
    with path.open() as f:
        return cast(dict[str, list[str]], json.load(f))


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
    recall_fixture: dict[str, list[str]],
    chroma_test_dir: Path,
    event_loop: asyncio.AbstractEventLoop,
) -> list[tuple[BenchFunc, set[str]]]:
    """Prepare retrieval cases for benchmarking based on fixture data."""

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

    def _add_memory(mid: str, content: str, step: int) -> None:
        embedding = vector.get_embedding(content)
        metadata = {
            "agent_id": "agent",
            "step": step,
            "event_type": "thought",
            "memory_type": "raw",
            "timestamp": datetime.utcnow().isoformat(),
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

    event_loop.run_until_complete(service.run_semantic_job("agent"))

    cases: list[tuple[BenchFunc, set[str]]] = []

    def make_retrieval(q: str) -> BenchFunc:
        async def _retrieval() -> Sequence[Any]:
            return cast(
                Sequence[Any],
                await service.retrieve_relevant_memories("agent", q, k=5),
            )

        return _retrieval

    for query, ids in recall_fixture.items():
        retrieval = make_retrieval(query)
        for _ in range(30):
            cases.append((retrieval, set(ids)))

    return cases


@pytest_asyncio.fixture
async def benchmark_results(
    recall_benchmark: BenchFixture, benchmark_cases: list[tuple[BenchFunc, set[str]]]
) -> tuple[list[float], list[float]]:
    latencies: list[float] = []
    precisions: list[float] = []
    for retrieval, relevant_ids in benchmark_cases:
        stats = await recall_benchmark(retrieval, relevant_ids, 5)
        precisions.append(stats["p_at_k"])
        latencies.append(stats["latency"])
    return latencies, precisions


@pytest_asyncio.fixture
async def p_at_5(benchmark_results: tuple[list[float], list[float]]) -> float:
    _, precisions = benchmark_results
    return sum(precisions) / len(precisions)


@pytest_asyncio.fixture
async def latency_p95_ms(benchmark_results: tuple[list[float], list[float]]) -> float:
    latencies, _ = benchmark_results
    return p95(latencies) * 1000.0


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.memory
async def test_recall_benchmark(p_at_5: float, latency_p95_ms: float) -> None:
    assert p_at_5 >= 0.7
    assert latency_p95_ms <= 50
