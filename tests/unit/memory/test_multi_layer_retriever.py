from pathlib import Path
from typing import Any

import pytest

from src.agents.memory.memory_service import MemoryService
from src.agents.memory.multi_layer_retriever import MultiLayerRetriever
from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.interfaces import metrics
from tests.utils.dummy_chromadb import setup_dummy_chromadb

pytest.importorskip("sklearn")

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _dummy_chroma() -> None:
    setup_dummy_chromadb()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_merges_and_sorts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=str(tmp_path), embedding_function=lambda t: [[0.0] for _ in t]
    )
    semantic = SemanticMemoryManager(vector, driver=None)
    retriever = MultiLayerRetriever(vector, semantic)

    async def fake_episodic(agent_id: str, query: str, k: int) -> list[dict[str, Any]]:
        return [
            {"content": "e1", "relevance_score": 0.5},
            {"content": "e2", "relevance_score": 0.2},
        ]

    def fake_semantic(agent_id: str, query: str, k: int) -> list[dict[str, Any]]:
        return [
            {"content": "s1", "relevance_score": 0.9},
            {"content": "s2", "relevance_score": 0.6},
        ]

    monkeypatch.setattr(vector, "aretrieve_relevant_memories", fake_episodic)
    monkeypatch.setattr(semantic, "retrieve_context_with_scores", fake_semantic)

    results = await retriever.retrieve("agent", "q", k=4)

    assert [r["content"] for r in results] == ["s1", "s2", "e1", "e2"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_and_update_calls_semantic_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=str(tmp_path), embedding_function=lambda t: [[0.0] for _ in t]
    )
    semantic = SemanticMemoryManager(vector, driver=None)
    retriever = MultiLayerRetriever(vector, semantic)

    episodic = [{"content": "e"}]

    async def fake_episodic(agent_id: str, query: str, k: int) -> list[dict[str, str]]:
        return episodic

    calls: list[tuple[str, list[dict[str, str]]]] = []

    async def fake_run_job(agent_id: str, memories: list[dict[str, str]] | None = None) -> None:
        calls.append((agent_id, memories or []))

    monkeypatch.setattr(vector, "aretrieve_relevant_memories", fake_episodic)
    monkeypatch.setattr(semantic, "run_nightly_job", fake_run_job)

    results = await retriever.retrieve_and_update_semantic("agent", "q", k=1)

    assert results == episodic
    assert calls == [("agent", episodic)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metrics_increment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=str(tmp_path), embedding_function=lambda t: [[0.0] for _ in t]
    )
    semantic = SemanticMemoryManager(vector, driver=None)
    retriever = MultiLayerRetriever(vector, semantic)

    async def fake_success(agent_id: str, query: str, k: int) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(vector, "aretrieve_relevant_memories", fake_success)
    before = metrics.get_memory_retrievals()
    await retriever.retrieve("agent", "q")
    assert metrics.get_memory_retrievals() == before + 1

    async def fake_fail(agent_id: str, query: str, k: int) -> list[dict[str, Any]]:
        raise RuntimeError("boom")

    monkeypatch.setattr(vector, "aretrieve_relevant_memories", fake_fail)
    err_before = metrics.get_memory_retrieval_errors()
    with pytest.raises(RuntimeError):
        await retriever.retrieve("agent", "q")
    assert metrics.get_memory_retrieval_errors() == err_before + 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_service_metrics_increment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=str(tmp_path), embedding_function=lambda t: [[0.0] for _ in t]
    )
    semantic = SemanticMemoryManager(vector, driver=None)
    service = MemoryService(vector_store=vector, semantic_manager=semantic)

    async def fake_success(agent_id: str, query: str, k: int) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(service.retriever, "retrieve_and_update_semantic", fake_success)
    before = metrics.get_memory_retrievals()
    await service.retrieve_episodic_and_update_semantic("agent", "q")
    assert metrics.get_memory_retrievals() == before + 1

    async def fake_fail(agent_id: str, query: str, k: int) -> list[dict[str, Any]]:
        raise RuntimeError("boom")

    monkeypatch.setattr(service.retriever, "retrieve_and_update_semantic", fake_fail)
    err_before = metrics.get_memory_retrieval_errors()
    with pytest.raises(RuntimeError):
        await service.retrieve_episodic_and_update_semantic("agent", "q")
    assert metrics.get_memory_retrieval_errors() == err_before + 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_recall_metric_only_called_with_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=str(tmp_path), embedding_function=lambda t: [[0.0] for _ in t]
    )
    semantic = SemanticMemoryManager(vector, driver=None)
    retriever = MultiLayerRetriever(vector, semantic)

    async def fake_episodic(agent_id: str, query: str, k: int) -> list[dict[str, Any]]:
        return [{"content": "e1"}]

    monkeypatch.setattr(vector, "aretrieve_relevant_memories", fake_episodic)

    recorded: list[float] = []

    def fake_record_recall(value: float) -> None:
        recorded.append(value)

    monkeypatch.setattr(
        "src.agents.memory.multi_layer_retriever.infra_metrics.record_recall_p5",
        fake_record_recall,
    )

    await retriever.retrieve("agent", "query")
    assert recorded == []

    async def fake_episodic_with_recall(
        agent_id: str, query: str, k: int
    ) -> list[dict[str, Any]]:
        return [
            {
                "content": "e2",
                "metrics": {"recall_p5": 0.8},
            }
        ]

    monkeypatch.setattr(vector, "aretrieve_relevant_memories", fake_episodic_with_recall)

    await retriever.retrieve("agent", "query")
    assert recorded == [pytest.approx(0.8)]
