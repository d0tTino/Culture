import pytest

from src.agents.memory.memory_service import MemoryService
from src.interfaces import metrics

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_rag_hit_rate_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MemoryService()

    async def fake_episodic(agent_id: str, query: str, k: int) -> list[dict[str, int]]:
        return [{"id": 1}, {"id": 2}]

    def fake_semantic(agent_id: str, limit: int) -> list[str]:
        return ["s1"]

    monkeypatch.setattr(service, "retrieve_episodic_and_update_semantic", fake_episodic)
    monkeypatch.setattr(service, "get_recent_semantic_summaries", fake_semantic)
    metrics.RAG_HIT_RATE.set(0)

    await service.get_context_pipeline("agent", "query", k=5, semantic_limit=3)

    assert metrics.get_rag_hit_rate() == pytest.approx(3 / 8)
