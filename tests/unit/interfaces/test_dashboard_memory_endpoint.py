import json

import pytest

from src.interfaces import dashboard_backend as db


class DummyVectorStore:
    def retrieve_filtered_memories(self, agent_id: str, limit: int = 5):
        return [{"content": "m1"}][:limit]


class DummyMemoryService:
    def __init__(self) -> None:
        self.vector_store = DummyVectorStore()


class DummySim:
    def __init__(self) -> None:
        self.memory_service = DummyMemoryService()


class DummyManager:
    def get_semantic_summaries(self, agent_id: str, limit: int = 5):
        return ["s1"][:limit]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(db.DEFAULT_CONTEXT.sim_state, "semantic_manager", DummyManager())
    monkeypatch.setitem(db.DEFAULT_CONTEXT.sim_state, "simulation", DummySim())

    resp = await db.api_memory("agent", limit=1)
    data = json.loads(resp.body)
    assert data == {"semantic": ["s1"], "episodic": [{"content": "m1"}]}
