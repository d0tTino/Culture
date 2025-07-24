import json

import pytest

from src.interfaces import dashboard_backend as db


class DummyAgentState:
    def __init__(self, mood: float = 0.5) -> None:
        self.mood_value = mood


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = DummyAgentState()


class DummyMemoryService:
    def get_recent_semantic_summaries(self, agent_id: str, limit: int = 1) -> list[str]:
        return [f"summary-{agent_id}"][:limit]


class DummyMap:
    def to_dict(self) -> dict[str, object]:
        return {"width": 1, "height": 1, "agents": {"a1": [0, 0]}}


class DummySim:
    def __init__(self) -> None:
        self.world_map = DummyMap()
        self.agents = [DummyAgent("a1")]
        self.memory_service = DummyMemoryService()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_map_returns_state(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = DummySim()
    monkeypatch.setitem(db.SIM_STATE, "simulation", sim)
    resp = await db.api_map()
    data = json.loads(resp.body)
    assert data["world_map"]["agents"]["a1"] == [0, 0]
    assert data["agents"]["a1"]["mood"] == 0.5
    assert data["agents"]["a1"]["summary"] == "summary-a1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_map_no_sim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(db.SIM_STATE, "simulation", None)
    resp = await db.api_map()
    data = json.loads(resp.body)
    assert data["world_map"] == {}
    assert data["agents"] == {}
