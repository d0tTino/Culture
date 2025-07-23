import json

import pytest

from src.interfaces import dashboard_backend as db


class DummyMap:
    def to_dict(self) -> dict[str, object]:
        return {"width": 1, "height": 1, "agents": {"a1": [0, 0]}}


class DummySim:
    def __init__(self) -> None:
        self.world_map = DummyMap()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_map_returns_state(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = DummySim()
    monkeypatch.setitem(db.SIM_STATE, "simulation", sim)
    resp = await db.api_map()
    data = json.loads(resp.body)
    assert data["world_map"]["agents"]["a1"] == [0, 0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_map_no_sim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(db.SIM_STATE, "simulation", None)
    resp = await db.api_map()
    data = json.loads(resp.body)
    assert data["world_map"] == {}
