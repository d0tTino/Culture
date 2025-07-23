import json

import pytest

from src.interfaces import dashboard_backend as db


class DummyManager:
    def __init__(self, summaries: list[str] | None = None, raise_exc: bool = False) -> None:
        self.summaries = summaries or []
        self.raise_exc = raise_exc
        self.calls: list[tuple[str, int]] = []

    def get_semantic_summaries(self, agent_id: str, limit: int = 3) -> list[str]:
        self.calls.append((agent_id, limit))
        if self.raise_exc:
            raise RuntimeError("boom")
        return self.summaries[:limit]


class DummyResponse:
    def __init__(self, content: object, status_code: int = 200, **_: object) -> None:
        self.body = json.dumps(content).encode("utf-8")
        self.status_code = status_code


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_semantic_summaries_with_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = DummyManager(["s1", "s2"])
    monkeypatch.setitem(db.SIM_STATE, "semantic_manager", manager)

    resp = await db.get_semantic_summaries("agent", limit=1)
    data = json.loads(resp.body)
    assert data == {"summaries": ["s1"]}
    assert manager.calls == [("agent", 1)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_semantic_summaries_no_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(db.SIM_STATE, "semantic_manager", None)

    resp = await db.get_semantic_summaries("agent")
    data = json.loads(resp.body)
    assert data == {"summaries": []}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_semantic_summaries_error(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = DummyManager(raise_exc=True)
    monkeypatch.setitem(db.SIM_STATE, "semantic_manager", manager)

    resp = await db.get_semantic_summaries("agent")
    data = json.loads(resp.body)
    assert data == db.SEMANTIC_SUMMARIES_ERROR


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_semantic_summaries_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = DummyManager(raise_exc=True)
    monkeypatch.setitem(db.SIM_STATE, "semantic_manager", manager)
    monkeypatch.setattr(db, "JSONResponse", DummyResponse)

    resp = await db.get_semantic_summaries("agent")
    assert resp.status_code == 500
