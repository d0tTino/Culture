import json

import pytest

from src.interfaces import dashboard_backend as db


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_missions_returns_versioned_schema(tmp_path, monkeypatch):
    missions = [{"id": 1, "name": "Test", "status": "Pending", "progress": 0}]
    path = tmp_path / "missions.json"
    path.write_text(json.dumps(missions))
    monkeypatch.setattr(db, "MISSIONS_PATH", path)

    response = await db.get_missions()
    payload = json.loads(response.body)
    assert payload["schema"] == "dashboard.missions"
    assert payload["enabled"] is True
    assert payload["missions"] == missions
    assert payload["data"]["missions"] == missions


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_missions_returns_deterministic_fallback_when_unseeded(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "MISSIONS_PATH", tmp_path / "missing.json")

    response = await db.get_missions()
    payload = json.loads(response.body)
    assert payload["enabled"] is False
    assert payload["missions"] == []
    assert "Run scripts/seed_dashboard_dev.py" in payload["fallback"]["reason"]
