import json

import pytest

from src.interfaces import dashboard_backend as db


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_missions_returns_data(tmp_path, monkeypatch):
    missions = [{"id": 1, "name": "Test", "status": "Pending", "progress": 0}]
    path = tmp_path / "missions.json"
    path.write_text(json.dumps(missions))
    monkeypatch.setattr(db, "MISSIONS_PATH", path)

    response = await db.get_missions()
    assert json.loads(response.body) == missions
