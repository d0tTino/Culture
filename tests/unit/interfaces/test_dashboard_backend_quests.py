import json

import pytest

from src.interfaces import dashboard_backend as db
from src.sim.quests import Quest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_get_quests_returns_data(monkeypatch: pytest.MonkeyPatch) -> None:
    quests = [Quest(id=1, title="Q", description="D", progress=0, status="pending")]
    monkeypatch.setattr(db, "get_quests", lambda: quests)

    response = await db.get_quests_api()
    assert json.loads(response.body) == {"quests": [q.model_dump() for q in quests]}
