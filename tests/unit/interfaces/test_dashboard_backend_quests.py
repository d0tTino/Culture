import json

import pytest

from src.interfaces import dashboard_backend as db

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_get_quests_returns_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        db.ledger,
        "get_quests",
        lambda: [
            {
                "id": 1,
                "title": "Q",
                "description": "D",
                "progress": 0,
                "status": "pending",
            }
        ],
    )

    response = await db.get_quests_api()
    assert json.loads(response.body)["quests"][0]["title"] == "Q"
