import json

import pytest

from src.interfaces import dashboard_backend as db


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_widget_adds_name() -> None:
    db.REGISTERED_WIDGETS.clear()
    resp = await db.register_widget({"name": "extra"})
    data = json.loads(resp.body)
    assert "extra" in db.REGISTERED_WIDGETS
    assert "extra" in data["widgets"]
