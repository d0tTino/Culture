import json

import pytest

from src.interfaces import dashboard_backend as db


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_widget_adds_name() -> None:
    db.WIDGET_REGISTRY._widgets.clear()
    resp = await db.register_widget({"name": "extra", "script_url": "s.js"})
    data = json.loads(resp.body)
    assert db.WIDGET_REGISTRY.get("extra") == {"script_url": "s.js"}
    assert any(w["name"] == "extra" for w in data["widgets"])
