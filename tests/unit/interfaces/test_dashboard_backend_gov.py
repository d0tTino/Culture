import json

import pytest

pytest.importorskip("pytest_asyncio")

from src.interfaces import dashboard_backend as db


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_get_gov_without_sim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(db.SIM_STATE, "simulation", None)
    monkeypatch.setattr(
        db.governance_rules_engine,
        "active_rules_read_model",
        lambda: [{"rule_id": "r1", "effective_date": "now", "enforcement_stats": {}}],
    )
    resp = await db.api_get_gov()
    assert json.loads(resp.body) == {
        "rules": [{"rule_id": "r1", "effective_date": "now", "enforcement_stats": {}}]
    }
