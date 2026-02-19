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
        "current_rules",
        lambda: [{"rule_id": "r1", "effective_date": "now", "enforcement_stats": {}}],
    )
    monkeypatch.setattr(db.governance_rules_engine, "pending_votes", lambda: [{"id": "p1"}])
    monkeypatch.setattr(db.governance_rules_engine, "active_offices", lambda: [{"office": "council"}])
    monkeypatch.setattr(db.governance_rules_engine, "sanctions", lambda: [{"agent_id": "a1"}])

    resp = await db.api_get_gov()
    assert json.loads(resp.body) == {
        "rules": [{"rule_id": "r1", "effective_date": "now", "enforcement_stats": {}}],
        "current_rules": [{"rule_id": "r1", "effective_date": "now", "enforcement_stats": {}}],
        "pending_votes": [{"id": "p1"}],
        "active_offices": [{"office": "council"}],
        "sanctions": [{"agent_id": "a1"}],
    }
