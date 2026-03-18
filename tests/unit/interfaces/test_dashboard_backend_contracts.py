import json

import pytest

from src.interfaces import dashboard_backend as db


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_capabilities_reports_expected_subsystems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(db, "MISSIONS_PATH", db.DEV_DATA_DIR / "missing.json")
    monkeypatch.setitem(db.DEFAULT_CONTEXT.sim_state, "simulation", None)

    response = await db.api_capabilities()
    payload = json.loads(response.body)

    assert payload["schema"] == "dashboard.capabilities"
    assert set(payload["capabilities"]).issuperset(
        {"graph_kb", "governance", "map", "memory", "moderation"}
    )
    assert payload["capabilities"]["governance"]["enabled"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_versioned_endpoint_contracts_embed_data_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(db.DEFAULT_CONTEXT.sim_state, "simulation", None)

    map_payload = json.loads((await db.api_map()).body)
    memory_payload = json.loads((await db.api_memory("agent-1")).body)
    gov_payload = json.loads((await db.api_get_gov()).body)
    observability_payload = json.loads((await db.api_observability_metrics()).body)

    assert map_payload["schema"] == "dashboard.map_state"
    assert map_payload["enabled"] is False
    assert map_payload["data"]["world_map"] == {}

    assert memory_payload["schema"] == "dashboard.memory_views"
    assert memory_payload["episodic"] == []
    assert memory_payload["fallback"]["action"]

    assert gov_payload["schema"] == "dashboard.governance"
    assert gov_payload["enabled"] is False
    assert "rules" in gov_payload["data"]

    assert observability_payload["schema"] == "dashboard.observability"
    assert "du_per_1k_tokens" in observability_payload["data"]
