from __future__ import annotations

import importlib
import json

import pytest


@pytest.mark.integration
def test_du_budget_exceeded_event_logged(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "event_log.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(log_path))

    import src.infra.event_log as event_log_module

    importlib.reload(event_log_module)

    import src.sim.resource_manager as resource_manager_module

    importlib.reload(resource_manager_module)

    rm = resource_manager_module.ResourceManager(5.0, 5.0)
    rm.set_du_budget("agent-1", 0.1)

    with pytest.raises(RuntimeError):
        rm.ensure_du_budget("agent-1", 0.5)

    assert log_path.exists()

    with log_path.open("r", encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh if line.strip()]

    du_events = [event for event in events if event.get("type") == "du_budget_exceeded"]
    assert du_events, "Expected a du_budget_exceeded event to be logged"

    event = du_events[-1]
    assert event["agent"] == "agent-1"
    assert event["required"] == pytest.approx(0.5)
    assert event["remaining"] == pytest.approx(0.1)
    assert isinstance(event.get("seed"), int)
