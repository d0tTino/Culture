from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from src.infra import config
from src.interfaces import dashboard_backend as db


@pytest.mark.integration
def test_public_persistent_profile_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE", "public_persistent")
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setenv("OPA_URL", "http://localhost:8181")
    monkeypatch.setenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

    cfg = config.load_config(validate_required=False)

    assert cfg["PROFILE"] == "public_persistent"
    assert cfg["SNAPSHOT_INTERVAL_STEPS"] == 25
    assert cfg["MEMORY_PRUNING_ENABLED"] is True
    assert cfg["USE_COUNCIL_MODE"] is True
    assert cfg["KNOWLEDGE_BOARD_BACKEND"] == "graph"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_public_persistent_dashboard_and_event_ingestion() -> None:
    health = await db.health()
    ready = await db.readiness()

    health_status = getattr(health, "status_code", 200)
    ready_status = getattr(ready, "status_code", 200)
    health_body = getattr(health, "body", b"")
    ready_body = getattr(ready, "body", b"")

    assert health_status == 200
    assert ready_status in {200, 503}
    assert b"subsystems" in health_body
    assert b"subsystems" in ready_body

    embed = db.board_payload_to_embed({"agent_id": "agent_1", "content": "hello", "step": 1})
    assert "Knowledge Board Entry" in embed["title"]

    bus = db.get_event_bus()
    queue = bus.subscribe()
    try:
        await db.emit_event(db.SimulationEvent(type="discord_message", data={"step": 1}))
        event = await asyncio.wait_for(queue.get(), timeout=1)
    finally:
        bus.unsubscribe(queue)

    assert event is not None
    assert event.type == "discord_message"


@pytest.mark.integration
def test_public_persistent_startup_script_reports_actionable_failures() -> None:
    script_path = (
        Path(__file__).resolve().parents[1] / ".." / "scripts" / "start_public_persistent.sh"
    ).resolve()
    env = dict(os.environ)
    env["LLM_API_BASE"] = "http://127.0.0.1:9"
    result = subprocess.run(
        ["bash", str(script_path), "--steps", "1"],
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
        check=False,
    )

    assert result.returncode != 0
    assert "Public persistent profile preflight failed" in result.stderr
    assert "LLM endpoint not reachable" in result.stderr
