from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.sim.contracts.lifecycle import EVENT_STEP_LIFECYCLE_CONTRACTS, LIFECYCLE_CONTRACT_VERSION
from src.sim.simulation import EVENT_STEP_LIFECYCLE_MUST_NOT_CHANGE, Simulation

FIXTURES = Path(__file__).with_name("fixtures")


class DummySnapshotState:
    def __init__(self) -> None:
        self.ip = 0.0
        self.du = 0.0
        self.mood_level = 0.0
        self.lifecycle_state = None
        self.lifecycle_history = []
        self.legacy_artifacts = {}
        self.memory_archival_policy = {}
        self.predecessor_id = None
        self.successor_id = None


class DummySnapshotAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = DummySnapshotState()

    def get_id(self) -> str:
        return self.agent_id


@pytest.mark.integration
def test_lifecycle_contract_alias_is_backward_compatible() -> None:
    assert LIFECYCLE_CONTRACT_VERSION == "1.0.0"
    assert EVENT_STEP_LIFECYCLE_MUST_NOT_CHANGE == EVENT_STEP_LIFECYCLE_CONTRACTS


@pytest.mark.integration
def test_from_snapshot_preserves_historical_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.agents.core.base_agent.Agent", DummySnapshotAgent)
    historical = json.loads((FIXTURES / "historical_snapshot_v2.json").read_text(encoding="utf-8"))
    normalized = json.loads(
        (FIXTURES / "normalized_from_v2_snapshot_v3.json").read_text(encoding="utf-8")
    )

    sim = Simulation.from_snapshot(historical)

    assert sim.current_step == normalized["step"]
    assert sim.world_day == normalized["environment_state"]["world_day"]
    assert sim.world_hour == normalized["environment_state"]["world_hour"]
    assert sim.world_tick_index == normalized["environment_state"]["world_tick"]


@pytest.mark.integration
def test_replay_from_snapshot_preserves_event_reducer_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.agents.core.base_agent.Agent", DummySnapshotAgent)
    snapshot = json.loads((FIXTURES / "normalized_from_v1_snapshot_v3.json").read_text(encoding="utf-8"))
    events = [
        {"type": "tick", "step": snapshot["step"] + 1, "agent_id": snapshot["agents"][0]["agent_id"]}
    ]

    monkeypatch.setattr(
        "src.sim.persistence.snapshot_service.SnapshotPersistenceService.load", lambda _p: snapshot
    )
    monkeypatch.setattr("src.infra.event_log.stream_events", lambda **_kwargs: iter(events))

    replay = Simulation.replay_from_snapshot("ignored.json")

    assert replay.current_step >= snapshot["step"]
