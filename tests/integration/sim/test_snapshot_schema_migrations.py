from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.sim.persistence.snapshot_migrations import (
    CURRENT_SNAPSHOT_SCHEMA_VERSION,
    migrate_snapshot,
)
from src.sim.simulation import Simulation

_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    with (_FIXTURE_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.parametrize(
    ("historical_fixture", "normalized_fixture"),
    [
        ("historical_snapshot_v1.json", "normalized_from_v1_snapshot_v3.json"),
        ("historical_snapshot_v2.json", "normalized_from_v2_snapshot_v3.json"),
    ],
)
@pytest.mark.integration
def test_historical_snapshot_normalization_is_deterministic(
    historical_fixture: str,
    normalized_fixture: str,
) -> None:
    historical_snapshot = _load_fixture(historical_fixture)
    expected = _load_fixture(normalized_fixture)

    normalized = migrate_snapshot(historical_snapshot)
    normalized_again = migrate_snapshot(normalized)

    assert normalized == expected
    assert normalized_again == expected
    assert normalized["snapshot_schema_version"] == CURRENT_SNAPSHOT_SCHEMA_VERSION


@pytest.mark.parametrize("historical_fixture", ["historical_snapshot_v1.json", "historical_snapshot_v2.json"])
@pytest.mark.integration
def test_historical_snapshot_replay_compatibility(historical_fixture: str, tmp_path: Path) -> None:
    raw_snapshot = _load_fixture(historical_fixture)
    snapshot_path = tmp_path / "snapshot.json"
    events_path = tmp_path / "events.jsonl"

    snapshot_path.write_text(json.dumps(raw_snapshot), encoding="utf-8")
    events_path.write_text(json.dumps({"type": "header", "seed": raw_snapshot.get("seed", 0)}) + "\n")

    normalized = migrate_snapshot(raw_snapshot)
    from_normalized = Simulation.from_snapshot(normalized)
    replayed = Simulation.replay_from_snapshot(snapshot_path, events_path=events_path)

    assert replayed.current_step == from_normalized.current_step
    assert replayed.turns_per_world_tick == from_normalized.turns_per_world_tick
    assert replayed.world_day == from_normalized.world_day
    assert replayed.world_hour == from_normalized.world_hour
    assert replayed.world_tick_index == from_normalized.world_tick_index
    assert replayed.world_map.agent_positions == from_normalized.world_map.agent_positions
    assert [agent.agent_id for agent in replayed.agents] == [
        agent.agent_id for agent in from_normalized.agents
    ]
