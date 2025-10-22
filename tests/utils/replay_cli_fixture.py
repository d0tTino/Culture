from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from src.infra import event_log
from src.infra.snapshot import compute_trace_hash, save_snapshot
from src.sim.knowledge_board import BoardEntry
from src.sim.simulation import Simulation

__all__ = [
    "ReplayCLIArtifacts",
    "ReplayFixtureAgent",
    "ReplayFixtureState",
    "create_replay_cli_artifacts",
]


@dataclass(slots=True)
class ReplayCLIArtifacts:
    """Artifacts produced by the replay CLI fixture."""

    snapshot_path: Path
    events_path: Path
    replay_slice_path: Path
    start_step: int
    end_step: int
    expected_step: int
    expected_knowledge_entries: list[dict]


class ReplayFixtureState(SimpleNamespace):
    ip: float = 0.0
    du: float = 0.0
    short_term_memory: ClassVar[list] = []
    messages_sent_count: int = 0
    last_message_step: int = 0
    relationships: ClassVar[dict] = {}
    role: str = "dummy"
    steps_in_current_role: int = 0
    mood_level: float = 0.0

    def update_collective_metrics(self, ip: float, du: float) -> None:  # pragma: no cover - stub
        return None


class ReplayFixtureAgent:
    def __init__(self, agent_id: str = "agent-1") -> None:
        self.agent_id = agent_id
        self.state = ReplayFixtureState()
        self._added = False

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, new_state: ReplayFixtureState) -> None:
        self.state = new_state

    async def run_turn(  # pragma: no cover - executed via simulation
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        memory_service: object | None = None,
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
        **_: object,
    ) -> dict:
        if not self._added and knowledge_board is not None:
            knowledge_board.add_entry(
                BoardEntry(
                    content_full="fixture-entry",
                    entry_type="test",
                    tags=["replay", "fixture"],
                ),
                self.agent_id,
                simulation_step,
            )
            self._added = True
        return {}


def _snapshot_from_sim(sim: Simulation) -> dict:
    snapshot = {
        "step": sim.current_step,
        "collective_ip": sim.collective_ip,
        "collective_du": sim.collective_du,
        "knowledge_board": sim.knowledge_board.to_dict(),
        "world_map": sim.world_map.to_dict(),
        "agents": [
            {
                "agent_id": agent.agent_id,
                "ip": agent.state.ip,
                "du": agent.state.du,
            }
            for agent in sim.agents
        ],
    }
    snapshot_no_vector = {
        **{k: v for k, v in snapshot.items() if k != "trace_hash"},
        "knowledge_board": {
            k: v for k, v in snapshot["knowledge_board"].items() if k != "vector"
        },
        "world_map": {k: v for k, v in snapshot["world_map"].items() if k != "vector"},
    }
    snapshot["trace_hash"] = compute_trace_hash(snapshot_no_vector)
    return snapshot


async def create_replay_cli_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ReplayCLIArtifacts:
    """Generate snapshot and replay slice suitable for exercising the replay CLI."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENABLE_REDPANDA", "0")
    monkeypatch.setenv("SNAPSHOT_INTERVAL_STEPS", "1")
    events_path = tmp_path / "event_log.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(events_path))
    monkeypatch.setattr(event_log, "_header_written", set(), raising=False)
    monkeypatch.setattr(event_log, "_seed_cache", {}, raising=False)
    monkeypatch.setattr(event_log, "_seed", None, raising=False)
    monkeypatch.setattr(event_log, "_last_hash", None, raising=False)
    monkeypatch.setattr(event_log, "_producer", None, raising=False)
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)
    monkeypatch.setattr("src.sim.simulation.upload_snapshot", lambda *_, **__: None)

    def _save(step: int, data: dict, directory: Path = tmp_path) -> None:
        save_snapshot(step, data, directory=directory)

    monkeypatch.setattr("src.sim.simulation.save_snapshot", _save)

    agent = ReplayFixtureAgent()
    sim = Simulation(agents=[agent])
    sim.evaluation_hooks = []

    base_snapshot = _snapshot_from_sim(sim)
    save_snapshot(0, base_snapshot, directory=tmp_path)
    snapshot_path = tmp_path / "snapshot_0.json"

    await sim.run_step()
    end_step = sim.current_step
    expected_entries = list(sim.knowledge_board.to_dict().get("entries", []))

    replay_slice_path = event_log.store_replay_slice(1, end_step, directory=tmp_path)

    sim.close()

    return ReplayCLIArtifacts(
        snapshot_path=snapshot_path,
        events_path=events_path,
        replay_slice_path=replay_slice_path,
        start_step=1,
        end_step=end_step,
        expected_step=end_step,
        expected_knowledge_entries=expected_entries,
    )
