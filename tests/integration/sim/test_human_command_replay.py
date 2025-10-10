from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from src.infra import event_log
from src.infra.ledger import Ledger
from src.sim.simulation import Simulation


@dataclass
class RecordingState:
    ip: float = 5.0
    du: float = 5.0
    short_term_memory: list[dict[str, Any]] = field(default_factory=list)
    messages_sent_count: int = 0
    last_message_step: int | None = None
    collective_ip: float = 0.0
    collective_du: float = 0.0
    role: str = "tester"
    current_role: str = "tester"
    steps_in_current_role: int = 0
    mood_level: float = 0.0
    mood_value: float = 0.0
    current_project_id: str | None = None
    name: str = "Recorder"
    is_alive: bool = True
    age: int = 0


class RecordingAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = RecordingState()
        self.perceptions: list[list[dict[str, Any]]] = []

    def get_id(self) -> str:  # pragma: no cover - used by scheduler
        return self.agent_id

    def update_state(self, state: RecordingState) -> None:  # pragma: no cover - simple setter
        self.state = state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict[str, Any] | None = None,
        vector_store_manager: Any | None = None,
        knowledge_board: Any | None = None,
        memory_service: Any | None = None,
    ) -> dict[str, Any]:
        perceived = environment_perception.get("perceived_messages", []) if environment_perception else []
        self.perceptions.append([dict(msg) for msg in perceived])
        return {}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_human_command_event_replays_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    ledger = Ledger(ledger_dir / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr("src.sim.simulation.ledger", ledger)

    events_dir = tmp_path / "events"
    events_dir.mkdir()
    event_path = events_dir / "events.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(event_path))
    monkeypatch.setattr(event_log, "_producer", None, raising=False)
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)
    event_log._last_hash = None
    event_log._seed = None
    event_log._seed_cache = {}

    agent = RecordingAgent("agent-1")
    ledger.log_change(agent.agent_id, agent.state.ip, agent.state.du, "init")
    sim = Simulation([agent], seed=123)

    await sim._handle_human_command("hello there")
    await sim.run_step()
    original_messages = agent.perceptions[-1]

    await sim.stop_event_listener()
    sim.close()

    events = event_log.fetch_events(after_step=0, path=event_path)
    assert any(ev.get("type") == "human_command" for ev in events)

    replay_agent = RecordingAgent("agent-1")
    replay_sim = Simulation([replay_agent], seed=123)
    for ev in events:
        replay_sim.apply_event(ev)

    await replay_sim.run_step()
    replayed_messages = replay_agent.perceptions[-1]

    await replay_sim.stop_event_listener()
    replay_sim.close()

    assert replayed_messages == original_messages
