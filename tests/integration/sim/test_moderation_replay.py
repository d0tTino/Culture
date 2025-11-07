from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.infra import event_log
from src.infra.ledger import Ledger
from src.sim.simulation import Simulation


@dataclass
class ModeratedAgentState:
    ip: float = 10.0
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
    name: str = "Moderated Agent"
    is_alive: bool = True
    age: int = 0


class ModeratedAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = ModeratedAgentState()

    def get_id(self) -> str:  # pragma: no cover - scheduler compatibility
        return self.agent_id

    def update_state(self, state: ModeratedAgentState) -> None:  # pragma: no cover - simple setter
        self.state = state

    async def run_turn(  # pragma: no cover - not used in this test
        self,
        simulation_step: int,
        environment_perception: dict[str, Any] | None = None,
        vector_store_manager: Any | None = None,
        knowledge_board: Any | None = None,
        memory_service: Any | None = None,
    ) -> dict[str, Any]:
        return {}


class RecordingMemoryService:
    def __init__(self) -> None:
        self.vector_store = SimpleNamespace()
        self.semantic_manager = None
        self.reset_calls: list[str] = []
        self.contents: dict[str, list[str]] = {}

    def reset_agent(self, agent_id: str) -> None:
        self.reset_calls.append(agent_id)
        self.contents[agent_id] = []

    def close(self) -> None:  # pragma: no cover - compatibility stub
        pass


def _prepare_event_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    event_path = events_dir / "events.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(event_path))
    monkeypatch.setenv("ENABLE_REDPANDA", "0")
    monkeypatch.setattr(event_log, "_producer", None, raising=False)
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)
    event_log._last_hash = None
    event_log._seed = None
    event_log._seed_cache = {}
    event_log._header_written = set()
    return event_path


def _prepare_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Ledger:
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    ledger = Ledger(ledger_dir / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr("src.sim.simulation.ledger", ledger)
    monkeypatch.setattr("src.sim.resources.ledger", ledger)
    return ledger


@pytest.mark.asyncio
@pytest.mark.integration
async def test_moderation_events_replay_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    event_path = _prepare_event_log(tmp_path, monkeypatch)
    ledger = _prepare_ledger(tmp_path, monkeypatch)

    live_memory = RecordingMemoryService()
    agent = ModeratedAgent("agent-moderated")
    live_memory.contents[agent.agent_id] = ["keep"]
    ledger.log_change(agent.agent_id, agent.state.ip, agent.state.du, "init")
    sim = Simulation([agent], memory_service=live_memory, seed=321)

    await sim.mute_agent(agent.agent_id)
    sim.current_step += 1
    await sim.apply_penalty(agent.agent_id, ip=2.0, du=1.0)
    sim.current_step += 1
    await sim.reset_memory(agent.agent_id)
    sim.current_step += 1
    await sim.unmute_agent(agent.agent_id)

    expected_ip = agent.state.ip
    expected_du = agent.state.du

    await sim.stop_event_listener()
    sim.close()

    events = event_log.fetch_events(after_step=-1, path=event_path)
    moderation_events = [ev for ev in events if ev.get("type") == "moderation"]
    assert {ev.get("action") for ev in moderation_events} == {
        "mute",
        "penalty",
        "reset_memory",
        "unmute",
    }

    replay_tmp = tmp_path / "replay"
    replay_tmp.mkdir()
    replay_ledger = _prepare_ledger(replay_tmp, monkeypatch)

    replay_memory = RecordingMemoryService()
    replay_agent = ModeratedAgent(agent.agent_id)
    replay_agent.state.ip = 10.0
    replay_agent.state.du = 5.0
    replay_memory.contents[replay_agent.agent_id] = ["stale"]
    replay_ledger.log_change(
        replay_agent.agent_id, replay_agent.state.ip, replay_agent.state.du, "init"
    )
    replay_sim = Simulation([replay_agent], memory_service=replay_memory, seed=321)

    for event in moderation_events:
        replay_sim.apply_event(event)

    assert replay_agent.agent_id not in replay_sim.muted_agents
    assert replay_agent.state.ip == pytest.approx(expected_ip)
    assert replay_agent.state.du == pytest.approx(expected_du)
    assert replay_memory.reset_calls == [replay_agent.agent_id]
    assert replay_memory.contents[replay_agent.agent_id] == []

    balance_ip, balance_du = replay_ledger.get_balance(replay_agent.agent_id)
    assert balance_ip == pytest.approx(expected_ip)
    assert balance_du == pytest.approx(expected_du)

    await replay_sim.stop_event_listener()
    replay_sim.close()
