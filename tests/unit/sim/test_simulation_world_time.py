from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.sim.persistence.snapshot_service import SnapshotPersistenceService
from src.sim.simulation import Simulation


class DummyAgent:
    def __init__(self) -> None:
        self.agent_id = "agent"
        self.last_perception = None
        self.state = SimpleNamespace(
            agent_id="agent",
            ip=1.0,
            du=1.0,
            messages_sent_count=0,
            last_message_step=0,
            short_term_memory=[],
            mood_level=0.0,
            is_alive=True,
            age=0,
        )

    async def run_turn(self, **kwargs):
        self.last_perception = kwargs.get("environment_perception")
        return {"action_intent": "idle"}

    def update_state(self, state):
        self.state = state

    def get_id(self):
        return self.agent_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_environment_tick_rollover_and_daily_reset(monkeypatch):
    monkeypatch.setenv("WORLD_TICKS_PER_DAY", "2")
    monkeypatch.setenv("WORLD_SEASON_LENGTH_DAYS", "0")
    monkeypatch.setenv("WORLD_TIME_BROADCAST_CADENCE_TICKS", "2")

    monkeypatch.setattr("src.sim.simulation.evaluate_policy", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "src.sim.simulation.log_event", lambda data: {**data, "trace_hash": "hash"}
    )
    emit_event = AsyncMock()
    monkeypatch.setattr("src.sim.simulation.emit_event", emit_event)
    rm = SimpleNamespace(
        cap_tick=lambda **kwargs: None, set_du_budget=lambda *args, **kwargs: None
    )
    monkeypatch.setattr("src.sim.simulation.get_resource_manager", lambda: rm)

    sim = Simulation(
        [DummyAgent()], memory_service=SimpleNamespace(vector_store=None, semantic_manager=None)
    )
    sim.personality_engine.apply_experience_drift = lambda *args, **kwargs: []
    sim._assert_trait_update_invariants = lambda *args, **kwargs: None
    await sim._run_agent_turn(0)
    assert sim.world_hour == 0
    assert sim.world_day == 0

    await sim._run_agent_turn(0)
    assert sim.world_hour == 1
    assert sim.world_day == 0

    await sim._run_agent_turn(0)
    assert sim.world_hour == 0
    assert sim.world_day == 1

    env_events = [
        call.args[0].data
        for call in emit_event.await_args_list
        if call.args[0].type == "environment"
    ]
    assert any(event.get("event_name") == "daily_reset" for event in env_events)
    assert any(event.get("event_name") == "world_time" for event in env_events)
    await sim.stop_event_listener()
    sim.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_environment_context_in_perception(monkeypatch):
    monkeypatch.setenv("WORLD_TICKS_PER_DAY", "24")
    monkeypatch.setenv("WORLD_SEASON_LENGTH_DAYS", "2")

    monkeypatch.setattr("src.sim.simulation.evaluate_policy", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "src.sim.simulation.log_event", lambda data: {**data, "trace_hash": "hash"}
    )
    monkeypatch.setattr("src.sim.simulation.emit_event", AsyncMock())
    rm = SimpleNamespace(
        cap_tick=lambda **kwargs: None, set_du_budget=lambda *args, **kwargs: None
    )
    monkeypatch.setattr("src.sim.simulation.get_resource_manager", lambda: rm)

    agent = DummyAgent()
    sim = Simulation(
        [agent], memory_service=SimpleNamespace(vector_store=None, semantic_manager=None)
    )
    sim.personality_engine.apply_experience_drift = lambda *args, **kwargs: []
    sim._assert_trait_update_invariants = lambda *args, **kwargs: None
    await sim._run_agent_turn(0)

    assert agent.last_perception is not None
    environment_context = agent.last_perception.get("environment_context")
    assert environment_context is not None
    assert "time" in environment_context
    assert "effect_hooks" in environment_context
    assert environment_context["time"]["world_hour"] == sim.world_hour
    await sim.stop_event_listener()
    sim.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_action_event_contains_environment_context(monkeypatch):
    monkeypatch.setenv("WORLD_TICK_TURN_QUANTUM", "2")

    monkeypatch.setattr("src.sim.simulation.evaluate_policy", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "src.sim.simulation.log_event", lambda data: {**data, "trace_hash": "hash"}
    )
    emit_event = AsyncMock()
    monkeypatch.setattr("src.sim.simulation.emit_event", emit_event)
    rm = SimpleNamespace(
        cap_tick=lambda **kwargs: None, set_du_budget=lambda *args, **kwargs: None
    )
    monkeypatch.setattr("src.sim.simulation.get_resource_manager", lambda: rm)

    agent = DummyAgent()
    sim = Simulation(
        [agent], memory_service=SimpleNamespace(vector_store=None, semantic_manager=None)
    )
    sim.personality_engine.apply_experience_drift = lambda *args, **kwargs: []
    sim._assert_trait_update_invariants = lambda *args, **kwargs: None

    await sim._run_agent_turn(0)

    agent_action_events = [
        call.args[0].data
        for call in emit_event.await_args_list
        if call.args[0].type == "agent_action"
    ]
    assert agent_action_events
    payload = agent_action_events[-1]
    assert payload["turn_index"] == sim.current_step
    assert payload["environment_context"]["time"]["world_tick"] == sim.world_tick_index
    assert payload["environment_context"]["time"]["world_day"] == sim.world_day
    await sim.stop_event_listener()
    sim.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_world_context_projection_consistent_across_consumers(monkeypatch):
    monkeypatch.setenv("WORLD_TICK_TURN_QUANTUM", "1")
    monkeypatch.setenv("WORLD_TIME_BROADCAST_CADENCE_TICKS", "1")
    monkeypatch.setenv("SNAPSHOT_INTERVAL_STEPS", "1")

    monkeypatch.setattr("src.sim.simulation.evaluate_policy", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "src.sim.simulation.log_event", lambda data: {**data, "trace_hash": "hash"}
    )
    emit_event = AsyncMock()
    monkeypatch.setattr("src.sim.simulation.emit_event", emit_event)
    rm = SimpleNamespace(
        cap_tick=lambda **kwargs: None, set_du_budget=lambda *args, **kwargs: None
    )
    monkeypatch.setattr("src.sim.simulation.get_resource_manager", lambda: rm)

    captured_snapshots: list[dict] = []

    def _save(step: int, snapshot: dict, directory: str = "snapshots") -> None:
        captured_snapshots.append(snapshot)

    monkeypatch.setattr(SnapshotPersistenceService, "save", staticmethod(_save))
    monkeypatch.setattr(SnapshotPersistenceService, "upload", staticmethod(lambda *args, **kwargs: None))

    agent = DummyAgent()
    sim = Simulation(
        [agent], memory_service=SimpleNamespace(vector_store=None, semantic_manager=None)
    )
    sim.personality_engine.apply_experience_drift = lambda *args, **kwargs: []
    sim._assert_trait_update_invariants = lambda *args, **kwargs: None

    await sim._run_agent_turn(0)
    await sim._run_agent_turn(0)

    projection_from_perception = agent.last_perception["environment_context"]

    environment_event = next(
        call.args[0].data
        for call in emit_event.await_args_list
        if call.args[0].type == "environment" and call.args[0].data.get("step") == sim.current_step
    )
    agent_action_event = next(
        call.args[0].data
        for call in emit_event.await_args_list
        if call.args[0].type == "agent_action" and call.args[0].data.get("step") == sim.current_step
    )
    snapshot = captured_snapshots[-1]

    expected_projection = agent_action_event["world_context_projection"]
    assert environment_event["world_context_projection"] == expected_projection
    assert snapshot["metadata"]["world_context_projection"] == expected_projection
    assert projection_from_perception["time"] == expected_projection["time"]
    assert projection_from_perception["weather"] == expected_projection["weather"]
    assert projection_from_perception["season"] == expected_projection["season"]
    assert projection_from_perception["council_window_active"] == expected_projection["council"][
        "council_window_active"
    ]
    assert projection_from_perception["map_neighborhood"] == expected_projection["map_neighborhood"]
    assert expected_projection["projection_version"] == 1

    await sim.stop_event_listener()
    sim.close()
