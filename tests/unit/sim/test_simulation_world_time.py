from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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
async def test_world_time_rollover_hour_to_day(monkeypatch):
    monkeypatch.setenv("WORLD_TICKS_PER_DAY", "2")
    monkeypatch.setenv("WORLD_SEASON_LENGTH_DAYS", "0")
    monkeypatch.setenv("WORLD_TIME_BROADCAST_CADENCE_TICKS", "2")

    monkeypatch.setattr("src.sim.simulation.evaluate_policy", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "src.sim.simulation.log_event", lambda data: {**data, "trace_hash": "hash"}
    )
    monkeypatch.setattr("src.sim.simulation.emit_event", AsyncMock())
    rm = SimpleNamespace(
        cap_tick=lambda **kwargs: None, set_du_budget=lambda *args, **kwargs: None
    )
    monkeypatch.setattr("src.sim.simulation.get_resource_manager", lambda: rm)

    sim = Simulation(
        [DummyAgent()], memory_service=SimpleNamespace(vector_store=None, semantic_manager=None)
    )
    sim.current_step = 1
    await sim._advance_world_time()
    assert sim.world_hour == 1
    assert sim.world_day == 0

    sim.current_step = 2
    await sim._advance_world_time()
    assert sim.world_hour == 0
    assert sim.world_day == 1
    assert sim.world_season is None
    sim.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_world_time_rollover_day_to_season(monkeypatch):
    monkeypatch.setenv("WORLD_TICKS_PER_DAY", "2")
    monkeypatch.setenv("WORLD_SEASON_LENGTH_DAYS", "2")
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

    for step in range(1, 5):
        sim.current_step = step
        await sim._advance_world_time()

    assert sim.world_day == 2
    assert sim.world_season == 1
    assert any(call.args[0].type == "world_time" for call in emit_event.await_args_list)
    sim.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_turn_includes_world_time_perception(monkeypatch):
    monkeypatch.setenv("WORLD_TICKS_PER_DAY", "24")
    monkeypatch.setenv("WORLD_SEASON_LENGTH_DAYS", "0")
    monkeypatch.setenv("WORLD_TIME_BROADCAST_CADENCE_TICKS", "24")

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
    await sim._run_agent_turn(0)

    assert agent.last_perception is not None
    world_time = agent.last_perception.get("world_time")
    assert world_time is not None
    assert world_time["world_hour"] == sim.world_hour
    assert world_time["world_day"] == sim.world_day
    assert "formatted" in world_time
    sim.close()
