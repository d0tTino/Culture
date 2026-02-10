import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class DummyNeo4j:
    Driver = object
    GraphDatabase = object


class SeedAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._state = SimpleNamespace(ip=0.0, du=0.0)

    def get_id(self) -> str:
        return self.agent_id

    @property
    def state(self) -> SimpleNamespace:
        return self._state


@pytest.mark.unit
@pytest.mark.asyncio
async def test_spawn_control_applies_role_traits_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.sim.simulation import Simulation

    spawned: list[object] = []

    class DummySpawnAgent:
        def __init__(self, agent_id: str, name: str, initial_state: dict | None = None) -> None:
            state = initial_state or {}
            self.agent_id = agent_id
            self.name = name
            self._state = SimpleNamespace(
                ip=0.0,
                du=0.0,
                current_role=state.get("current_role"),
                traits=state.get("traits"),
                persona=state.get("persona"),
                backstory=state.get("backstory"),
                parent_id=None,
                genes={},
            )
            spawned.append(self)

        def get_id(self) -> str:
            return self.agent_id

        @property
        def state(self) -> SimpleNamespace:
            return self._state

    import src.agents.core.base_agent as base_agent_module

    monkeypatch.setattr(base_agent_module, "Agent", DummySpawnAgent)

    sim = Simulation([SeedAgent("seed")])
    await sim.handle_control_command(
        {
            "command": "spawn",
            "agent_id": "child-1",
            "role": "Analyzer",
            "persona": "Precise and skeptical",
            "backstory": "Former auditor",
            "traits": {"openness": 0.2, "resilience": "0.9"},
        }
    )

    assert len(spawned) == 1
    child = spawned[0]
    assert child.state.current_role.name == "Analyzer"
    assert child.state.traits.openness == pytest.approx(0.2)
    assert child.state.traits.resilience == pytest.approx(0.9)
    assert child.state.persona == "Precise and skeptical"
    assert child.state.backstory == "Former auditor"
    sim.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_spawn_control_rejects_invalid_trait_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.sim.simulation import Simulation

    class DummySpawnAgent:
        def __init__(self, agent_id: str, name: str, initial_state: dict | None = None) -> None:
            self.agent_id = agent_id
            self._state = SimpleNamespace(ip=0.0, du=0.0, parent_id=None, genes={})

        def get_id(self) -> str:
            return self.agent_id

        @property
        def state(self) -> SimpleNamespace:
            return self._state

    import src.agents.core.base_agent as base_agent_module

    monkeypatch.setattr(base_agent_module, "Agent", DummySpawnAgent)

    sim = Simulation([SeedAgent("seed")])
    before = len(sim.agents)
    await sim.handle_control_command(
        {
            "command": "spawn",
            "agent_id": "bad-child",
            "traits": {"openness": 2.0},
        }
    )
    assert len(sim.agents) == before
    sim.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_spawn_control_rejects_duplicate_agent_id(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.interfaces.metrics import ACTIVE_AGENT_COUNT
    from src.sim import simulation as simulation_module
    from src.sim.simulation import Simulation

    spawned: list[object] = []

    class DummySpawnAgent:
        def __init__(self, agent_id: str, name: str, initial_state: dict | None = None) -> None:
            self.agent_id = agent_id
            self.name = name
            self._state = SimpleNamespace(ip=0.0, du=0.0, parent_id=None, genes={})
            spawned.append(self)

        def get_id(self) -> str:
            return self.agent_id

        @property
        def state(self) -> SimpleNamespace:
            return self._state

    import src.agents.core.base_agent as base_agent_module

    monkeypatch.setattr(base_agent_module, "Agent", DummySpawnAgent)
    emit_event_mock = AsyncMock()
    monkeypatch.setattr(simulation_module, "emit_event", emit_event_mock)

    sim = Simulation([SeedAgent("seed")])
    before = len(sim.agents)

    await sim.handle_control_command({"command": "spawn", "agent_id": "child-1"})
    after_first_spawn = len(sim.agents)
    await sim.handle_control_command({"command": "spawn", "agent_id": "child-1"})

    assert len(spawned) == 1
    assert after_first_spawn == before + 1
    assert len(sim.agents) == after_first_spawn
    assert ACTIVE_AGENT_COUNT._value.get() == after_first_spawn
    assert emit_event_mock.await_count == 1
    event = emit_event_mock.await_args.args[0]
    assert event.type == "spawn_rejected"
    assert event.data["reason"] == "duplicate_agent_id"
    assert event.data["agent_id"] == "child-1"
    sim.close()
