import sys

import pytest

from src.interfaces.metrics import ACTIVE_AGENT_COUNT

pytestmark = pytest.mark.unit


class DummyNeo4j:
    Driver = object
    GraphDatabase = object


class DummyAgentState:
    def __init__(self) -> None:
        self.ip = 0.0
        self.du = 0.0
        self.short_term_memory = []
        self.messages_sent_count = 0
        self.last_message_step = None
        self.collective_ip = 0.0
        self.collective_du = 0.0
        self.is_alive = True
        self.inheritance = 0.0
        self.parent_id: str | None = None
        self.genes: dict[str, float] = {}


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._state = DummyAgentState()

    def get_id(self) -> str:
        return self.agent_id

    @property
    def state(self) -> DummyAgentState:
        return self._state

    def update_state(self, state: DummyAgentState) -> None:
        self._state = state


@pytest.mark.asyncio
async def test_active_agent_count_gauge_updates() -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.sim.simulation import Simulation

    agent_a = DummyAgent("A")
    sim = Simulation([agent_a])

    assert ACTIVE_AGENT_COUNT._value.get() == 1

    agent_b = DummyAgent("B")
    await sim.spawn_agent(agent_b)
    assert ACTIVE_AGENT_COUNT._value.get() == 2

    await sim.handle_control_command({"command": "kill_agent", "agent_id": "A"})
    assert ACTIVE_AGENT_COUNT._value.get() == 1

    await sim.handle_control_command({"command": "kill_agent", "agent_id": "B"})
    assert ACTIVE_AGENT_COUNT._value.get() == 0

    sim.close()


@pytest.mark.asyncio
async def test_retire_and_kill_remove_agent_from_world_map_state() -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.sim.simulation import Simulation

    agent_a = DummyAgent("A")
    agent_b = DummyAgent("B")
    sim = Simulation([agent_a, agent_b])

    await sim.retire_agent(agent_a)
    assert agent_a in sim.agents
    assert "A" not in sim.world_map.agent_positions
    assert "A" not in sim.world_map.agent_resources

    await sim.handle_control_command({"command": "kill_agent", "agent_id": "B"})
    assert all(agent.agent_id != "B" for agent in sim.agents)
    assert "B" not in sim.world_map.agent_positions
    assert "B" not in sim.world_map.agent_resources

    sim.close()
