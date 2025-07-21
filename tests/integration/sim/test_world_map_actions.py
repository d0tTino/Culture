import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest


class DummyNeo4j:
    Driver = object
    GraphDatabase = object


sys.modules.setdefault("neo4j", DummyNeo4j())

from src.infra.ledger import Ledger
from src.sim.simulation import Simulation
from src.sim.world_map import ResourceToken, StructureType
from src.sim.world_map_actions import process_map_action


class DummyState(SimpleNamespace):
    ip: float = 0.0
    du: float = 0.0
    short_term_memory: ClassVar[list[Any]] = []
    messages_sent_count: int = 0
    last_message_step: int = 0
    relationships: ClassVar[dict[str, Any]] = {}
    role: str = "dummy"
    steps_in_current_role: int = 0

    def update_collective_metrics(self, ip: float, du: float) -> None:
        pass


class DummyAgent:
    def __init__(self, agent_id: str = "agent") -> None:
        self.agent_id = agent_id
        self.state = DummyState()

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, new_state: DummyState) -> None:
        self.state = new_state


@pytest.fixture()
def sim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Simulation, Ledger, DummyAgent]:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    agent = DummyAgent()
    sim = Simulation(agents=[agent])
    yield sim, ledger, agent
    sim.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_move_action_updates_position_and_ledger(
    sim: tuple[Simulation, Ledger, DummyAgent],
) -> None:
    sim_obj, ledger, agent = sim
    await process_map_action(
        sim_obj, 0, agent.agent_id, agent.state, {"action": "move", "dx": 1, "dy": 0}
    )

    assert sim_obj.world_map.agent_positions[agent.agent_id] == (1, 0)
    rows = ledger.conn.execute("SELECT reason FROM transactions").fetchall()
    assert rows == [("move",)]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_move_blocked_by_obstacle(sim: tuple[Simulation, Ledger, DummyAgent]) -> None:
    sim_obj, ledger, agent = sim
    async with sim_obj.world_map.lock:
        sim_obj.world_map.add_obstacle(1, 0)
    await process_map_action(
        sim_obj, 0, agent.agent_id, agent.state, {"action": "move", "dx": 1, "dy": 0}
    )

    assert sim_obj.world_map.agent_positions[agent.agent_id] == (0, 0)
    rows = ledger.conn.execute("SELECT reason FROM transactions").fetchall()
    assert rows == [("move",)]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gather_action_success_and_failure(
    sim: tuple[Simulation, Ledger, DummyAgent],
) -> None:
    sim_obj, ledger, agent = sim
    await sim_obj.world_map.add_resource(0, 0, ResourceToken.WOOD, 1)
    await process_map_action(
        sim_obj,
        0,
        agent.agent_id,
        agent.state,
        {"action": "gather", "resource": ResourceToken.WOOD.value},
    )

    assert sim_obj.world_map.agent_resources[agent.agent_id].get("wood", 0) == 1
    rows = ledger.conn.execute("SELECT reason FROM transactions").fetchall()
    assert rows == [("gather",)]

    await process_map_action(
        sim_obj,
        0,
        agent.agent_id,
        agent.state,
        {"action": "gather", "resource": ResourceToken.WOOD.value},
    )

    assert sim_obj.world_map.agent_resources[agent.agent_id].get("wood", 0) == 1
    rows = ledger.conn.execute("SELECT reason FROM transactions").fetchall()
    assert rows == [("gather",)]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_build_action_success_and_failure(
    sim: tuple[Simulation, Ledger, DummyAgent],
) -> None:
    sim_obj, ledger, agent = sim
    await process_map_action(
        sim_obj,
        0,
        agent.agent_id,
        agent.state,
        {"action": "build", "structure": StructureType.HUT.value},
    )

    assert not sim_obj.world_map.buildings
    rows = ledger.conn.execute("SELECT reason FROM transactions").fetchall()
    assert rows == []

    sim_obj.world_map.agent_resources[agent.agent_id] = {"wood": 1}
    await process_map_action(
        sim_obj,
        0,
        agent.agent_id,
        agent.state,
        {"action": "build", "structure": StructureType.HUT.value},
    )

    assert sim_obj.world_map.buildings[(0, 0)] == StructureType.HUT.value
    rows = ledger.conn.execute("SELECT reason FROM transactions").fetchall()
    assert rows == [("build",)]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gather_then_build_updates_ledger_and_map(
    sim: tuple[Simulation, Ledger, DummyAgent],
) -> None:
    sim_obj, ledger, agent = sim
    await sim_obj.world_map.add_resource(0, 0, ResourceToken.WOOD, 1)
    await process_map_action(
        sim_obj,
        0,
        agent.agent_id,
        agent.state,
        {"action": "gather", "resource": ResourceToken.WOOD.value},
    )

    assert ledger.get_tokens(agent.agent_id, "wood") == 1

    await process_map_action(
        sim_obj,
        0,
        agent.agent_id,
        agent.state,
        {"action": "build", "structure": StructureType.HUT.value},
    )

    assert sim_obj.world_map.buildings[(0, 0)] == StructureType.HUT.value
    assert ledger.get_tokens(agent.agent_id, "wood") == 0
    rows = ledger.conn.execute("SELECT reason FROM transactions").fetchall()
    assert rows == [("gather",), ("build",)]
