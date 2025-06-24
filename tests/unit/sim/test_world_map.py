import asyncio
import sys
from pathlib import Path

import pytest

from src.sim.world_map import ResourceToken, StructureType, WorldMap

pytestmark = pytest.mark.unit


def test_move_within_bounds() -> None:
    m = WorldMap(width=3, height=3)
    m.add_agent("A")
    m.move("A", 1, 1)
    assert m.agent_positions["A"] == (1, 1)


def test_gather_resource() -> None:
    m = WorldMap()
    m.add_agent("A")
    m.add_resource(0, 0, ResourceToken.WOOD, 1)
    assert m.gather("A", ResourceToken.WOOD)
    assert m.agent_resources["A"]["wood"] == 1
    assert m.resources[(0, 0)].get("wood", 0) == 0


def test_build_structure() -> None:
    m = WorldMap()
    m.add_agent("A")
    m.agent_resources["A"] = {"wood": 1}
    assert m.build("A", StructureType.HUT)
    assert m.buildings[(0, 0)] == StructureType.HUT.value
    assert m.agent_resources["A"].get("wood", 0) == 0


class DummyNeo4j:
    Driver = object
    GraphDatabase = object


class DummyState:
    def __init__(self) -> None:
        self.ip: float = 0.0
        self.du: float = 0.0
        self.short_term_memory: list[object] = []
        self.messages_sent_count: int = 0
        self.last_message_step: int | None = None


class ActionAgent:
    def __init__(self, action: dict[str, object], agent_id: str = "A") -> None:
        self.agent_id = agent_id
        self.state = DummyState()
        self._action = action

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, state: DummyState) -> None:
        self.state = state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict[str, object] | None = None,
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
    ) -> dict[str, object]:
        return {"map_action": self._action}


def test_move_updates_balance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.infra import config
    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)

    agent = ActionAgent({"action": "move", "dx": 1, "dy": 0})
    sim = Simulation([agent])  # type: ignore[arg-type,list-item]

    asyncio.run(sim.run_step())

    ip, du = ledger.get_balance(agent.agent_id)
    assert ip == pytest.approx(config.MAP_MOVE_IP_REWARD - config.MAP_MOVE_IP_COST)
    assert du == pytest.approx(config.MAP_MOVE_DU_REWARD - config.MAP_MOVE_DU_COST)
    assert sim.world_map.agent_positions[agent.agent_id] == (1, 0)


def test_gather_updates_balance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.infra import config
    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)

    agent = ActionAgent({"action": "gather", "resource": ResourceToken.WOOD.value})
    sim = Simulation([agent])  # type: ignore[arg-type,list-item]
    sim.world_map.add_resource(0, 0, ResourceToken.WOOD, 1)

    asyncio.run(sim.run_step())

    ip, du = ledger.get_balance(agent.agent_id)
    assert ip == pytest.approx(config.MAP_GATHER_IP_REWARD - config.MAP_GATHER_IP_COST)
    assert du == pytest.approx(config.MAP_GATHER_DU_REWARD - config.MAP_GATHER_DU_COST)
    assert sim.world_map.agent_resources[agent.agent_id].get("wood", 0) == 1


def test_build_updates_balance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.infra import config
    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)

    agent = ActionAgent({"action": "build", "structure": StructureType.HUT.value})
    sim = Simulation([agent])  # type: ignore[arg-type,list-item]
    sim.world_map.agent_resources[agent.agent_id] = {"wood": 1}

    asyncio.run(sim.run_step())

    ip, du = ledger.get_balance(agent.agent_id)
    assert ip == pytest.approx(config.MAP_BUILD_IP_REWARD - config.MAP_BUILD_IP_COST)
    assert du == pytest.approx(config.MAP_BUILD_DU_REWARD - config.MAP_BUILD_DU_COST)
    assert sim.world_map.buildings[(0, 0)] == StructureType.HUT.value
