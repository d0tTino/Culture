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
