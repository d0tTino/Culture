from unittest.mock import MagicMock

import pytest

from src.sim.world_map import ResourceToken, StructureType, WorldMap

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def ledger_stub(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the global ledger object used by ``WorldMap``."""
    import src.infra.ledger as ledger_module

    mock = MagicMock()
    monkeypatch.setattr(ledger_module, "ledger", mock)
    return mock


def test_move_updates_vector_and_position() -> None:
    m = WorldMap(width=3, height=3)
    m.add_agent("A")

    pos = m.move("A", 1, 1)

    assert pos == (1, 1)
    assert m.agent_positions["A"] == (1, 1)
    assert m.vector.to_dict() == {"A": 1}
    state = m.to_dict()
    assert state["vector"] == {"A": 1}
    assert state["agents"] == {"A": (1, 1)}


def test_gather_updates_resources_and_vector(ledger_stub: MagicMock) -> None:
    m = WorldMap()
    m.add_agent("A")
    m.add_resource(0, 0, ResourceToken.WOOD, 1)

    result = m.gather("A", ResourceToken.WOOD)

    assert result is True
    assert m.agent_resources["A"].get("wood", 0) == 1
    assert m.resources[(0, 0)].get("wood", 0) == 0
    ledger_stub.add_tokens.assert_called_once_with("A", "wood", 1)
    assert m.vector.to_dict() == {"A": 1}
    state = m.to_dict()
    assert state["vector"] == {"A": 1}
    assert state["agent_resources"]["A"]["wood"] == 1
    assert state["resources"][(0, 0)].get("wood", 0) == 0


def test_build_updates_buildings_and_vector(ledger_stub: MagicMock) -> None:
    m = WorldMap()
    m.add_agent("A")
    m.agent_resources["A"] = {"wood": 1}

    result = m.build("A", StructureType.HUT)

    assert result is True
    assert m.buildings[(0, 0)] == StructureType.HUT.value
    assert m.agent_resources["A"].get("wood") is None
    ledger_stub.remove_tokens.assert_called_once_with("A", "wood", 1)
    assert m.vector.to_dict() == {"A": 1}
    state = m.to_dict()
    assert state["vector"] == {"A": 1}
    assert state["buildings"] == {(0, 0): StructureType.HUT.value}
