from __future__ import annotations

from enum import Enum
from typing import Any


class ResourceToken(str, Enum):
    """Tokens representing resources that agents can collect."""

    WOOD = "wood"


class StructureType(str, Enum):
    """Types of structures that agents can build."""

    HUT = "hut"


class WorldMap:
    """Simple grid-based world map for agent interactions."""

    def __init__(self, width: int = 10, height: int = 10) -> None:
        self.width = width
        self.height = height
        self.agent_positions: dict[str, tuple[int, int]] = {}
        self.resources: dict[tuple[int, int], dict[str, int]] = {}
        self.buildings: dict[tuple[int, int], str] = {}
        self.agent_resources: dict[str, dict[str, int]] = {}

    def add_agent(self, agent_id: str, x: int = 0, y: int = 0) -> None:
        """Place ``agent_id`` on the map and initialize its inventory."""

        self.agent_positions[agent_id] = (x, y)
        self.agent_resources.setdefault(agent_id, {})

    def add_resource(self, x: int, y: int, resource: ResourceToken, amount: int = 1) -> None:
        """Add ``amount`` of ``resource`` to the specified cell."""

        cell = self.resources.setdefault((x, y), {})
        cell[resource.value] = cell.get(resource.value, 0) + amount

    def move(self, agent_id: str, dx: int, dy: int) -> tuple[int, int]:
        """Move ``agent_id`` by ``dx`` and ``dy`` within map bounds."""

        x, y = self.agent_positions.get(agent_id, (0, 0))
        new_x = min(max(x + dx, 0), self.width - 1)
        new_y = min(max(y + dy, 0), self.height - 1)
        self.agent_positions[agent_id] = (new_x, new_y)
        return new_x, new_y

    def gather(self, agent_id: str, resource: ResourceToken) -> bool:
        """Collect ``resource`` from the agent's current position."""

        pos = self.agent_positions.get(agent_id)
        if pos is None:
            return False
        cell = self.resources.get(pos)
        res_key = resource.value
        if not cell or cell.get(res_key, 0) <= 0:
            return False
        cell[res_key] -= 1
        if cell[res_key] == 0:
            del cell[res_key]
        bag = self.agent_resources.setdefault(agent_id, {})
        bag[res_key] = bag.get(res_key, 0) + 1
        return True

    def build(self, agent_id: str, structure: StructureType) -> bool:
        """Construct a ``structure`` at the agent's current location."""

        pos = self.agent_positions.get(agent_id)
        if pos is None:
            return False
        bag = self.agent_resources.get(agent_id, {})
        wood = bag.get(ResourceToken.WOOD.value, 0)
        if wood < 1:
            return False
        bag[ResourceToken.WOOD.value] = wood - 1
        if bag[ResourceToken.WOOD.value] == 0:
            del bag[ResourceToken.WOOD.value]
        self.buildings[pos] = structure.value
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "agents": self.agent_positions,
            "resources": self.resources,
            "buildings": self.buildings,
            "agent_resources": self.agent_resources,
        }
