from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from heapq import heappop, heappush
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
        self.obstacles: set[tuple[int, int]] = set()

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, x: int, y: int) -> bool:
        return (x, y) not in self.obstacles

    def add_obstacle(self, x: int, y: int) -> None:
        if self.in_bounds(x, y):
            self.obstacles.add((x, y))

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
        if not self.passable(new_x, new_y):
            return x, y
        self.agent_positions[agent_id] = (new_x, new_y)
        return new_x, new_y

    def move_to(self, agent_id: str, dest_x: int, dest_y: int) -> tuple[int, int]:
        """Move ``agent_id`` one step toward ``dest_x``, ``dest_y`` using A*."""

        start = self.agent_positions.get(agent_id, (0, 0))
        path = self.find_path(start, (dest_x, dest_y))
        if len(path) < 2:
            return start
        nxt = path[1]
        if self.passable(*nxt):
            self.agent_positions[agent_id] = nxt
            return nxt
        return start

    def neighbors(self, pos: tuple[int, int]) -> Iterable[tuple[int, int]]:
        x, y = pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny) and self.passable(nx, ny):
                yield (nx, ny)

    def heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """Return a path from ``start`` to ``goal`` using A* search."""

        if not self.in_bounds(*goal) or not self.passable(*goal):
            return [start]

        frontier: list[tuple[int, tuple[int, int]]] = []
        heappush(frontier, (0, start))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far: dict[tuple[int, int], int] = {start: 0}

        while frontier:
            _, current = heappop(frontier)

            if current == goal:
                break

            for nxt in self.neighbors(current):
                new_cost = cost_so_far[current] + 1
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self.heuristic(nxt, goal)
                    heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

        if goal not in came_from:
            return [start]

        path: list[tuple[int, int]] = []
        curr: tuple[int, int] | None = goal
        while curr is not None:
            path.append(curr)
            curr = came_from.get(curr)
        path.reverse()
        return path

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
        try:
            from src.infra.ledger import ledger

            ledger.add_tokens(agent_id, res_key, 1)
        except Exception:  # pragma: no cover - optional
            pass
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
        try:
            from src.infra.ledger import ledger

            ledger.remove_tokens(agent_id, ResourceToken.WOOD.value, 1)
        except Exception:  # pragma: no cover - optional
            pass
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "agents": self.agent_positions,
            "resources": self.resources,
            "buildings": self.buildings,
            "agent_resources": self.agent_resources,
            "obstacles": list(self.obstacles),
        }
