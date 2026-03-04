from __future__ import annotations

import asyncio
from collections.abc import Iterable
from enum import Enum
from heapq import heappop, heappush
from typing import Any

from .version_vector import VersionVector
from .world.state import WorldState


class ResourceToken(str, Enum):
    WOOD = "wood"


class StructureType(str, Enum):
    HUT = "hut"


class WorldMap:
    """Grid map backed by WorldState.spatial/resources."""

    def __init__(
        self, width: int = 50, height: int = 50, *, world_state: WorldState | None = None
    ) -> None:
        self.lock = asyncio.Lock()
        self.vector = VersionVector()
        self.world_state = world_state or WorldState()
        self.world_state.spatial.width = width
        self.world_state.spatial.height = height

    @property
    def width(self) -> int:
        return self.world_state.spatial.width

    @width.setter
    def width(self, value: int) -> None:
        self.world_state.spatial.width = int(value)

    @property
    def height(self) -> int:
        return self.world_state.spatial.height

    @height.setter
    def height(self, value: int) -> None:
        self.world_state.spatial.height = int(value)

    @property
    def agent_positions(self) -> dict[str, tuple[int, int]]:
        return self.world_state.spatial.agent_positions

    @agent_positions.setter
    def agent_positions(self, value: dict[str, tuple[int, int]]) -> None:
        self.world_state.spatial.agent_positions = value

    @property
    def resources(self) -> dict[tuple[int, int], dict[str, int]]:
        return self.world_state.spatial.resources

    @resources.setter
    def resources(self, value: dict[tuple[int, int], dict[str, int]]) -> None:
        self.world_state.spatial.resources = value

    @property
    def buildings(self) -> dict[tuple[int, int], str]:
        return self.world_state.spatial.buildings

    @buildings.setter
    def buildings(self, value: dict[tuple[int, int], str]) -> None:
        self.world_state.spatial.buildings = value

    @property
    def agent_resources(self) -> dict[str, dict[str, int]]:
        return self.world_state.resources.agent_inventories

    @agent_resources.setter
    def agent_resources(self, value: dict[str, dict[str, int]]) -> None:
        self.world_state.resources.agent_inventories = value

    @property
    def obstacles(self) -> set[tuple[int, int]]:
        return self.world_state.spatial.obstacles

    @obstacles.setter
    def obstacles(self, value: set[tuple[int, int]]) -> None:
        self.world_state.spatial.obstacles = value

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, x: int, y: int) -> bool:
        return (x, y) not in self.obstacles

    def add_obstacle(self, x: int, y: int) -> None:
        if self.in_bounds(x, y):
            self.obstacles.add((x, y))

    async def add_agent(self, agent_id: str, x: int = 0, y: int = 0) -> None:
        async with self.lock:
            self.agent_positions[agent_id] = (x, y)
            self.agent_resources.setdefault(agent_id, {})

    async def remove_agent(self, agent_id: str) -> None:
        async with self.lock:
            self.agent_positions.pop(agent_id, None)
            self.agent_resources.pop(agent_id, None)

    async def add_resource(self, x: int, y: int, resource: ResourceToken, amount: int = 1) -> None:
        async with self.lock:
            cell = self.resources.setdefault((x, y), {})
            cell[resource.value] = cell.get(resource.value, 0) + amount
            self.world_state.resources.global_inventory[resource.value] = (
                self.world_state.resources.global_inventory.get(resource.value, 0) + amount
            )

    async def move(
        self, agent_id: str, dx: int, dy: int, *, vector: dict[str, int] | None = None
    ) -> tuple[int, int]:
        async with self.lock:
            x, y = self.agent_positions.get(agent_id, (0, 0))
            new_x = min(max(x + dx, 0), self.width - 1)
            new_y = min(max(y + dy, 0), self.height - 1)
            if not self.passable(new_x, new_y):
                return x, y
            self.agent_positions[agent_id] = (new_x, new_y)
            (
                self.vector.merge(VersionVector(vector))
                if vector is not None
                else self.vector.increment(agent_id)
            )
            return new_x, new_y

    async def move_to(
        self, agent_id: str, dest_x: int, dest_y: int, *, vector: dict[str, int] | None = None
    ) -> tuple[int, int]:
        async with self.lock:
            start = self.agent_positions.get(agent_id, (0, 0))
            path = self.find_path(start, (dest_x, dest_y))
            if len(path) < 2:
                return start
            nxt = path[1]
            if self.passable(*nxt):
                self.agent_positions[agent_id] = nxt
                (
                    self.vector.merge(VersionVector(vector))
                    if vector is not None
                    else self.vector.increment(agent_id)
                )
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
                    heappush(frontier, (new_cost + self.heuristic(nxt, goal), nxt))
                    came_from[nxt] = current
        if goal not in came_from:
            return [start]
        path: list[tuple[int, int]] = []
        curr: tuple[int, int] | None = goal
        while curr is not None:
            path.append(curr)
            curr = came_from.get(curr)
        return list(reversed(path))

    async def gather(
        self, agent_id: str, resource: ResourceToken, *, vector: dict[str, int] | None = None
    ) -> bool:
        async with self.lock:
            pos = self.agent_positions.get(agent_id)
            if pos is None:
                return False
            cell = self.resources.get(pos)
            key = resource.value
            if not cell or cell.get(key, 0) <= 0:
                return False
            cell[key] -= 1
            if cell[key] == 0:
                del cell[key]
            bag = self.agent_resources.setdefault(agent_id, {})
            bag[key] = bag.get(key, 0) + 1
            self.world_state.resources.global_inventory[key] = max(
                0, self.world_state.resources.global_inventory.get(key, 0) - 1
            )
            try:
                from src.infra.ledger import ledger

                ledger.add_tokens(agent_id, key, 1)
            except Exception:
                pass
            (
                self.vector.merge(VersionVector(vector))
                if vector is not None
                else self.vector.increment(agent_id)
            )
            return True

    async def build(
        self, agent_id: str, structure: StructureType, *, vector: dict[str, int] | None = None
    ) -> bool:
        async with self.lock:
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
            except Exception:
                pass
            (
                self.vector.merge(VersionVector(vector))
                if vector is not None
                else self.vector.increment(agent_id)
            )
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
            "vector": self.vector.to_dict(),
        }
