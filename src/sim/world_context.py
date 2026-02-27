from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.sim.environment import EnvironmentSystem
from src.sim.world_map import WorldMap

WORLD_CONTEXT_PROJECTION_VERSION = 1


@dataclass(frozen=True)
class WorldTimeContext:
    world_tick: int
    world_hour: int
    world_day: int
    world_season: int | None
    formatted: str


@dataclass(frozen=True)
class GovernanceContext:
    council_window_active: bool
    phase: str


@dataclass(frozen=True)
class MapNeighborhoodContext:
    actor_id: str
    location: tuple[int, int]
    neighboring_cells: list[tuple[int, int]]
    nearby_agents: list[str]


@dataclass(frozen=True)
class WorldContextProjection:
    projection_version: int
    turn_index: int
    time: WorldTimeContext
    weather: str
    season: str | None
    council: GovernanceContext
    map_neighborhood: MapNeighborhoodContext

    @classmethod
    def build(
        cls,
        *,
        turn_index: int,
        actor_id: str,
        environment_system: EnvironmentSystem,
        world_map: WorldMap,
    ) -> WorldContextProjection:
        world_time = environment_system.world_time_snapshot()
        location = world_map.agent_positions.get(actor_id, (0, 0))
        nearby_agents = sorted(
            other_id
            for other_id, other_location in world_map.agent_positions.items()
            if other_id != actor_id and other_location == location
        )
        governance_phase = (
            "council_window" if environment_system.state.council_window_active else "standard"
        )
        return cls(
            projection_version=WORLD_CONTEXT_PROJECTION_VERSION,
            turn_index=turn_index,
            time=WorldTimeContext(
                world_tick=int(world_time["world_tick"]),
                world_hour=int(world_time["world_hour"]),
                world_day=int(world_time["world_day"]),
                world_season=(
                    int(world_time["world_season"])
                    if world_time.get("world_season") is not None
                    else None
                ),
                formatted=str(world_time["formatted"]),
            ),
            weather=str(environment_system.state.weather),
            season=environment_system._season_name(),
            council=GovernanceContext(
                council_window_active=bool(environment_system.state.council_window_active),
                phase=governance_phase,
            ),
            map_neighborhood=MapNeighborhoodContext(
                actor_id=actor_id,
                location=(int(location[0]), int(location[1])),
                neighboring_cells=[(int(x), int(y)) for x, y in world_map.neighbors(location)],
                nearby_agents=nearby_agents,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_version": self.projection_version,
            "turn_index": self.turn_index,
            "time": {
                "world_tick": self.time.world_tick,
                "world_hour": self.time.world_hour,
                "world_day": self.time.world_day,
                "world_season": self.time.world_season,
                "formatted": self.time.formatted,
            },
            "weather": self.weather,
            "season": self.season,
            "council": {
                "council_window_active": self.council.council_window_active,
                "phase": self.council.phase,
            },
            "map_neighborhood": {
                "actor_id": self.map_neighborhood.actor_id,
                "location": list(self.map_neighborhood.location),
                "neighboring_cells": [list(cell) for cell in self.map_neighborhood.neighboring_cells],
                "nearby_agents": list(self.map_neighborhood.nearby_agents),
            },
        }

    def to_environment_context(self, *, effect_hooks: dict[str, Any]) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "time": {
                "world_tick": self.time.world_tick,
                "world_hour": self.time.world_hour,
                "world_day": self.time.world_day,
                "world_season": self.time.world_season,
                "formatted": self.time.formatted,
            },
            "weather": self.weather,
            "season": self.season,
            "council_window_active": self.council.council_window_active,
            "governance_phase": self.council.phase,
            "map_neighborhood": {
                "actor_id": self.map_neighborhood.actor_id,
                "location": list(self.map_neighborhood.location),
                "neighboring_cells": [list(cell) for cell in self.map_neighborhood.neighboring_cells],
                "nearby_agents": list(self.map_neighborhood.nearby_agents),
            },
            "effect_hooks": effect_hooks,
        }
