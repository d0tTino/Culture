from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemporalWorldState:
    world_tick: int = 0
    world_hour: int = 0
    world_day: int = 0
    world_season: int | None = None


@dataclass
class EnvironmentWorldState:
    weather: str = "clear"
    season_effects: dict[str, Any] = field(default_factory=dict)
    active_global_modifiers: list[str] = field(default_factory=list)
    council_window_active: bool = False


@dataclass
class SpatialWorldState:
    width: int = 50
    height: int = 50
    agent_positions: dict[str, tuple[int, int]] = field(default_factory=dict)
    resources: dict[tuple[int, int], dict[str, int]] = field(default_factory=dict)
    buildings: dict[tuple[int, int], str] = field(default_factory=dict)
    obstacles: set[tuple[int, int]] = field(default_factory=set)


@dataclass
class ResourceWorldState:
    global_inventory: dict[str, int] = field(default_factory=dict)
    agent_inventories: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class WorldState:
    temporal: TemporalWorldState = field(default_factory=TemporalWorldState)
    environment: EnvironmentWorldState = field(default_factory=EnvironmentWorldState)
    spatial: SpatialWorldState = field(default_factory=SpatialWorldState)
    resources: ResourceWorldState = field(default_factory=ResourceWorldState)

    def snapshot(self) -> dict[str, Any]:
        return {
            "temporal": {
                "world_tick": self.temporal.world_tick,
                "world_hour": self.temporal.world_hour,
                "world_day": self.temporal.world_day,
                "world_season": self.temporal.world_season,
            },
            "environment": {
                "weather": self.environment.weather,
                "season_effects": dict(self.environment.season_effects),
                "active_global_modifiers": list(self.environment.active_global_modifiers),
                "council_window_active": self.environment.council_window_active,
            },
            "spatial": {
                "width": self.spatial.width,
                "height": self.spatial.height,
                "agent_positions": {
                    aid: [int(pos[0]), int(pos[1])]
                    for aid, pos in sorted(self.spatial.agent_positions.items())
                },
                "resources": {
                    f"{x},{y}": dict(values)
                    for (x, y), values in sorted(self.spatial.resources.items())
                },
                "buildings": {
                    f"{x},{y}": b for (x, y), b in sorted(self.spatial.buildings.items())
                },
                "obstacles": [list(item) for item in sorted(self.spatial.obstacles)],
            },
            "resources": {
                "global_inventory": dict(self.resources.global_inventory),
                "agent_inventories": {
                    aid: dict(values)
                    for aid, values in sorted(self.resources.agent_inventories.items())
                },
            },
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> WorldState:
        temporal = payload.get("temporal", {})
        env = payload.get("environment", {})
        spatial = payload.get("spatial", {})
        resources = payload.get("resources", {})

        spatial_resources: dict[tuple[int, int], dict[str, int]] = {}
        for key, val in (spatial.get("resources", {}) or {}).items():
            x_str, y_str = str(key).split(",", 1)
            spatial_resources[(int(x_str), int(y_str))] = dict(val)

        buildings: dict[tuple[int, int], str] = {}
        for key, building in (spatial.get("buildings", {}) or {}).items():
            x_str, y_str = str(key).split(",", 1)
            buildings[(int(x_str), int(y_str))] = str(building)

        return cls(
            temporal=TemporalWorldState(
                world_tick=int(temporal.get("world_tick", 0)),
                world_hour=int(temporal.get("world_hour", 0)),
                world_day=int(temporal.get("world_day", 0)),
                world_season=(
                    int(temporal["world_season"])
                    if temporal.get("world_season") is not None
                    else None
                ),
            ),
            environment=EnvironmentWorldState(
                weather=str(env.get("weather", "clear")),
                season_effects=dict(env.get("season_effects", {})),
                active_global_modifiers=list(env.get("active_global_modifiers", [])),
                council_window_active=bool(env.get("council_window_active", False)),
            ),
            spatial=SpatialWorldState(
                width=int(spatial.get("width", 50)),
                height=int(spatial.get("height", 50)),
                agent_positions={
                    aid: (int(value[0]), int(value[1]))
                    for aid, value in (spatial.get("agent_positions", {}) or {}).items()
                },
                resources=spatial_resources,
                buildings=buildings,
                obstacles={
                    (int(item[0]), int(item[1])) for item in (spatial.get("obstacles", []) or [])
                },
            ),
            resources=ResourceWorldState(
                global_inventory=dict(resources.get("global_inventory", {})),
                agent_inventories={
                    aid: dict(value)
                    for aid, value in (resources.get("agent_inventories", {}) or {}).items()
                },
            ),
        )
