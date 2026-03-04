from __future__ import annotations

from typing import Any

from src.sim.world.state import WorldState


class WorldPerceptionBuilder:
    def build(
        self,
        *,
        world_state: WorldState,
        actor_id: str,
        turn_index: int,
        effect_hooks: dict[str, Any],
        perceived_messages: list[Any],
        knowledge_board_content: list[str] | None,
    ) -> dict[str, Any]:
        location = world_state.spatial.agent_positions.get(actor_id, (0, 0))
        nearby_agents = sorted(
            other_id
            for other_id, other_location in world_state.spatial.agent_positions.items()
            if other_id != actor_id and other_location == location
        )

        neighboring_cells: list[list[int]] = []
        x, y = location
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < world_state.spatial.width and 0 <= ny < world_state.spatial.height:
                if (nx, ny) not in world_state.spatial.obstacles:
                    neighboring_cells.append([nx, ny])

        season_name = None
        if world_state.temporal.world_season is not None:
            names = ("spring", "summer", "autumn", "winter")
            season_name = names[world_state.temporal.world_season % len(names)]

        return {
            "perceived_messages": list(perceived_messages),
            "knowledge_board_content": list(knowledge_board_content or []),
            "environment_context": {
                "turn_index": turn_index,
                "time": {
                    "world_tick": world_state.temporal.world_tick,
                    "world_hour": world_state.temporal.world_hour,
                    "world_day": world_state.temporal.world_day,
                    "world_season": world_state.temporal.world_season,
                    "formatted": (
                        f"Day {world_state.temporal.world_day}, "
                        f"{world_state.temporal.world_hour:02d}:00"
                    ),
                },
                "weather": world_state.environment.weather,
                "season": season_name,
                "season_effects": dict(world_state.environment.season_effects),
                "active_global_modifiers": list(world_state.environment.active_global_modifiers),
                "council_window_active": world_state.environment.council_window_active,
                "governance_phase": (
                    "council_window"
                    if world_state.environment.council_window_active
                    else "standard"
                ),
                "map_neighborhood": {
                    "actor_id": actor_id,
                    "location": [x, y],
                    "neighboring_cells": neighboring_cells,
                    "nearby_agents": nearby_agents,
                },
                "effect_hooks": effect_hooks,
            },
        }
