from __future__ import annotations

from typing import Any

from src.sim.world.state import WorldState


class WorldPerceptionBuilder:
    def build(
        self,
        *,
        world_state: WorldState,
        world_projection: Any,
        effect_hooks: dict[str, Any],
        perceived_messages: list[Any],
        knowledge_board_content: list[str] | None,
    ) -> dict[str, Any]:
        return {
            "perceived_messages": list(perceived_messages),
            "knowledge_board_content": list(knowledge_board_content or []),
            "environment_context": {
                "turn_index": world_projection.turn_index,
                "time": {
                    "world_tick": world_projection.time.world_tick,
                    "world_hour": world_projection.time.world_hour,
                    "world_day": world_projection.time.world_day,
                    "world_season": world_projection.time.world_season,
                    "formatted": world_projection.time.formatted,
                },
                "weather": world_projection.weather,
                "season": world_projection.season,
                "season_effects": dict(world_state.environment.season_effects),
                "active_global_modifiers": list(world_state.environment.active_global_modifiers),
                "council_window_active": world_projection.council.council_window_active,
                "governance_phase": world_projection.council.phase,
                "map_neighborhood": {
                    "actor_id": world_projection.map_neighborhood.actor_id,
                    "location": list(world_projection.map_neighborhood.location),
                    "neighboring_cells": [
                        list(cell) for cell in world_projection.map_neighborhood.neighboring_cells
                    ],
                    "nearby_agents": list(world_projection.map_neighborhood.nearby_agents),
                },
                "effect_hooks": effect_hooks,
            },
        }
