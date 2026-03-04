from __future__ import annotations

from typing import Any

from src.sim.contracts.tick_context import TickContext


class WorldEngine:
    """Supplies time/weather/season/spatial context for the tick."""

    def build_tick_context(self, simulation: Any) -> TickContext:
        world_projection = simulation._build_world_context_projection(actor_id=None)
        world_time = TickContext.freeze_mapping(simulation._world_time_from_projection(world_projection))
        governance_state = TickContext.freeze_mapping(
            {
                "council_window_active": simulation.environment_state.council_window_active,
                "world_day": simulation.environment_state.world_day,
            }
        )
        return TickContext(
            step=int(simulation.current_step),
            world_time=world_time,
            weather=str(simulation.environment_state.weather),
            governance_state=governance_state,
            world_modifiers=TickContext.freeze_mapping(
                simulation.environment_state.active_global_modifiers
            ),
            replay_metadata=TickContext.freeze_mapping(
                {
                    "trace_hash": getattr(simulation, "_last_trace_hash", ""),
                    "vector": simulation.vector.to_dict(),
                }
            ),
        )
