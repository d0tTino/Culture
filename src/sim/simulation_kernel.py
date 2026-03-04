from __future__ import annotations

from typing import Any

from src.sim.engines.interaction_engine import InteractionEngine
from src.sim.engines.persistence_engine import PersistenceEngine
from src.sim.engines.reducers import DomainEventReducer
from src.sim.engines.society_engine import SocietyEngine
from src.sim.engines.turn_engine import TurnEngine
from src.sim.engines.world_engine import WorldEngine
from src.sim.runtime.step_context import StepContext


class SimulationKernel:
    """Thin lifecycle kernel that orchestrates phase ordering across isolated engines."""

    def __init__(self) -> None:
        self.turn_engine = TurnEngine()
        self.interaction_engine = InteractionEngine()
        self.world_engine = WorldEngine()
        self.society_engine = SocietyEngine()
        self.persistence_engine = PersistenceEngine()
        self.reducer = DomainEventReducer()

    async def run_tick(self, simulation: Any, context: StepContext) -> int:
        tick = self.world_engine.build_tick_context(simulation)
        context.tick_context = tick

        context.phase_order.append("perception")
        self.reducer.apply(simulation, context, await self.interaction_engine.ingress(simulation, tick))

        context.phase_order.append("decision")
        self.reducer.apply(simulation, context, await self.turn_engine.plan(simulation, context, tick))

        context.phase_order.append("action")
        self.reducer.apply(simulation, context, await self.turn_engine.commit(simulation, context, tick))

        context.phase_order.append("post_step")
        _ = self.society_engine.snapshot(simulation, tick)
        if not context.planned_outputs:
            await simulation.engine.emit_evaluation_events(context.events)

        if context.planned_outputs:
            return len(context.planned_outputs)
        return len(context.events)
