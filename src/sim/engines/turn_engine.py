from __future__ import annotations

from typing import Any

from src.shared.telemetry import trace_agent_action
from src.sim.contracts.tick_context import TickContext
from src.sim.engines.domain_events import DomainEvent
from src.sim.runtime.step_context import StepContext


class TurnEngine:
    """Coordinates agent planning and deterministic commit boundaries."""

    async def plan(self, simulation: Any, context: StepContext, tick: TickContext) -> list[DomainEvent]:
        if context.max_turns <= 1:
            return []
        planned = await simulation._run_step_pipeline(max_turns=context.max_turns)
        return [
            DomainEvent(
                domain="turn",
                name="planned_turns_ready",
                payload={"planned_outputs": planned, "tick_step": tick.step},
            )
        ]

    async def commit(
        self, simulation: Any, context: StepContext, tick: TickContext
    ) -> list[DomainEvent]:
        if context.planned_outputs:
            return []
        if simulation.event_kernel.empty():
            simulation.vector.increment(simulation.agents[simulation.current_agent_index].get_id())
            simulation.event_kernel.schedule_immediate_nowait(
                simulation._create_agent_event(simulation.current_agent_index),
                agent_id=simulation.agents[simulation.current_agent_index].get_id(),
                vector=simulation.vector,
            )

        agent_id = simulation.agents[simulation.current_agent_index].get_id()
        with trace_agent_action("tick", agent_id=agent_id, step=tick.step):
            events = await simulation.event_kernel.step(context.max_turns)
        return [
            DomainEvent(
                domain="turn",
                name="scheduler_events_ready",
                payload={"events": events},
            )
        ]
