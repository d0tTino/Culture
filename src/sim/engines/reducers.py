from __future__ import annotations

from typing import Any

from src.interfaces.metrics import STEP_PHASE_QUEUE_DEPTH
from src.sim.engines.domain_events import DomainEvent
from src.sim.runtime.step_context import StepContext


class DomainEventReducer:
    """Applies explicit domain events to simulation runtime state."""

    def apply(self, simulation: Any, context: StepContext, events: list[DomainEvent]) -> None:
        for event in events:
            if event.name == "queue_depth_observed":
                context.queue_depth = int(event.payload["queue_depth"])
                simulation._set_labeled_gauge(
                    STEP_PHASE_QUEUE_DEPTH,
                    phase="queue_pre_step",
                    value=context.queue_depth,
                )
            elif event.name == "planned_turns_ready":
                context.planned_outputs = list(event.payload.get("planned_outputs", []))
            elif event.name == "planned_turns_committed":
                context.planned_outputs = list(event.payload.get("committed_outputs", []))
                turn_count = int(event.payload.get("turn_count", len(context.planned_outputs)))
                if simulation.agents:
                    simulation.current_step += turn_count
                    simulation.current_agent_index = (
                        simulation.current_agent_index + turn_count
                    ) % len(simulation.agents)
                simulation.total_turns_executed += int(event.payload.get("accepted_turn_count", 0))
            elif event.name == "scheduler_events_ready":
                context.events = list(event.payload.get("events", []))
                simulation.total_turns_executed += len(context.events)
            elif event.name == "bootstrap_agent_event_requested":
                agent_index = int(event.payload["agent_index"])
                agent_id = str(event.payload["agent_id"])
                simulation.vector.increment(agent_id)
                simulation.event_kernel.schedule_immediate_nowait(
                    simulation._create_agent_event(agent_index),
                    agent_id=agent_id,
                    vector=simulation.vector,
                )
