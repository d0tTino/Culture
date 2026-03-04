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
            elif event.name == "scheduler_events_ready":
                context.events = list(event.payload.get("events", []))
