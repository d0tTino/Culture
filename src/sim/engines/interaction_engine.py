from __future__ import annotations

from typing import Any

from src.sim.contracts.tick_context import TickContext
from src.sim.engines.domain_events import DomainEvent


class InteractionEngine:
    """Handles human/bot command ingress boundaries for each tick."""

    async def ingress(self, simulation: Any, tick: TickContext) -> list[DomainEvent]:
        await simulation.start_event_listener()
        return [
            DomainEvent(
                domain="interaction",
                name="queue_depth_observed",
                payload={"queue_depth": simulation.event_kernel.queue_depth(), "tick_step": tick.step},
            )
        ]
