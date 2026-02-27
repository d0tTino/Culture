from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from src.interfaces.interaction_schema import (
    ENVELOPE_INTENTS,
    InteractionAuthScope,
    InteractionBudgetAttribution,
    InteractionContext,
    InteractionEnvelope,
    InteractionResult,
    InteractionRouting,
)

if TYPE_CHECKING:
    from src.sim.simulation import Simulation


class InteractionService:
    """Thin transport wrapper around SimulationCommandService."""

    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation

    async def execute(
        self,
        command: InteractionEnvelope,
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        return await self.simulation.command_service.execute(command, context=context)

    async def execute_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        return await self.simulation.command_service.execute_from_payload(payload, context=context)


__all__ = [
    "ENVELOPE_INTENTS",
    "InteractionAuthScope",
    "InteractionBudgetAttribution",
    "InteractionContext",
    "InteractionEnvelope",
    "InteractionResult",
    "InteractionRouting",
    "InteractionService",
]
