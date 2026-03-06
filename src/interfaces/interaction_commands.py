from __future__ import annotations

from typing import TYPE_CHECKING

from src.interfaces.interaction_schema import (
    ENVELOPE_INTENTS,
    InteractionAuthScope,
    InteractionBudgetAttribution,
    InteractionContext,
    InteractionIntent,
    InteractionResult,
    InteractionRouting,
)

if TYPE_CHECKING:
    from src.sim.commands.domain_commands import DomainCommandT
    from src.sim.simulation import Simulation


class InteractionService:
    """Thin transport wrapper around SimulationCommandService."""

    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation

    async def execute(
        self,
        command: DomainCommandT,
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        return await self.simulation.command_dispatcher.dispatch(command, context=context)

    async def execute_intent(
        self,
        intent: InteractionIntent,
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        return await self.simulation.command_dispatcher.dispatch_envelope(intent, context=context)

    async def execute_from_payload(
        self,
        payload: dict[str, object],
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        return await self.simulation.command_dispatcher.dispatch_payload(payload, context=context)


__all__ = [
    "ENVELOPE_INTENTS",
    "InteractionAuthScope",
    "InteractionBudgetAttribution",
    "InteractionContext",
    "InteractionIntent",
    "InteractionResult",
    "InteractionRouting",
    "InteractionService",
]
