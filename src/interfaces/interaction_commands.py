from __future__ import annotations

from typing import TYPE_CHECKING

from src.interfaces.domain_command_adapters import command_from_payload
from src.interfaces.interaction_schema import (
    ENVELOPE_INTENTS,
    InteractionAuthScope,
    InteractionBudgetAttribution,
    InteractionContext,
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

    async def execute_from_payload(
        self,
        payload: dict[str, object],
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        command = command_from_payload(payload, context=context)
        return await self.execute(command, context=context)


__all__ = [
    "ENVELOPE_INTENTS",
    "InteractionAuthScope",
    "InteractionBudgetAttribution",
    "InteractionContext",
    "InteractionResult",
    "InteractionRouting",
    "InteractionService",
]
