from __future__ import annotations

from typing import TYPE_CHECKING

from src.interfaces.interaction_schema import InteractionContext, InteractionResult
from src.sim.commands.domain_commands import DomainCommandT

if TYPE_CHECKING:
    from src.sim.command_service import SimulationCommandService


class SimulationCommandDispatcher:
    """Single command dispatcher for typed domain commands."""

    def __init__(self, command_service: SimulationCommandService) -> None:
        self.command_service = command_service

    async def dispatch(
        self,
        command: DomainCommandT,
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        return await self.command_service.execute(command, context=context)
