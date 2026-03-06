from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from src.interfaces.interaction_schema import (
    BaseInteractionEnvelope,
    InteractionContext,
    InteractionEnvelope,
    InteractionResult,
)

if TYPE_CHECKING:
    from src.interfaces.interaction_commands import InteractionService
    from src.sim.commands.domain_commands import DomainCommandT


class CommandBus:
    """Thin adapter that converts external payloads into typed domain commands."""

    def __init__(self, interaction_service: InteractionService) -> None:
        self.interaction_service = interaction_service

    async def dispatch(
        self,
        command: DomainCommandT | InteractionEnvelope,
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        dispatcher = self.interaction_service.simulation.command_dispatcher
        if isinstance(command, BaseInteractionEnvelope):
            return await dispatcher.dispatch_envelope(command, context=context)
        return await dispatcher.dispatch(command, context=context)

    async def dispatch_payload(
        self,
        payload: Mapping[str, Any],
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        dispatcher = self.interaction_service.simulation.command_dispatcher
        return await dispatcher.dispatch_payload(payload, context=context)
