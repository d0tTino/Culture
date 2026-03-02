from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from src.governance.decision_kernel import DecisionProvenance
from src.interfaces.domain_command_adapters import command_from_envelope, command_from_payload
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
        normalized = command_from_envelope(command) if isinstance(command, BaseInteractionEnvelope) else command
        return await self.interaction_service.execute(normalized, context=context)

    async def dispatch_payload(
        self,
        payload: Mapping[str, Any],
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        try:
            command = command_from_payload(payload, context=context)
        except ValidationError as exc:
            return InteractionResult(
                status="rejected",
                user_message=f"Invalid command payload: {exc}",
                reason_code="invalid_payload",
                decision_provenance=DecisionProvenance(
                    policy_id="interaction-policy-v1",
                    rule_id="policy.validation.payload",
                ),
            )
        return await self.dispatch(command, context=context)
