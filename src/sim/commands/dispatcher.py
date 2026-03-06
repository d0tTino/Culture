from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from src.governance.decision_kernel import DecisionProvenance
from src.interfaces.interaction_schema import InteractionContext, InteractionResult
from src.sim.commands.domain_commands import DomainCommandT

if TYPE_CHECKING:
    from src.sim.command_service import SimulationCommandService


class SimulationCommandDispatcher:
    """Single ingress dispatcher for payloads, envelopes, and typed domain commands."""

    def __init__(self, command_service: SimulationCommandService) -> None:
        self.command_service = command_service

    async def dispatch(
        self,
        command: DomainCommandT,
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        try:
            return await self.command_service.execute(command, context=context)
        except Exception as exc:  # pragma: no cover - defensive normalization path
            return self._error_result(str(exc), reason_code="command_dispatch_failed")

    async def dispatch_envelope(
        self,
        envelope: Any,
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        try:
            from src.interfaces.domain_command_adapters import command_from_envelope

            result = await self.dispatch(command_from_envelope(envelope), context=context)
        except ValidationError as exc:
            return self._error_result(
                f"Invalid interaction envelope: {exc}",
                reason_code="invalid_payload",
                status="rejected",
            )

        correlation_id = getattr(envelope, "correlation_id", None)
        if result.correlation_id is None and isinstance(correlation_id, str):
            return result.model_copy(update={"correlation_id": correlation_id})
        return result

    async def dispatch_payload(
        self,
        payload: Mapping[str, Any],
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        try:
            from src.interfaces.domain_command_adapters import parse_bus_command

            envelope = parse_bus_command(payload, context=context)
        except ValidationError as exc:
            return self._error_result(
                f"Invalid command payload: {exc}",
                reason_code="invalid_payload",
                status="rejected",
            )
        return await self.dispatch_envelope(envelope, context=context)

    def _error_result(
        self,
        message: str,
        *,
        reason_code: str,
        status: Literal["ok", "rejected", "error"] = "error",
    ) -> InteractionResult:
        return InteractionResult(
            status=status,
            user_message=message,
            reason_code=reason_code,
            decision_provenance=DecisionProvenance(
                policy_id="interaction-policy-v1",
                rule_id=f"dispatcher.{reason_code}",
            ),
        )
