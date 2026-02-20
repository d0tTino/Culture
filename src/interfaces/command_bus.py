from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from src.governance.decision_kernel import DecisionProvenance
from src.interfaces.interaction_commands import (
    InteractionContext,
    InteractionEnvelope,
    InteractionResult,
)

if TYPE_CHECKING:
    from src.interfaces.interaction_commands import InteractionService


class CommandBus:
    """Thin adapter that normalizes payload shape and forwards canonical envelopes."""

    def __init__(self, interaction_service: InteractionService) -> None:
        self.interaction_service = interaction_service

    async def dispatch(
        self,
        command: InteractionEnvelope,
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        return await self.interaction_service.execute(command, context=context)

    async def dispatch_payload(
        self,
        payload: Mapping[str, Any],
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        try:
            envelope = parse_bus_command(payload, context=context)
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
        return await self.dispatch(envelope, context=context)


def parse_bus_command(
    payload: Mapping[str, Any],
    *,
    context: InteractionContext | None = None,
) -> InteractionEnvelope:
    """Validate transport payload and normalize aliases to canonical envelope fields."""
    data = dict(payload)
    command = str(data.get("command_type") or data.get("command") or "").strip()
    if command == "dm":
        data["command_type"] = "direct_message"
    elif command == "kb":
        data["command_type"] = "knowledge_board"
    elif command in {"pause", "resume", "pause_all", "start", "stop", "set_speed", "kill_agent"}:
        data["command_type"] = "control"
    elif command == "inject_event":
        data["command_type"] = "inject_event"
    elif command:
        data["command_type"] = command

    ctx = context or InteractionContext()
    permissions = data.get("permissions")
    if isinstance(permissions, list | set | tuple):
        normalized_permissions = set(str(item) for item in permissions)
    else:
        normalized_permissions = set(ctx.permissions)

    envelope_payload: dict[str, Any] = {
        "intent": data.get("command_type") or "moderation",
        "content": data.get("content"),
        "action": data.get("action") or command or data.get("command_type") or data.get("command"),
        "value": data.get("value"),
        "tags": data.get("tags"),
        "prompt": data.get("prompt") or data.get("scope"),
        "text": data.get("text") or data.get("content"),
        "agent_id": data.get("agent_id") or data.get("author"),
        "role": data.get("role"),
        "persona": data.get("persona"),
        "backstory": data.get("backstory"),
        "traits": data.get("traits"),
        "routing": {
            "sender_id": str(data.get("sender_id", ctx.sender_id)),
            "source": str(data.get("source", ctx.source)),
            "channel_id": str(data.get("channel_id")) if data.get("channel_id") is not None else ctx.channel_id,
            "recipient_id": data.get("recipient_id"),
            "target_agent_id": data.get("target_agent_id"),
        },
        "auth": {"permissions": normalized_permissions},
        "budget": {"budget_agent_id": data.get("budget_agent_id")},
        "metadata": data,
    }
    return InteractionEnvelope.model_validate(envelope_payload)
