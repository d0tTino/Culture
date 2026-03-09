from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from src.interfaces.interaction_schema import (
    BroadcastEnvelope,
    DirectMessageEnvelope,
    HumanMessageEnvelope,
    InjectEventEnvelope,
    InteractionContext,
    InteractionEnvelope,
    KnowledgeBoardEnvelope,
    ModerationEnvelope,
    SpawnEnvelope,
    parse_interaction_envelope,
)
from src.sim.commands.domain_commands import (
    BroadcastCommand,
    ControlCommand,
    DirectMessageCommand,
    DomainCommandT,
    HumanMessageCommand,
    InjectEventCommand,
    KnowledgeBoardCommand,
    ModerationCommand,
    SpawnAgentCommand,
)


def parse_bus_command(
    payload: Mapping[str, Any],
    *,
    context: InteractionContext | None = None,
) -> InteractionEnvelope:
    data = _normalize_legacy_payload(payload)
    ctx = context or InteractionContext()
    intent = _canonical_intent(data)

    routing = data.get("routing") if isinstance(data.get("routing"), Mapping) else {}
    auth = data.get("auth") if isinstance(data.get("auth"), Mapping) else {}
    budget = data.get("budget") if isinstance(data.get("budget"), Mapping) else {}

    metadata = _metadata_with_context(data, ctx)
    correlation_id = (
        data.get("correlation_id") or metadata.get("correlation_id") or metadata.get("request_id")
    )

    envelope_payload: dict[str, Any] = {
        "intent": intent,
        "correlation_id": str(correlation_id) if correlation_id is not None else None,
        "routing": {
            "sender_id": routing.get("sender_id") or data.get("sender_id") or ctx.sender_id,
            "source": routing.get("source") or data.get("source") or ctx.source,
            "channel_id": routing.get("channel_id") or data.get("channel_id") or ctx.channel_id,
            "recipient_id": routing.get("recipient_id") or data.get("recipient_id"),
            "target_agent_id": routing.get("target_agent_id") or data.get("target_agent_id"),
        },
        "auth": {
            "permissions": auth.get("permissions")
            if auth.get("permissions") is not None
            else set(ctx.permissions),
        },
        "budget": {
            "budget_agent_id": budget.get("budget_agent_id") or data.get("budget_agent_id"),
            "attribution_scope": budget.get("attribution_scope") or "default",
        },
        "metadata": metadata,
    }

    if intent in {"human_message", "direct_message", "broadcast", "knowledge_board"}:
        envelope_payload["text"] = str(data.get("text") or "")
    elif intent == "spawn":
        envelope_payload.update(
            {
                "agent_id": _string_or_none(data.get("agent_id") or data.get("author")),
                "role": data.get("role"),
                "persona": _string_or_none(data.get("persona")),
                "backstory": _string_or_none(data.get("backstory")),
                "traits": data.get("traits") if isinstance(data.get("traits"), dict) else None,
            }
        )
    elif intent == "inject_event":
        envelope_payload.update(
            {
                "text": str(data.get("text") or ""),
                "scope": str(data.get("scope") or "global"),
                "agent_id": _string_or_none(data.get("agent_id") or data.get("author")),
            }
        )
    elif intent in {"moderation", "control"}:
        envelope_payload.update(
            {
                "action": str(data.get("action") or data.get("command") or intent),
                "value": _float_or_none(data.get("value")),
                "tags": _coerce_tags(data.get("tags")),
                "agent_id": _string_or_none(data.get("agent_id")),
            }
        )

    return parse_interaction_envelope(envelope_payload)


def command_from_payload(
    payload: Mapping[str, Any],
    *,
    context: InteractionContext | None = None,
) -> DomainCommandT:
    return command_from_envelope(parse_bus_command(payload, context=context))


def command_from_discord_message(
    *,
    content: str,
    recipient_id: str | None,
    is_broadcast: bool,
    target_agent_id: str | None,
    budget_agent_id: str | None = None,
    raw_payload: Mapping[str, Any] | None = None,
) -> DomainCommandT:
    metadata = dict(raw_payload or {})
    if is_broadcast:
        return BroadcastCommand(
            content=content, budget_agent_id=budget_agent_id, metadata=metadata
        )
    if recipient_id is not None:
        return DirectMessageCommand(
            content=content,
            recipient_id=recipient_id,
            target_agent_id=target_agent_id,
            budget_agent_id=budget_agent_id,
            metadata=metadata,
        )
    return HumanMessageCommand(
        content=content,
        recipient_id=recipient_id,
        target_agent_id=target_agent_id,
        metadata=metadata,
    )


def command_from_envelope(envelope: InteractionEnvelope) -> DomainCommandT:
    if isinstance(envelope, HumanMessageEnvelope):
        return HumanMessageCommand(
            content=envelope.text,
            recipient_id=envelope.routing.recipient_id,
            target_agent_id=envelope.routing.target_agent_id,
            metadata=envelope.metadata,
        )
    if isinstance(envelope, DirectMessageEnvelope):
        return DirectMessageCommand(
            content=envelope.text,
            recipient_id=envelope.routing.recipient_id,
            target_agent_id=envelope.routing.target_agent_id,
            budget_agent_id=envelope.budget.budget_agent_id,
            metadata=envelope.metadata,
        )
    if isinstance(envelope, BroadcastEnvelope):
        return BroadcastCommand(
            content=envelope.text,
            budget_agent_id=envelope.budget.budget_agent_id,
            metadata=envelope.metadata,
        )
    if isinstance(envelope, KnowledgeBoardEnvelope):
        return KnowledgeBoardCommand(content=envelope.text, metadata=envelope.metadata)
    if isinstance(envelope, SpawnEnvelope):
        return SpawnAgentCommand(
            agent_id=envelope.agent_id,
            role=envelope.role,
            persona=envelope.persona,
            backstory=envelope.backstory,
            traits=envelope.traits,
            metadata=envelope.metadata,
        )
    if isinstance(envelope, InjectEventEnvelope):
        return InjectEventCommand(
            text=envelope.text,
            scope=envelope.scope,
            agent_id=envelope.agent_id,
            metadata=envelope.metadata,
        )
    if isinstance(envelope, ModerationEnvelope):
        return ModerationCommand(
            action=envelope.action,
            agent_id=envelope.agent_id,
            value=envelope.value,
            metadata=envelope.metadata,
        )
    return ControlCommand(
        action=envelope.action,
        value=envelope.value,
        tags=envelope.tags,
        agent_id=envelope.agent_id,
        metadata=envelope.metadata,
    )


def _canonical_intent(data: Mapping[str, Any]) -> str:
    command = str(
        data.get("intent")
        or data.get("type")
        or data.get("command_type")
        or data.get("command")
        or ""
    ).strip()
    if command == "dm":
        return "direct_message"
    if command == "kb":
        return "knowledge_board"
    if command in {
        "pause",
        "resume",
        "pause_all",
        "start",
        "stop",
        "set_speed",
        "kill_agent",
        "checkpoint",
        "replay_to_step",
    }:
        return "control"
    if command in {"moderation", "reset_memory", "penalty", "mute", "unmute"}:
        return "moderation"
    if command == "inject_event":
        return "inject_event"
    if command in {
        "human_message",
        "direct_message",
        "broadcast",
        "knowledge_board",
        "spawn",
        "control",
    }:
        return command
    return "human_message"


def _normalize_legacy_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    if "content" in data and "text" not in data:
        data["text"] = data["content"]
        warnings.warn("'content' is deprecated; use 'text'.", DeprecationWarning, stacklevel=3)
    if "prompt" in data and "scope" not in data:
        data["scope"] = data["prompt"]
        warnings.warn("'prompt' is deprecated; use 'scope'.", DeprecationWarning, stacklevel=3)
    if "command_type" in data and "intent" not in data and "type" not in data:
        warnings.warn(
            "'command_type' is deprecated; use 'intent'.", DeprecationWarning, stacklevel=3
        )
    if "type" in data and "intent" not in data:
        data["intent"] = data["type"]
        warnings.warn("'type' is deprecated; use 'intent'.", DeprecationWarning, stacklevel=3)
    return data


def _metadata_with_context(data: Mapping[str, Any], context: InteractionContext) -> dict[str, Any]:
    return {**dict(data), "adapter_context": context.model_dump()}


def _string_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_tags(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]
