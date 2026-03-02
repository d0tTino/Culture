from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.interfaces.interaction_schema import InteractionContext, InteractionEnvelope
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


def command_from_payload(
    payload: Mapping[str, Any],
    *,
    context: InteractionContext | None = None,
) -> DomainCommandT:
    data = dict(payload)
    ctx = context or InteractionContext()
    command = str(data.get("command_type") or data.get("command") or "").strip()
    if command == "dm":
        command = "direct_message"
    elif command == "kb":
        command = "knowledge_board"
    elif command in {"pause", "resume", "pause_all", "start", "stop", "set_speed", "kill_agent"}:
        command = "control"
    elif command == "inject_event":
        command = "inject_event"

    metadata = {**data, "adapter_context": ctx.model_dump()}
    if command == "direct_message":
        return DirectMessageCommand(
            content=str(data.get("content") or data.get("text") or ""),
            recipient_id=_string_or_none(data.get("recipient_id")),
            target_agent_id=_string_or_none(data.get("target_agent_id")),
            budget_agent_id=_string_or_none(data.get("budget_agent_id")),
            metadata=metadata,
        )
    if command == "broadcast":
        return BroadcastCommand(
            content=str(data.get("content") or data.get("text") or ""),
            budget_agent_id=_string_or_none(data.get("budget_agent_id")),
            metadata=metadata,
        )
    if command == "knowledge_board":
        return KnowledgeBoardCommand(content=str(data.get("content") or ""), metadata=metadata)
    if command == "spawn":
        return SpawnAgentCommand(
            agent_id=_string_or_none(data.get("agent_id") or data.get("author")),
            role=data.get("role"),
            persona=_string_or_none(data.get("persona")),
            backstory=_string_or_none(data.get("backstory")),
            traits=data.get("traits") if isinstance(data.get("traits"), dict) else None,
            metadata=metadata,
        )
    if command == "inject_event":
        return InjectEventCommand(
            text=str(data.get("text") or data.get("content") or ""),
            scope=str(data.get("prompt") or data.get("scope") or "global"),
            agent_id=_string_or_none(data.get("agent_id") or data.get("author")),
            metadata=metadata,
        )
    if command in {"moderation", "reset_memory", "penalty", "mute", "unmute"}:
        return ModerationCommand(
            action=str(data.get("action") or command or data.get("command") or "moderation"),
            agent_id=_string_or_none(data.get("agent_id")),
            value=_float_or_none(data.get("value")),
            metadata=metadata,
        )
    if command == "control" or command in {
        "pause",
        "resume",
        "pause_all",
        "start",
        "stop",
        "set_speed",
        "kill_agent",
        "set_breakpoints",
    }:
        return ControlCommand(
            action=str(data.get("action") or data.get("command") or "control"),
            value=_float_or_none(data.get("value")),
            tags=_coerce_tags(data.get("tags")),
            agent_id=_string_or_none(data.get("agent_id")),
            metadata=metadata,
        )
    return HumanMessageCommand(
        content=str(data.get("content") or data.get("text") or ""),
        recipient_id=_string_or_none(data.get("recipient_id")),
        target_agent_id=_string_or_none(data.get("target_agent_id")),
        metadata=metadata,
    )


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
        return BroadcastCommand(content=content, budget_agent_id=budget_agent_id, metadata=metadata)
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
    payload = envelope.model_dump()
    payload["command_type"] = envelope.intent
    if envelope.action is not None:
        payload["command"] = envelope.action
    return command_from_payload(payload)


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
