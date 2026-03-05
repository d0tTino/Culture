from __future__ import annotations

from typing import Any

from src.interfaces.interaction_commands import InteractionContext


def build_interaction_context(payload: dict[str, Any], *, default_source: str) -> InteractionContext:
    permissions = payload.get("permissions", [])
    return InteractionContext(
        sender_id=str(payload.get("sender_id", payload.get("author", "external"))),
        channel_id=str(payload.get("channel_id")) if payload.get("channel_id") else None,
        source=str(payload.get("source", default_source)),
        permissions=(set(permissions) if isinstance(permissions, list | set | tuple) else set()),
        metadata={k: v for k, v in payload.items() if k != "permissions"},
    )


def normalize_human_command_payload(text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(metadata or {})
    command_type = payload.get("command_type")
    if not command_type and bool(payload.get("broadcast")):
        command_type = "broadcast"
    return {"command_type": command_type or "human_message", "content": text, **payload}
