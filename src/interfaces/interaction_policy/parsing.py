from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_MENTION_TARGET_RE = re.compile(r"^@(?P<agent>[\w.-]+)\s*:\s*(?P<content>.+)$", re.DOTALL)
_DM_TARGET_RE = re.compile(r"^/dm\s+(?P<agent>[\w.-]+)\s+(?P<content>.+)$", re.DOTALL)


def parse_message_routing(content: str) -> tuple[str | None, bool, str, str | None]:
    """Parse optional transport-neutral routing directives from text input."""
    cleaned = content.strip()
    if not cleaned:
        return None, False, "", None

    if cleaned == "/broadcast":
        return None, True, "", "Broadcast message cannot be empty. Use /broadcast <message>."

    if cleaned.startswith("/broadcast "):
        payload = cleaned[len("/broadcast ") :].strip()
        if not payload:
            return (
                None,
                True,
                "",
                "Broadcast message cannot be empty. Use /broadcast <message>.",
            )
        return None, True, payload, None

    mention_match = _MENTION_TARGET_RE.match(cleaned)
    if mention_match:
        return mention_match.group("agent"), False, mention_match.group("content").strip(), None

    dm_match = _DM_TARGET_RE.match(cleaned)
    if dm_match:
        return dm_match.group("agent"), False, dm_match.group("content").strip(), None

    return None, False, cleaned, None


def discord_message_to_intent_payload(
    *,
    content: str,
    sender_agent_id: str | None,
    fallback_agent_id: str | None,
    raw_metadata: Mapping[str, Any] | None = None,
    mode: str = "participant",
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert text content into canonical interaction payload."""
    recipient, is_broadcast, parsed_content, validation_error = parse_message_routing(content)
    if validation_error is not None:
        return None, validation_error
    if not parsed_content:
        return None, None

    target_agent_id = recipient or sender_agent_id or fallback_agent_id

    payload: dict[str, Any] = {
        "intent": "broadcast" if is_broadcast else ("direct_message" if recipient else "human_message"),
        "mode": mode,
        "text": parsed_content,
        "routing": {
            "target_agent_id": target_agent_id,
            "recipient_id": recipient,
        },
        "metadata": dict(raw_metadata or {}),
    }
    return payload, None


def parse_discord_message_routing(content: str) -> tuple[str | None, bool, str, str | None]:
    """Backward-compatible alias for transport adapters."""
    return parse_message_routing(content)
