from __future__ import annotations

import re

_MENTION_TARGET_RE = re.compile(r"^@(?P<agent>[\w.-]+)\s*:\s*(?P<content>.+)$", re.DOTALL)
_DM_TARGET_RE = re.compile(r"^/dm\s+(?P<agent>[\w.-]+)\s+(?P<content>.+)$", re.DOTALL)


def parse_discord_message_routing(content: str) -> tuple[str | None, bool, str, str | None]:
    """Parse optional routing directives from plain Discord messages."""
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
