"""Simple policy filter for outgoing messages."""

from __future__ import annotations

import os


def allow_message(content: str | None) -> bool:
    """Return True if the message is allowed by policy."""
    if content is None:
        return True
    blocked_words = os.getenv("OPA_BLOCKLIST", "").split(",")
    content_lower = content.lower()
    return not any(word.strip().lower() in content_lower for word in blocked_words if word.strip())
