"""Simple policy filter for outgoing messages."""

from __future__ import annotations

import asyncio
import logging
import os
import typing

import requests

from src.infra import config


def allow_message(content: str | None) -> bool:
    """Return True if the message is allowed by policy."""
    if content is None:
        return True
    blocked_words = os.getenv("OPA_BLOCKLIST", "").split(",")
    content_lower = content.lower()
    return not any(word.strip().lower() in content_lower for word in blocked_words if word.strip())


async def evaluate_with_opa(content: str) -> tuple[bool, str]:
    """Check message content against OPA policy."""
    url = typing.cast(str, config.get_config("OPA_URL"))
    if not url:
        return True, content
    try:
        response = await asyncio.to_thread(
            requests.post, url, json={"input": {"message": content}}, timeout=2
        )
        data = response.json()
        result = typing.cast(dict[str, typing.Any], data.get("result", {}))
        allow = typing.cast(bool, result.get("allow", True))
        new_content = typing.cast(str, result.get("content", content))
        return allow, new_content
    except Exception as e:  # pragma: no cover - network failures
        logging.getLogger(__name__).warning("OPA evaluation failed: %s", e)
        return True, content
