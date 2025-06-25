from __future__ import annotations

import asyncio
import logging
from typing import Any

import requests

from src.infra import config

logger = logging.getLogger(__name__)

_POLICY_CONTENT: str | None = None


def load_policy(path: str) -> str:
    """Load an OPA policy file into memory."""
    global _POLICY_CONTENT
    with open(path, encoding="utf-8") as f:
        _POLICY_CONTENT = f.read()
    return _POLICY_CONTENT


async def evaluate_policy(action: str) -> bool:
    """Evaluate an action against the loaded OPA policy via HTTP."""
    url = config.get_config("OPA_URL")
    if not url:
        return True
    payload: dict[str, Any] = {"input": {"action": action}}
    if _POLICY_CONTENT is not None:
        payload["policy"] = _POLICY_CONTENT
    try:
        response = await asyncio.to_thread(requests.post, url, json=payload, timeout=2)
        data = response.json()
        result = data.get("result", {})
        return bool(result.get("allow", True))
    except Exception as exc:  # pragma: no cover - network issues
        logger.warning("OPA policy evaluation failed: %s", exc)
        return True
