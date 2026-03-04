from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from src.infra import config
from src.utils.policy import evaluate_with_opa

if TYPE_CHECKING:
    from src.interfaces.interaction_schema import InteractionContext

logger = logging.getLogger(__name__)

_TRUE_BOOL_VALUES = {"1", "true", "yes", "on"}
_FALSE_BOOL_VALUES = {"0", "false", "no", "off"}
_DEFAULT_RATE_LIMIT_WINDOW_SECONDS: float = 60.0

_COMMAND_HISTORY: dict[str, deque[float]] = {}
_COMMAND_LOCKS: dict[str, asyncio.Lock] = {}
_MAX_RATE: int = 5


def has_admin_permission(user: Any) -> bool:
    perms = getattr(getattr(user, "guild_permissions", None), "administrator", False)
    return bool(perms)


def _coerce_to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_BOOL_VALUES:
            return True
        if lowered in _FALSE_BOOL_VALUES:
            return False
    return bool(value)


def allow_control_via_opa() -> bool:
    overrides = getattr(config, "CONFIG_OVERRIDES", {})
    value = overrides.get("DISCORD_ALLOW_OPA_CONTROL_COMMANDS")
    if value is None:
        value = config.get_config("DISCORD_ALLOW_OPA_CONTROL_COMMANDS")
    return _coerce_to_bool(value)


def get_command_rate_limit_window() -> float:
    overrides = getattr(config, "CONFIG_OVERRIDES", {})
    value = overrides.get("DISCORD_COMMAND_RATE_LIMIT_SECONDS")
    if value is None:
        value = config.get_config("DISCORD_COMMAND_RATE_LIMIT_SECONDS")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return _DEFAULT_RATE_LIMIT_WINDOW_SECONDS


async def has_control_command_permission(
    user: Any,
    command: str,
    *,
    agent_id: str | None = None,
) -> bool:
    if has_admin_permission(user):
        return True
    if not allow_control_via_opa():
        return False
    payload = {
        "command": command,
        "user_id": str(getattr(user, "id", "")),
    }
    if agent_id is not None:
        payload["agent_id"] = agent_id
    allowed, _ = await evaluate_with_opa(json.dumps(payload))
    return bool(allowed)


async def check_command_rate_limit(user: Any) -> bool:
    user_id = str(getattr(user, "id", ""))
    if not user_id:
        return True
    lock = _COMMAND_LOCKS.setdefault(user_id, asyncio.Lock())
    async with lock:
        history = _COMMAND_HISTORY.setdefault(user_id, deque())
        now = time.monotonic()
        window_seconds = get_command_rate_limit_window()
        if window_seconds <= 0:
            history.clear()
        else:
            cutoff = now - window_seconds
            while history and history[0] <= cutoff:
                history.popleft()
        if len(history) >= _MAX_RATE:
            logger.warning("Rate limit exceeded for user %s", user_id)
            return False
        history.append(now)
    return True


def reset_command_counts(user_id: str | None = None) -> None:
    if user_id is not None:
        history = _COMMAND_HISTORY.pop(user_id, None)
        if history is not None:
            history.clear()
    else:
        _COMMAND_HISTORY.clear()


def set_max_rate(value: int) -> None:
    global _MAX_RATE
    _MAX_RATE = max(1, int(value))


def context_is_authorized(context: InteractionContext, required: set[str]) -> bool:
    if not required:
        return True
    return bool(context.permissions & required)


def check_cooldown(
    *,
    key: str,
    now: float,
    cooldown: float,
    state: dict[str, float],
) -> float | None:
    last_seen = state.get(key, 0.0)
    elapsed = now - last_seen
    if elapsed < cooldown:
        return max(0.0, cooldown - elapsed)
    state[key] = now
    return None
