"""Plug-in providing a simple dance map action."""

# mypy: ignore-errors

from __future__ import annotations

from typing import Any

from src.extensions import PluginResult, register_map_action


def dance_action(
    sim: Any, agent_index: int, agent_id: str, state: Any, action: dict[str, Any]
) -> dict[str, Any]:
    """Mark the agent as having danced and return a result."""
    setattr(state, "has_danced", True)
    return {"handled": True}


def setup() -> PluginResult:
    """Entry point for :func:`load_plugins`."""
    register_map_action("dance", dance_action)
    return None
