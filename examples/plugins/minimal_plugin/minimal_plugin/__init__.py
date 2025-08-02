"""Minimal plug-in registering a widget and agent behavior."""

from typing import Any

from src.extensions import PluginResult, register_agent_behavior


def echo(agent: Any, output: dict[str, Any]) -> None:
    """Log the agent output for demonstration."""
    print(f"[MINIMAL_PLUGIN] {getattr(agent, 'agent_id', 'unknown')}: {output}")


def setup() -> PluginResult:
    """Entry point for :func:`load_plugins`."""
    register_agent_behavior(echo)
    return {
        "name": "MinimalWidget",
        "script_url": "http://localhost:5173/minimal.js",
    }
