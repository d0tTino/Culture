"""Example plug-in providing both widget and agent hooks."""

from typing import Any

from src.extensions import PluginResult, register_agent_behavior


def log_turn(agent: Any, output: dict[str, Any]) -> None:
    """Log the agent's output to stdout."""
    print(f"[EXAMPLE_PLUGIN] {getattr(agent, 'agent_id', 'unknown')}: {output}")


def setup() -> PluginResult:
    """Entry point used by :func:`load_plugins`."""
    register_agent_behavior(log_turn)
    return {
        "name": "ExampleWidget",
        "script_url": "http://localhost:5173/example.js",
    }
