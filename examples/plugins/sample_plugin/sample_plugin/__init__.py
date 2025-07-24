"""Sample plug-in demonstrating Culture's extension hooks."""

from typing import Any

from src.extensions import register_agent_behavior


def log_turn(agent: Any, output: dict[str, Any]) -> None:
    """Log the agent's output."""
    print(f"[SAMPLE_PLUGIN] {getattr(agent, 'agent_id', 'unknown')}: {output}")


def setup() -> None:
    """Entry point for :func:`load_plugins`."""
    register_agent_behavior(log_turn)
    # Provide widget info for registration by :func:`load_plugins`
    return {
        "name": "SampleWidget",
        "script_url": "http://localhost:5173/sample.js",
    }
