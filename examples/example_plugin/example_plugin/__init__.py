"""Example plug-in providing both widget and agent hooks."""

from src.extensions import register_agent_behavior


def log_turn(agent: object, output: dict[str, object]) -> None:
    """Log the agent's output to stdout."""
    print(f"[EXAMPLE_PLUGIN] {getattr(agent, 'agent_id', 'unknown')}: {output}")


def setup() -> dict[str, str]:
    """Entry point used by :func:`load_plugins`."""
    register_agent_behavior(log_turn)
    return {
        "name": "ExampleWidget",
        "script_url": "http://localhost:5173/example.js",
    }
