"""Example plug-in adding custom behavior after each agent turn."""

from src.extensions import register_agent_behavior


def log_turn(agent: object, output: dict[str, object]) -> None:
    """Simple behavior that logs the agent's message."""
    print(f"[PLUGIN] {getattr(agent, 'agent_id', 'unknown')}: {output}")


register_agent_behavior(log_turn)
