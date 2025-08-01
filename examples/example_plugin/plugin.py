"""Example Culture plug-in demonstrating widget and agent hooks."""

from typing import Any

from src.extensions import PluginResult, register_agent_behavior

# Agent-behavior callback executed after every agent action.
def log_turn(agent: Any, output: dict[str, Any]) -> None:
    """Log each agent turn.

    This function is registered as an agent-behavior callback via
    :func:`register_agent_behavior`. Culture will invoke it after every
    agent action, providing the agent instance and the emitted output.
    """

    print(f"[EXAMPLE_PLUGIN] {getattr(agent, 'agent_id', 'unknown')}: {output}")


# Entry point discovered by ``load_plugins`` during app startup.
def setup() -> PluginResult:
    """Register hooks with Culture and expose the ExampleWidget.

    The :func:`load_plugins` helper calls this entry point during
    application startup. It registers ``log_turn`` so the callback is
    executed on each agent turn and returns metadata for the UI widget.
    """

    # Register ``log_turn`` so Culture logs every agent output.
    register_agent_behavior(log_turn)
    return {
        "name": "ExampleWidget",
        # UI will load this widget script from the plug-in's ``static`` directory.
        "script_url": "static/example.js",
    }
