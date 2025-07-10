# Extensions and Plug-ins

Culture allows third-party packages to contribute UI widgets and agent behaviors.
This document shows how to create plug-ins using the hooks exposed in
`src/extensions`.

## Widget Registration

Use `register_widget_backend` to inform the backend about a widget:

```python
from src.extensions import register_widget_backend

register_widget_backend(
    name="ExampleWidget",
    script_url="http://localhost:5173/example.js",
)
```

The backend stores the widget metadata so the dashboard can load it dynamically.

## Agent Behavior Hooks

Plug-ins can extend agent behavior by registering a callback executed after each
agent turn:

```python
from src.extensions import register_agent_behavior

def log_turn(agent: object, output: dict[str, object]) -> None:
    print(f"[PLUGIN] {agent.agent_id}: {output}")

register_agent_behavior(log_turn)
```

Each callback receives the agent instance and the dictionary returned from
`Agent.run_turn`. Multiple behaviors can be registered and will be invoked in the
order added.

