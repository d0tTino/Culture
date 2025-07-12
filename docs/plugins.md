# Extensions and Plug-ins

Culture allows third-party packages to contribute UI widgets and agent behaviors.
This document shows how to create plug-ins using the hooks exposed in
`src/extensions`.

## Widget Registration

Use `register_widget_backend` to inform the backend about a widget:

```python
from src.extensions import register_widget_backend

await register_widget_backend(
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

## Loading Plug-ins via Entry Points

Third-party packages can expose a plug-in through the `culture.plugins` entry
point. Implement a function that registers behaviors or widgets and return a
dictionary with optional widget information:

```python
# my_plugin/__init__.py
from src.extensions import register_agent_behavior

def greet(agent: object, output: dict[str, object]) -> None:
    print("hello from plugin")

def setup() -> dict[str, str] | None:
    register_agent_behavior(greet)
    return {"name": "ExampleWidget", "script_url": "http://localhost:5173/example.js"}
```

Declare the entry point in your package configuration:

```toml
[project.entry-points."culture.plugins"]
my_plugin = "my_plugin:setup"
```

Call `load_plugins()` during application startup to execute all discovered
plug-ins. Any dictionary returned is passed to `register_widget_backend`:

```python
from src.extensions import load_plugins

await load_plugins()
```


## Installing the Example Plug-in

An example package is included in `examples/example_plugin`. Install it in editable mode:

```bash
pip install -e examples/example_plugin
```

After installation call `load_plugins()` so Culture can register its hooks:

```python
from src.extensions import load_plugins

await load_plugins()
```
