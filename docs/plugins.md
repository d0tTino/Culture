# Extensions and Plug-ins

Culture allows third-party packages to contribute UI widgets and agent behaviors.
This document shows how to create plug-ins using the hooks exposed in
`src/extensions`.

## Reference

Culture exposes a small API for plug-in authors:

| Function | Description |
| -------- | ----------- |
| `register_widget_backend(name, script_url, backend_url="http://localhost:8000")` | Register a widget with the backend so the dashboard can load it. |
| `register_agent_behavior(func)` | Register a callback executed after each agent turn. |
| `register_map_action(name, handler)` | Register a custom world map action handler. |
| `load_plugins(group="culture.plugins", *, backend_url="http://localhost:8000")` | Load plug-ins declared under the given entry point group. |

Two protocol classes define the expected signatures:

```python
from src.extensions import AgentBehavior, Plugin

def behavior(agent: object, output: dict[str, object]) -> None:
    ...

def setup() -> dict[str, str] | None:
    ...

behavior_fn: AgentBehavior = behavior
plugin: Plugin = setup
```

`AgentBehavior` is any callable that accepts the running agent instance and the
dictionary returned from `Agent.run_turn`. A `Plugin` is a callable (sync or
async) invoked by `load_plugins`; it may return a dictionary containing
`name` and `script_url` to automatically register a widget.

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

## Map Action Hooks

Custom world map actions let plug-ins modify the simulation when an agent issues
an unknown action. Use `register_map_action` to associate a handler with an
action name:

```python
from src.extensions import register_map_action

def dance_action(sim, idx, agent_id, state, action):
    state.has_danced = True
    return {"result": "danced"}

register_map_action("dance", dance_action)
```

When an agent outputs `{"action": "dance"}` the handler is invoked and its
returned dictionary is included in the emitted map event.

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

## Step-by-Step Tutorial

The following guide walks through creating, installing, and verifying a simple
plug-in that registers both an agent behavior and a UI widget.

1. **Create a package** with an entry point:

   ```text
   my_plugin/
       __init__.py
       pyproject.toml
   ```

2. **Implement the plug-in** in `my_plugin/__init__.py`:

   ```python
   from typing import Any
   from src.extensions import PluginResult, register_agent_behavior

   def greet(agent: Any, output: dict[str, Any]) -> None:
       print(f"[MY_PLUGIN] {output}")

   def setup() -> PluginResult:
       register_agent_behavior(greet)
       return {
           "name": "MyWidget",
           "script_url": "http://localhost:5173/my_widget.js",
       }
   ```

3. **Declare the entry point** in `pyproject.toml`:

   ```toml
   [project.entry-points."culture.plugins"]
   my_plugin = "my_plugin:setup"
   ```

4. **Install your plug-in** in editable mode and load it:

   ```bash
   pip install -e path/to/my_plugin
   ```

   ```python
   from src.extensions import load_plugins
   await load_plugins()
   ```

5. **Verify installation** by running Culture and observing the `[MY_PLUGIN]`
   log message. The dashboard should also list `MyWidget` if the widget URL is
   reachable.


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

## Installing the Sample Plug-in

For a minimal demonstration look at `examples/plugins/sample_plugin`. Install it
in editable mode:

```bash
pip install -e examples/plugins/sample_plugin
```

Load the plug-in at startup so Culture can register its hooks:

```python
from src.extensions import load_plugins

await load_plugins()
```
