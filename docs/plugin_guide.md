# Plug-in Quickstart Guide

This guide explains how to implement and register a Culture plug-in using the `culture.plugins` entry point.

## 1. Create the plug-in code

Define a function that registers any widgets or agent behaviors and return optional widget information.

```python
# my_plugin/__init__.py
from typing import Any
from src.extensions import PluginResult, register_agent_behavior

def log_turn(agent: Any, output: dict[str, Any]) -> None:
    print(f"[MY_PLUGIN] {output}")

def setup() -> PluginResult:
    register_agent_behavior(log_turn)
    return {
        "name": "MyWidget",
        "script_url": "http://localhost:5173/my_widget.js",
    }
```

## 2. Declare the entry point

Expose the plug-in via `pyproject.toml` so Culture can discover it:

```toml
[project.entry-points."culture.plugins"]
my_plugin = "my_plugin:setup"
```

## 3. Install and load the plug-in

Install the package in editable mode and call `load_plugins()` at startup:

```bash
pip install -e path/to/my_plugin
```

```python
from src.extensions import load_plugins

await load_plugins()
```

Culture will execute the `setup` function and register any declared widgets or behaviors. See [plugins.md](plugins.md) for a more detailed reference.
