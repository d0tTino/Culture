# Resource Plug-in

This example package registers a custom map action `gather_crystal` that lets agents collect crystals from the world map.

Install the package in editable mode:

```bash
pip install -e examples/plugins/resource_plugin
```

Load the plug-in so Culture can register the action:

```python
from src.extensions import load_plugins

await load_plugins()
```
