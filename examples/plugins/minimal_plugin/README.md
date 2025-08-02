# Minimal Plug-in

This example package demonstrates how to register a dashboard widget and an agent behavior.

Install the package in editable mode:

```bash
pip install -e examples/plugins/minimal_plugin
```

Load plug-ins so Culture discovers it:

```python
from src.extensions import load_plugins

await load_plugins()
```
