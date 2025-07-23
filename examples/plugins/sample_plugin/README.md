# Sample Culture Plug-in

This minimal package demonstrates how a third-party plug-in can extend Culture.
It registers a simple agent behavior and directly registers a UI widget using
`register_widget_backend`.

Install the package in editable mode:

```bash
pip install -e examples/plugins/sample_plugin
```

Then load the plug-in at application startup:

```python
from src.extensions import load_plugins

await load_plugins()
```
