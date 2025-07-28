# Memory Widget Plug-in

This example package registers a widget that displays recent semantic summaries for an agent. The widget's script is served from `https://localhost:5173/memory_widget.js`.

Install the package in editable mode:

```bash
pip install -e examples/plugins/memory_widget
```

Load the plug-in so Culture can register the widget:

```python
from src.extensions import load_plugins

await load_plugins()
```
