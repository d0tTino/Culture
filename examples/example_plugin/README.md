# Example Culture Plug-in

This example package shows how a third-party distribution can extend Culture.
It registers a simple agent behavior and a UI widget using the
`culture.plugins` entry point.

For a step-by-step guide, see [../../docs/plugin_guide.md](../../docs/plugin_guide.md).

Install the package in editable mode:

```bash
pip install -e examples/example_plugin
```

Then call `load_plugins()` at application startup to register the hooks.

