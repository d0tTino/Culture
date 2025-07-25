import subprocess
import sys
from importlib import metadata

import pytest

from src.extensions import load_plugins
from src.interfaces import dashboard_backend as db


@pytest.mark.integration
@pytest.mark.asyncio
async def test_plugin_workflow(tmp_path, monkeypatch):
    """Generate, install, and load a temporary plug-in."""
    plugin_root = tmp_path / "temp_plugin"
    package_dir = plugin_root / "temp_plugin"
    package_dir.mkdir(parents=True)

    # Create minimal plugin implementation
    (package_dir / "__init__.py").write_text(
        """
from typing import Any
from src.extensions import PluginResult, register_agent_behavior

def log_turn(agent: Any, output: dict[str, Any]) -> None:
    print(f"[TEMP_PLUGIN] {output}")

def setup() -> PluginResult:
    register_agent_behavior(log_turn)
    return {"name": "TempWidget", "script_url": "http://localhost:5173/temp.js"}
""",
    )

    # Declare entry point
    (plugin_root / "pyproject.toml").write_text(
        """
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "temp-plugin"
version = "0.1.0"
requires-python = ">=3.10"

[project.entry-points."culture.plugins"]
temp_plugin = "temp_plugin:setup"
""",
    )

    # Install the temporary plugin in editable mode
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(plugin_root)],
        check=True,
    )

    # Reload site paths so the .pth file created by pip is honored
    import site

    for sp in site.getsitepackages():
        site.addsitedir(sp)

    metadata.distribution("temp-plugin")  # ensure metadata cache refreshed

    widgets: list[tuple[str, str, str]] = []

    async def register_widget_backend(
        name: str, script_url: str, *, backend_url: str = "http://localhost:8000"
    ) -> None:
        widgets.append((name, script_url, backend_url))
        db.WIDGET_REGISTRY.register(name, {"script_url": script_url})

    monkeypatch.setattr("src.extensions.register_widget_backend", register_widget_backend)
    db.WIDGET_REGISTRY._widgets.clear()

    await load_plugins()

    assert widgets == [("TempWidget", "http://localhost:5173/temp.js", "http://localhost:8000")]
    assert db.WIDGET_REGISTRY.get("TempWidget") == {"script_url": "http://localhost:5173/temp.js"}

    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "temp-plugin"], check=True)
