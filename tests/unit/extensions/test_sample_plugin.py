from __future__ import annotations

import types
from importlib import metadata

import pytest

from examples.plugins.sample_plugin import sample_plugin
from src.extensions import BEHAVIOR_REGISTRY, load_plugins


class DummyEntryPoints(list[object]):
    def select(self, group: str | None = None) -> list[object]:
        return self if group == "culture.plugins" else []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sample_plugin_load(monkeypatch: pytest.MonkeyPatch) -> None:
    BEHAVIOR_REGISTRY._behaviors.clear()
    widgets: list[tuple[str, str, str]] = []

    async def register_widget_backend(
        name: str, script_url: str, backend_url: str = "http://localhost:8000"
    ) -> None:
        widgets.append((name, script_url, backend_url))

    ep = types.SimpleNamespace(load=lambda: sample_plugin.setup, name="sample_plugin")
    monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints([ep]))
    monkeypatch.setattr("src.extensions.register_widget_backend", register_widget_backend)

    await load_plugins(backend_url="http://backend")

    assert sample_plugin.log_turn in BEHAVIOR_REGISTRY._behaviors
    assert widgets == [("SampleWidget", "http://localhost:5173/sample.js", "http://backend")]
