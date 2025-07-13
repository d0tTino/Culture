from __future__ import annotations

import types
from importlib import metadata
from typing import Any

import pytest

from src.extensions import load_plugins


class DummyEntryPoints(list[Any]):
    def select(self, group: str | None = None) -> list[Any]:
        return self if group == "culture.plugins" else []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_plugins_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []

    def plugin() -> None:
        called.append(True)

    ep = types.SimpleNamespace(load=lambda: plugin, name="dummy")
    monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints([ep]))

    await load_plugins()

    assert called


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_plugins_registers_widget(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[tuple[str, str, str]] = []

    def plugin() -> dict[str, str]:
        return {"name": "Widget", "script_url": "http://x/y.js"}

    async def register_widget_backend(
        name: str, script_url: str, backend_url: str = "http://localhost:8000"
    ) -> None:
        recorded.append((name, script_url, backend_url))

    ep = types.SimpleNamespace(load=lambda: plugin, name="widget")
    monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints([ep]))
    monkeypatch.setattr("src.extensions.register_widget_backend", register_widget_backend)

    await load_plugins(backend_url="http://backend")

    assert recorded == [("Widget", "http://x/y.js", "http://backend")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_plugins_awaits_coroutine(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []

    async def plugin() -> None:
        called.append(True)

    ep = types.SimpleNamespace(load=lambda: plugin, name="dummy")
    monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints([ep]))

    await load_plugins()

    assert called


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_plugin_registers_widget(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[tuple[str, str, str]] = []

    async def plugin() -> dict[str, str]:
        return {"name": "Widget", "script_url": "http://x/y.js"}

    async def register_widget_backend(
        name: str, script_url: str, backend_url: str = "http://localhost:8000"
    ) -> None:
        recorded.append((name, script_url, backend_url))

    ep = types.SimpleNamespace(load=lambda: plugin, name="widget")
    monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints([ep]))
    monkeypatch.setattr("src.extensions.register_widget_backend", register_widget_backend)

    await load_plugins(backend_url="http://backend")

    assert recorded == [("Widget", "http://x/y.js", "http://backend")]
