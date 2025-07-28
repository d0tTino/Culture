from __future__ import annotations

import types
from importlib import metadata

import pytest

from src.extensions import BEHAVIOR_REGISTRY, load_plugins


class DummyEntryPoints(list[object]):
    def select(self, group: str | None = None) -> list[object]:
        return self if group == "culture.plugins" else []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_setup_registers_widget_and_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    BEHAVIOR_REGISTRY._behaviors.clear()
    called: list[bool] = []
    recorded: list[tuple[str, str, str]] = []

    def behavior(
        agent: object, output: dict[str, object]
    ) -> None:  # pragma: no cover - simple stub
        pass

    async def setup() -> dict[str, str]:
        from src.extensions import register_agent_behavior

        called.append(True)
        register_agent_behavior(behavior)
        return {"name": "AsyncWidget", "script_url": "http://x/y.js"}

    async def register_widget_backend(
        name: str, script_url: str, backend_url: str = "http://localhost:8000"
    ) -> None:
        recorded.append((name, script_url, backend_url))

    ep = types.SimpleNamespace(load=lambda: setup, name="async_plugin")
    monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints([ep]))
    monkeypatch.setattr("src.extensions.register_widget_backend", register_widget_backend)

    await load_plugins(backend_url="http://backend")

    assert called
    assert recorded == [("AsyncWidget", "http://x/y.js", "http://backend")]
    assert behavior in BEHAVIOR_REGISTRY._behaviors
