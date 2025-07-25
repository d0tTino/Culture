from __future__ import annotations

import types
from importlib import metadata

import pytest

from src.extensions import MAP_ACTION_REGISTRY, load_plugins
from src.sim.world_map_actions import process_map_action


class DummyEntryPoints(list[object]):
    def select(self, group: str | None = None) -> list[object]:
        return self if group == "culture.plugins" else []


calls: list[tuple[int, str, dict[str, object]]] = []


def custom_action(sim: object, idx: int, aid: str, state: object, action: dict[str, object]):
    calls.append((idx, aid, action))
    return {"handled": True}


def plugin_setup() -> None:
    from src.extensions import register_map_action

    register_map_action("dance", custom_action)


class DummyEventKernel:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def schedule_immediate(self, cb, *, vector) -> None:
        await cb()

    async def emit_environment_event(self, event: dict[str, object]) -> None:
        self.events.append(event)


class DummySimulation:
    def __init__(self) -> None:
        self.world_map = types.SimpleNamespace(to_dict=lambda: {})
        self.vector = types.SimpleNamespace(to_dict=lambda: {})
        self.event_kernel = DummyEventKernel()
        self.current_step = 0
        self.discord_bot = None
        self.agents = [types.SimpleNamespace(update_state=lambda s: None)]


async def noop(*args: object, **kwargs: object) -> None:  # pragma: no cover - helper
    pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_custom_map_action(monkeypatch: pytest.MonkeyPatch) -> None:
    MAP_ACTION_REGISTRY._actions.clear()
    ep = types.SimpleNamespace(load=lambda: plugin_setup, name="dance_plugin")
    monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints([ep]))
    monkeypatch.setattr("src.sim.world_map_actions.emit_map_change_event", noop)

    await load_plugins()

    assert "dance" in MAP_ACTION_REGISTRY._actions

    sim = DummySimulation()
    state = types.SimpleNamespace()
    await process_map_action(sim, 0, "agent", state, {"action": "dance"})

    assert calls == [(0, "agent", {"action": "dance"})]
    assert sim.event_kernel.events == [
        {
            "type": "map_action",
            "agent_id": "agent",
            "step": 0,
            "action": "dance",
            "handled": True,
        }
    ]
