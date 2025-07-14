"""Hooks for extending Culture via external packages."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from importlib import metadata
from typing import Any, Protocol, runtime_checkable

import httpx
from typing_extensions import Self

logger = logging.getLogger(__name__)

__all__ = [
    "BEHAVIOR_REGISTRY",
    "AgentBehavior",
    "Plugin",
    "load_plugins",
    "register_agent_behavior",
    "register_widget_backend",
]


@runtime_checkable
class AgentBehavior(Protocol):
    """Callable invoked after each agent turn."""

    def __call__(self, agent: Any, output: dict[str, Any]) -> None: ...


PluginResult = dict[str, str] | None


@runtime_checkable
class Plugin(Protocol):
    """Callable loaded via entry points."""

    def __call__(self) -> PluginResult | Awaitable[PluginResult]: ...


class BehaviorRegistry:
    """Registry for callables that modify agent behavior."""

    def __init__(self: Self) -> None:
        self._behaviors: list[AgentBehavior] = []

    def register(self: Self, func: AgentBehavior) -> None:
        self._behaviors.append(func)

    def run(self: Self, agent: Any, output: dict[str, Any]) -> None:
        for func in list(self._behaviors):
            func(agent, output)


BEHAVIOR_REGISTRY = BehaviorRegistry()


def register_agent_behavior(func: AgentBehavior) -> None:
    """Register a callback executed after each agent turn."""

    BEHAVIOR_REGISTRY.register(func)


async def register_widget_backend(
    name: str,
    script_url: str,
    backend_url: str = "http://localhost:8000",
) -> None:
    """Register a UI widget with the Culture backend."""

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{backend_url.rstrip('/')}/api/register_widget",
                json={"name": name, "script_url": script_url},
            )
            resp.raise_for_status()
    except httpx.HTTPError as exc:  # pragma: no cover - network failures
        logger.exception("Widget registration failed: %s", exc)


DEFAULT_PLUGIN_GROUP = "culture.plugins"


async def load_plugins(
    group: str = DEFAULT_PLUGIN_GROUP, *, backend_url: str = "http://localhost:8000"
) -> None:
    """Load entry-point plugins and optionally register their widgets."""

    try:
        eps = metadata.entry_points()
    except Exception as exc:  # pragma: no cover - extremely unlikely
        logger.exception("Failed to load entry points: %s", exc)
        return

    for ep in eps.select(group=group):
        try:
            plugin: Plugin = ep.load()
            result: Any = plugin()
            if asyncio.iscoroutine(result):
                result = await result
            elif callable(result):
                result = result()
                if asyncio.iscoroutine(result):
                    result = await result
            if isinstance(result, dict) and {
                "name",
                "script_url",
            }.issubset(result):
                await register_widget_backend(
                    name=result["name"],
                    script_url=result["script_url"],
                    backend_url=backend_url,
                )
        except Exception as exc:  # pragma: no cover - safety net
            logger.exception("Error loading plugin %s: %s", ep.name, exc)
