"""Hooks for extending Culture via external packages."""

from __future__ import annotations

import asyncio
import logging
from importlib import metadata
from typing import Any, Callable

import httpx
from typing_extensions import Self

logger = logging.getLogger(__name__)

__all__ = [
    "BEHAVIOR_REGISTRY",
    "load_plugins",
    "register_agent_behavior",
    "register_widget_backend",
]


class BehaviorRegistry:
    """Registry for callables that modify agent behavior."""

    def __init__(self: Self) -> None:
        self._behaviors: list[Callable[[Any, dict[str, Any]], None]] = []

    def register(self: Self, func: Callable[[Any, dict[str, Any]], None]) -> None:
        self._behaviors.append(func)

    def run(self: Self, agent: Any, output: dict[str, Any]) -> None:
        for func in list(self._behaviors):
            func(agent, output)


BEHAVIOR_REGISTRY = BehaviorRegistry()


def register_agent_behavior(func: Callable[[Any, dict[str, Any]], None]) -> None:
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
            plugin = ep.load()
            result = plugin()
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
