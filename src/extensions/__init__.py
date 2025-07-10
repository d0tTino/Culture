"""Hooks for extending Culture via external packages."""

from __future__ import annotations

from typing import Any, Callable

import httpx


class BehaviorRegistry:
    """Registry for callables that modify agent behavior."""

    def __init__(self) -> None:
        self._behaviors: list[Callable[[Any, dict[str, Any]], None]] = []

    def register(self, func: Callable[[Any, dict[str, Any]], None]) -> None:
        self._behaviors.append(func)

    def run(self, agent: Any, output: dict[str, Any]) -> None:
        for func in list(self._behaviors):
            func(agent, output)


BEHAVIOR_REGISTRY = BehaviorRegistry()


def register_agent_behavior(func: Callable[[Any, dict[str, Any]], None]) -> None:
    """Register a callback executed after each agent turn."""

    BEHAVIOR_REGISTRY.register(func)


def register_widget_backend(
    name: str,
    script_url: str,
    backend_url: str = "http://localhost:8000",
) -> None:
    """Register a UI widget with the Culture backend."""

    resp = httpx.post(
        f"{backend_url.rstrip('/')}/api/register_widget",
        json={"name": name, "script_url": script_url},
        timeout=10,
    )
    resp.raise_for_status()
