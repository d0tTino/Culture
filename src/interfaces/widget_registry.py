from __future__ import annotations

from typing import Any


class WidgetRegistry:
    """Simple registry for dashboard widgets."""

    def __init__(self) -> None:
        self._widgets: dict[str, dict[str, Any]] = {}

    def register(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        self._widgets[name] = metadata or {}

    def get(self, name: str) -> dict[str, Any] | None:
        return self._widgets.get(name)

    def list(self) -> list[dict[str, Any]]:
        return [{"name": n, **meta} for n, meta in sorted(self._widgets.items())]


__all__ = ["WidgetRegistry"]
