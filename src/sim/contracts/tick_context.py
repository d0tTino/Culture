from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class TickContext:
    """Immutable cross-domain context for one simulation tick."""

    step: int
    world_time: Mapping[str, Any]
    weather: str
    governance_state: Mapping[str, Any]
    world_modifiers: Mapping[str, Any]
    replay_metadata: Mapping[str, Any]

    @staticmethod
    def freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
        return MappingProxyType(dict(value or {}))
