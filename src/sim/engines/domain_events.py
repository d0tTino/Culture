from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DomainEvent:
    domain: str
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
