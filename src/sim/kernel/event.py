"""Event dataclass for discrete event simulation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass(order=True)
class Event:
    """A scheduled callback executed at a given timestamp."""

    ts: int
    seq: int
    callback: Callable[[], Awaitable[None]] = field(compare=False)
