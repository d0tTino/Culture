from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .event_bus import get_event_bus


@dataclass
class SimulationContext:
    """Container for cross-module simulation state."""

    sim_state: dict[str, Any] = field(
        default_factory=lambda: {
            "paused": False,
            "speed": 1.0,
            "semantic_manager": None,
            "simulation": None,
            "discord_bot": None,
        }
    )
    message_queue: asyncio.Queue[Any] = field(default_factory=lambda: asyncio.Queue(maxsize=1000))
    _event_queue: asyncio.Queue[Any] | None = field(default=None, init=False)
    _event_queue_loop: asyncio.AbstractEventLoop | None = field(default=None, init=False)

    def get_event_queue(self: SimulationContext) -> asyncio.Queue[Any]:
        """Return an event queue bound to the current loop."""
        bus = get_event_bus()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        if self._event_queue is None or self._event_queue_loop is not loop:
            if self._event_queue is not None:
                bus.unsubscribe(self._event_queue)
            self._event_queue = bus.subscribe()
            self._event_queue_loop = loop
        return self._event_queue
