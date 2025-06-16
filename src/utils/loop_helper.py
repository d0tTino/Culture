"""Event loop configuration utilities."""

from __future__ import annotations

import asyncio
import logging
import os

_logger = logging.getLogger(__name__)


def use_uvloop_if_available() -> None:
    """Try to set ``uvloop`` as the event loop when not running on Windows."""
    if os.name == "nt":
        _logger.debug("uvloop skipped on Windows")
        return
    try:
        import uvloop

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        _logger.debug("uvloop enabled")
    except Exception as exc:  # pragma: no cover - best effort
        _logger.debug("uvloop not available: %s", exc)
