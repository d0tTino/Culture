from __future__ import annotations

import argparse
from collections.abc import AsyncGenerator
from typing import Any

import uvicorn
from fastapi import Request

from src.infra import config
from src.interfaces.dashboard_backend import (
    EventSourceResponse,
    SimulationEvent,
    app,
    event_queue,
)


async def generate_events(
    request: Request,
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield simulation events from the shared queue."""
    while True:
        if await request.is_disconnected():
            break
        event: SimulationEvent | None = await event_queue.get()
        if event is None:
            break
        yield {"event": "simulation_event", "data": event.json()}


@app.get("/stream/events")
async def stream_events(request: Request) -> EventSourceResponse:
    """Return an SSE stream of simulation events."""
    return EventSourceResponse(generate_events(request))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FastAPI dashboard backend.")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the Culture.ai version and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the FastAPI application for the dashboard backend."""
    args = parse_args(argv or [])

    if args.version:
        from src import __version__

        print(__version__)
        return

    host = str(config.get_config("HTTP_HOST") or "0.0.0.0")
    port_str = str(config.get_config("HTTP_PORT") or "8000")
    try:
        port = int(port_str)
    except ValueError as exc:  # pragma: no cover - simple runtime safeguard
        raise ValueError(f"Invalid HTTP_PORT value: {port_str}") from exc
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
