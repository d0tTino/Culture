from __future__ import annotations

import argparse
import os

import uvicorn

from src.interfaces.dashboard_backend import app


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

    host = os.getenv("HTTP_HOST", "0.0.0.0")
    port_str = os.getenv("HTTP_PORT", "8000")
    try:
        port = int(port_str)
    except ValueError as exc:  # pragma: no cover - simple runtime safeguard
        raise ValueError(f"Invalid HTTP_PORT value: {port_str}") from exc
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
