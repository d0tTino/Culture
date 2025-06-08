from __future__ import annotations

import os

import uvicorn

from src.interfaces.dashboard_backend import app


def main() -> None:
    """Run the FastAPI application for the dashboard backend."""
    host = os.getenv("HTTP_HOST", "0.0.0.0")
    port_str = os.getenv("HTTP_PORT", "8000")
    try:
        port = int(port_str)
    except ValueError as exc:  # pragma: no cover - simple runtime safeguard
        raise ValueError(f"Invalid HTTP_PORT value: {port_str}") from exc
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
