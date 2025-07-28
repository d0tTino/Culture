"""Widget showing recent semantic summaries."""

from src.extensions import PluginResult


def setup() -> PluginResult:
    """Register the MemoryWidget."""
    return {
        "name": "MemoryWidget",
        "script_url": "https://localhost:5173/memory_widget.js",
    }
