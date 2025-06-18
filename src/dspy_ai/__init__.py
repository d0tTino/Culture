"""Compatibility wrapper for the dspy package."""

from __future__ import annotations

try:
    import dspy as _dspy
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    # Fallback to the local stub implementation used in tests
    from src.infra.dspy_ollama_integration import dspy as _dspy

# Re-export everything from dspy
from dspy import *  # noqa: F403

__all__ = [name for name in dir(_dspy) if not name.startswith("_")]
