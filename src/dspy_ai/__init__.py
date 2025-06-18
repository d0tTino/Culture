"""Compatibility wrapper for the dspy package."""

from __future__ import annotations

try:
    import dspy as _dspy
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError("dspy_ai requires the 'dspy' package to be installed") from exc

# Re-export everything from dspy
from dspy import *  # noqa: F403

__all__ = [name for name in dir(_dspy) if not name.startswith("_")]
