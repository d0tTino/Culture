"""Compatibility wrapper for the dspy package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dspy as _dspy
else:
    try:
        import dspy as _dspy
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        from src.infra.dspy_ollama_integration import dspy as _dspy

__all__ = [name for name in dir(_dspy) if not name.startswith("_")]
globals().update({name: getattr(_dspy, name) for name in __all__})
