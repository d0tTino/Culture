from __future__ import annotations

from src.sim.population_service import LifecycleTransitionResult, PopulationService


class LifecycleService(PopulationService):
    """Backward-compatible alias for population lifecycle domain service."""


__all__ = ["LifecycleService", "LifecycleTransitionResult"]
