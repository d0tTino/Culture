"""Compatibility re-export for canonical resource-management APIs.

Use ``src.sim.resource_manager`` as the source of truth.
"""

from src.sim.resource_manager import (
    BudgetCharger,
    BudgetChecker,
    HasResources,
    ResourceManager,
    TickCapper,
    get_budget_charger,
    get_budget_checker,
    get_resource_manager,
    get_tick_capper,
)

__all__ = [
    "BudgetCharger",
    "BudgetChecker",
    "HasResources",
    "ResourceManager",
    "TickCapper",
    "get_budget_charger",
    "get_budget_checker",
    "get_resource_manager",
    "get_tick_capper",
]
