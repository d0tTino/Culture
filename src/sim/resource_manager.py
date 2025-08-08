from __future__ import annotations

from typing import Protocol


class HasResources(Protocol):
    ip: float
    du: float


class ResourceManager:
    """Manage per-tick caps and per-agent DU budgets."""

    def __init__(self, max_ip_per_tick: float, max_du_per_tick: float) -> None:
        self.max_ip_per_tick = float(max_ip_per_tick)
        self.max_du_per_tick = float(max_du_per_tick)
        self._du_budgets: dict[str, float] = {}

    def cap_tick(self, *, ip_start: float, du_start: float, obj: HasResources) -> None:
        """Clamp the object's IP and DU gains for the current tick."""
        ip_gain = obj.ip - ip_start
        if ip_gain > self.max_ip_per_tick:
            obj.ip = ip_start + self.max_ip_per_tick
        du_gain = obj.du - du_start
        if du_gain > self.max_du_per_tick:
            obj.du = du_start + self.max_du_per_tick

    def set_du_budget(self, agent_id: str, budget: float) -> None:
        self._du_budgets[agent_id] = float(budget)

    def charge_du(self, agent_id: str, amount: float) -> None:
        remaining = self._du_budgets.get(agent_id, 0.0)
        if amount > remaining:
            raise RuntimeError(f"Agent {agent_id} exceeded DU budget")
        self._du_budgets[agent_id] = remaining - amount

    def get_du_budget(self, agent_id: str) -> float:
        return float(self._du_budgets.get(agent_id, 0.0))


_resource_manager: ResourceManager | None = None


def get_resource_manager() -> ResourceManager:
    """Return a singleton instance of :class:`ResourceManager`."""
    global _resource_manager
    if _resource_manager is None:
        from src.infra.config import get_config

        _resource_manager = ResourceManager(
            float(get_config("MAX_IP_PER_TICK")), float(get_config("MAX_DU_PER_TICK"))
        )
    return _resource_manager
