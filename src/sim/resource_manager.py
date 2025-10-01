from __future__ import annotations

from typing import Protocol

from src.infra import event_log
from src.infra.metrics import record_du_budget


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
        remaining = float(budget)
        self._du_budgets[agent_id] = remaining
        record_du_budget(agent_id, remaining)

    def ensure_du_budget(self, agent_id: str, amount: float) -> None:
        """Verify that the agent has at least ``amount`` DU available."""
        remaining = self._du_budgets.get(agent_id, 0.0)
        if amount > remaining:
            try:
                event_log.log_event(
                    {
                        "type": "du_budget_exceeded",
                        "agent": agent_id,
                        "required": float(amount),
                        "remaining": float(remaining),
                    }
                )
            except Exception:  # pragma: no cover - best effort
                pass
            try:
                from src.interfaces.discord_bot import notify_budget_exceeded

                notify_budget_exceeded(agent_id, amount, remaining)
            except Exception:  # pragma: no cover - best effort
                pass
            self._du_budgets[agent_id] = 0.0
            raise RuntimeError(f"Agent {agent_id} exceeded DU budget")

    def charge_du(self, agent_id: str, amount: float) -> None:
        self.ensure_du_budget(agent_id, amount)
        remaining = self._du_budgets.get(agent_id, 0.0)
        updated = max(remaining - amount, 0.0)
        self._du_budgets[agent_id] = updated
        record_du_budget(agent_id, updated)

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
