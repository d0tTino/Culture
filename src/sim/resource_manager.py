from __future__ import annotations

from threading import RLock
from typing import Protocol

from src.infra import event_log
from src.infra.metrics import record_du_budget


class HasResources(Protocol):
    ip: float
    du: float


class BudgetChecker(Protocol):
    def budget_check(self, agent_id: str, amount: float) -> None: ...


class BudgetCharger(Protocol):
    def charge(self, agent_id: str, amount: float) -> float: ...


class TickCapper(Protocol):
    def tick_cap(self, *, ip_start: float, du_start: float, obj: HasResources) -> None: ...


class ResourceManager:
    """Manage per-tick caps and per-agent DU budgets."""

    def __init__(self, max_ip_per_tick: float, max_du_per_tick: float) -> None:
        self.max_ip_per_tick = float(max_ip_per_tick)
        self.max_du_per_tick = float(max_du_per_tick)
        self._du_budgets: dict[str, float] = {}
        self._budget_lock = RLock()

    def tick_cap(self, *, ip_start: float, du_start: float, obj: HasResources) -> None:
        """Clamp the object's IP and DU gains for the current tick."""
        ip_gain = obj.ip - ip_start
        if ip_gain > self.max_ip_per_tick:
            obj.ip = ip_start + self.max_ip_per_tick
        du_gain = obj.du - du_start
        if du_gain > self.max_du_per_tick:
            obj.du = du_start + self.max_du_per_tick

    def cap_tick(self, *, ip_start: float, du_start: float, obj: HasResources) -> None:
        """Backward-compatible alias for :meth:`tick_cap`."""
        self.tick_cap(ip_start=ip_start, du_start=du_start, obj=obj)

    def set_du_budget(self, agent_id: str, budget: float) -> None:
        remaining = float(budget)
        with self._budget_lock:
            self._du_budgets[agent_id] = remaining
        record_du_budget(agent_id, remaining)

    def has_du_budget(self, agent_id: str) -> bool:
        """Return whether an explicit DU budget exists for ``agent_id``."""

        with self._budget_lock:
            return agent_id in self._du_budgets

    def budget_check(self, agent_id: str, amount: float) -> None:
        """Verify that the agent has at least ``amount`` DU available."""
        with self._budget_lock:
            remaining = self._du_budgets.get(agent_id, 0.0)
            if amount <= remaining:
                return
            self._du_budgets[agent_id] = 0.0

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
        raise RuntimeError(f"Agent {agent_id} exceeded DU budget")

    def ensure_du_budget(self, agent_id: str, amount: float) -> None:
        """Backward-compatible alias for :meth:`budget_check`."""
        self.budget_check(agent_id, amount)

    def charge(self, agent_id: str, amount: float) -> float:
        """Charge DU from an agent budget and return remaining budget."""
        with self._budget_lock:
            remaining = self._du_budgets.get(agent_id, 0.0)
            if amount > remaining:
                self._du_budgets[agent_id] = 0.0
                exceeded = True
                updated = 0.0
            else:
                updated = max(remaining - amount, 0.0)
                self._du_budgets[agent_id] = updated
                exceeded = False

        if exceeded:
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
            raise RuntimeError(f"Agent {agent_id} exceeded DU budget")

        record_du_budget(agent_id, updated)
        return updated

    def charge_du(self, agent_id: str, amount: float) -> None:
        """Backward-compatible DU charge API."""
        self.charge(agent_id, amount)

    def get_du_budget(self, agent_id: str) -> float:
        with self._budget_lock:
            return float(self._du_budgets.get(agent_id, 0.0))

    def reserve_du_budget(self, agent_id: str, amount: float, *, reason: str = "du_reserve") -> float:
        """Reserve DU for ``agent_id`` and record the debit in the ledger."""

        reserve_amount = float(max(amount, 0.0))
        if reserve_amount <= 0:
            return self.get_du_budget(agent_id)

        if (not self.has_du_budget(agent_id)) or self.get_du_budget(agent_id) < reserve_amount:
            self.set_du_budget(agent_id, reserve_amount)

        remaining = self.charge(agent_id, reserve_amount)

        try:
            from src.infra.ledger import ledger

            ledger.log_change(agent_id, 0.0, -reserve_amount, reason)
        except Exception:  # pragma: no cover - optional logging
            pass
        return remaining


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


def get_budget_checker() -> BudgetChecker:
    """Return the canonical budget-check interface used by action/LLM paths."""

    return get_resource_manager()


def get_budget_charger() -> BudgetCharger:
    """Return the canonical charge interface used by action/LLM paths."""

    return get_resource_manager()


def get_tick_capper() -> TickCapper:
    """Return the canonical tick-cap interface used by simulation tick updates."""

    return get_resource_manager()
