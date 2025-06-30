from __future__ import annotations

from typing import Protocol

# Skip self argument annotation warnings for simple methods


class HasResources(Protocol):
    ip: float
    du: float


class ResourceManager:
    """Utility to cap per-tick resource accumulation."""

    def __init__(self, max_ip_per_tick: float, max_du_per_tick: float) -> None:
        self.max_ip_per_tick = float(max_ip_per_tick)
        self.max_du_per_tick = float(max_du_per_tick)

    def cap_tick(self, *, ip_start: float, du_start: float, obj: HasResources) -> None:
        """Clamp the object's IP and DU gains for the current tick."""
        ip_gain = obj.ip - ip_start
        if ip_gain > self.max_ip_per_tick:
            obj.ip = ip_start + self.max_ip_per_tick
        du_gain = obj.du - du_start
        if du_gain > self.max_du_per_tick:
            obj.du = du_start + self.max_du_per_tick
