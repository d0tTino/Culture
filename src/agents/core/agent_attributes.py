from __future__ import annotations

from dataclasses import dataclass, field

from src.infra import config


@dataclass
class AgentAttributes:
    """Lightweight container for agent state used in simple tests."""

    id: str
    mood: float
    goals: list[str]
    resources: dict[str, object]
    relationships: dict[str, float]
    relationship_momentum: dict[str, float] = field(default_factory=dict)
    unlocked_capabilities: set[str] = field(default_factory=set)
    ip: float = field(default_factory=lambda: float(config.INITIAL_INFLUENCE_POINTS))
    du: float = field(default_factory=lambda: float(config.INITIAL_DATA_UNITS))
