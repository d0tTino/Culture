from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VersionVector:
    """Simple version vector for optimistic concurrency."""

    clock: dict[str, int] = field(default_factory=dict)

    def increment(self: VersionVector, agent_id: str) -> None:
        self.clock[agent_id] = self.clock.get(agent_id, 0) + 1

    def merge(self: VersionVector, other: VersionVector) -> None:
        for aid, counter in other.clock.items():
            self.clock[aid] = max(self.clock.get(aid, 0), counter)

    def compare(self: VersionVector, other: VersionVector) -> int:
        """Return -1 if self < other, 0 if equal, 1 if self > other, 2 if concurrent."""
        self_greater = False
        other_greater = False
        keys = set(self.clock) | set(other.clock)
        for key in keys:
            a = self.clock.get(key, 0)
            b = other.clock.get(key, 0)
            if a < b:
                other_greater = True
            elif a > b:
                self_greater = True
        if self_greater and other_greater:
            return 2
        if self_greater:
            return 1
        if other_greater:
            return -1
        return 0

    def to_dict(self: VersionVector) -> dict[str, int]:
        return dict(self.clock)
