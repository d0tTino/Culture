"""Health signal helpers for council members."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_MIN_PARTICIPATIONS = 10
DEFAULT_WIN_RATE_THRESHOLD = 0.2


def iter_health_warnings(
    members: Iterable[Mapping[str, Any]],
    *,
    min_participations: int = DEFAULT_MIN_PARTICIPATIONS,
    win_rate_threshold: float = DEFAULT_WIN_RATE_THRESHOLD,
) -> Iterable[str]:
    """Yield health warnings for members below win-rate thresholds."""
    for member in members:
        participations = int(member.get("participations", 0) or 0)
        win_rate = float(member.get("win_rate", 0.0) or 0.0)
        if participations >= min_participations and win_rate < win_rate_threshold:
            member_id = member.get("member_id", "unknown")
            yield (
                f"Member {member_id} is a candidate for deactivation "
                f"(win rate {win_rate:.2%} below {win_rate_threshold:.2%} after "
                f"{participations} participations)."
            )


__all__ = [
    "DEFAULT_MIN_PARTICIPATIONS",
    "DEFAULT_WIN_RATE_THRESHOLD",
    "iter_health_warnings",
]
