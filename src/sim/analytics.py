"""User-value analytics for simulation event streams and knowledge board data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

USER_VALUE_KPI_PAYLOAD_VERSION = 2


@dataclass(frozen=True, slots=True)
class SimulationStagnationThresholds:
    """Alert limits for detecting simulation stagnation signals."""

    min_novelty_score: float = 0.2
    min_interaction_diversity: float = 0.2
    max_repetitive_intents_ratio: float = 0.75
    min_social_graph_change_count: int = 1

    def as_dict(self) -> dict[str, float | int]:
        return {
            "min_novelty_score": self.min_novelty_score,
            "min_interaction_diversity": self.min_interaction_diversity,
            "max_repetitive_intents_ratio": self.max_repetitive_intents_ratio,
            "min_social_graph_change_count": self.min_social_graph_change_count,
        }


@dataclass(frozen=True, slots=True)
class UserValueKPIReport:
    """Canonical user-value KPI payload consumed by dashboards and summaries."""

    payload_version: int
    thresholds: SimulationStagnationThresholds
    narrative_continuity_score: float
    unresolved_conflict_count: int
    cross_agent_interaction_diversity: float
    user_intervention_rate: float
    return_session_continuity: float
    novelty_score: float
    repetitive_intents_ratio: float
    social_graph_change_count: int
    stagnation_alerts: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stagnation_alerts"] = list(self.stagnation_alerts)
        payload["thresholds"] = self.thresholds.as_dict()
        return payload


def _as_step(entry: Mapping[str, Any]) -> int:
    try:
        return int(entry.get("step", 0))
    except Exception:
        return 0


def compute_user_value_kpis(
    *,
    events: Iterable[Mapping[str, Any]],
    knowledge_entries: Iterable[Mapping[str, Any]],
    thresholds: SimulationStagnationThresholds | None = None,
) -> UserValueKPIReport:
    """Compute user-value KPIs from event logs and board entries."""

    cfg = thresholds or SimulationStagnationThresholds()
    event_rows = [dict(event) for event in events]
    entries = [dict(entry) for entry in knowledge_entries]

    entry_ids = {str(entry.get("entry_id", "")) for entry in entries if entry.get("entry_id")}
    parent_refs = [
        str(entry.get("parent_entry_id", ""))
        for entry in entries
        if isinstance(entry.get("parent_entry_id"), str)
    ]
    continuation_ratio = (
        sum(1 for parent in parent_refs if parent in entry_ids) / len(entries) if entries else 0.0
    )
    event_steps = sorted({_as_step(event) for event in event_rows if _as_step(event) > 0})
    if len(event_steps) > 1:
        contiguous = sum(1 for prev, curr in pairwise(event_steps) if curr - prev <= 1)
        event_continuity = contiguous / (len(event_steps) - 1)
    elif event_steps:
        event_continuity = 1.0
    else:
        event_continuity = 0.0
    narrative_continuity_score = (continuation_ratio + event_continuity) / 2.0

    conflict_entries: dict[str, dict[str, Any]] = {}
    for entry in entries:
        entry_id = str(entry.get("entry_id", ""))
        if not entry_id:
            continue
        tags = {str(tag).lower() for tag in entry.get("tags", [])}
        entry_type = str(entry.get("entry_type", "")).lower()
        summary = str(entry.get("content_summary", "")).lower()
        if "conflict" in tags or "conflict" in entry_type or "conflict" in summary:
            conflict_entries[entry_id] = entry

    resolved_conflicts: set[str] = set()
    for entry in entries:
        parent_id = str(entry.get("parent_entry_id", "") or "")
        if not parent_id or parent_id not in conflict_entries:
            continue
        tags = {str(tag).lower() for tag in entry.get("tags", [])}
        entry_type = str(entry.get("entry_type", "")).lower()
        summary = str(entry.get("content_summary", "")).lower()
        if (
            "resolution" in tags
            or "resolved" in tags
            or "resolve" in entry_type
            or "resolved" in summary
            or "resolution" in summary
        ):
            resolved_conflicts.add(parent_id)
    unresolved_conflict_count = max(0, len(conflict_entries) - len(resolved_conflicts))

    agents = {
        str(event.get("agent_id"))
        for event in event_rows
        if isinstance(event.get("agent_id"), str) and str(event.get("agent_id"))
    }
    interaction_pairs: set[tuple[str, str]] = set()
    for event in event_rows:
        origin = event.get("agent_id")
        if not isinstance(origin, str) or not origin:
            continue
        for field in ("recipient_id", "target_agent_id"):
            target = event.get(field)
            if isinstance(target, str) and target and target != origin:
                interaction_pairs.add((origin, target))
    possible_pairs = max(len(agents) * max(len(agents) - 1, 0), 1)
    cross_agent_interaction_diversity = min(1.0, len(interaction_pairs) / possible_pairs)

    user_interventions = sum(1 for event in event_rows if event.get("type") == "human_command")
    user_intervention_rate = user_interventions / len(event_rows) if event_rows else 0.0

    snapshot_steps = sorted(
        {_as_step(event) for event in event_rows if str(event.get("type")) == "snapshot"}
    )
    if len(snapshot_steps) >= 2:
        resumed_pairs = sum(1 for prev, curr in pairwise(snapshot_steps) if curr > prev)
        return_session_continuity = resumed_pairs / (len(snapshot_steps) - 1)
    elif len(snapshot_steps) == 1:
        return_session_continuity = 1.0
    else:
        return_session_continuity = 0.0

    intent_values = [
        str(event.get("action_intent", "")).strip()
        for event in event_rows
        if isinstance(event.get("action_intent"), str) and str(event.get("action_intent", "")).strip()
    ]
    if intent_values:
        counts = Counter(intent_values)
        most_common = counts.most_common(1)[0][1]
        repetitive_intents_ratio = most_common / len(intent_values)
        novelty_score = len(counts) / len(intent_values)
    else:
        repetitive_intents_ratio = 0.0
        novelty_score = 0.0

    social_graph_change_count = 0
    for event in event_rows:
        payload_blob = " ".join(
            [
                str(event.get("type", "")).lower(),
                str(event.get("action_intent", "")).lower(),
                str(event.get("content", "")).lower(),
            ]
        )
        if any(token in payload_blob for token in ("relationship", "coalition", "ally", "rival")):
            social_graph_change_count += 1

    stagnation_alerts: list[str] = []
    if novelty_score < cfg.min_novelty_score:
        stagnation_alerts.append("low_novelty")
    if cross_agent_interaction_diversity < cfg.min_interaction_diversity:
        stagnation_alerts.append("low_interaction_diversity")
    if repetitive_intents_ratio > cfg.max_repetitive_intents_ratio:
        stagnation_alerts.append("repetitive_intents")
    if social_graph_change_count < cfg.min_social_graph_change_count:
        stagnation_alerts.append("no_social_graph_change")

    return UserValueKPIReport(
        payload_version=USER_VALUE_KPI_PAYLOAD_VERSION,
        thresholds=cfg,
        narrative_continuity_score=round(narrative_continuity_score, 4),
        unresolved_conflict_count=unresolved_conflict_count,
        cross_agent_interaction_diversity=round(cross_agent_interaction_diversity, 4),
        user_intervention_rate=round(user_intervention_rate, 4),
        return_session_continuity=round(return_session_continuity, 4),
        novelty_score=round(novelty_score, 4),
        repetitive_intents_ratio=round(repetitive_intents_ratio, 4),
        social_graph_change_count=social_graph_change_count,
        stagnation_alerts=tuple(stagnation_alerts),
    )
