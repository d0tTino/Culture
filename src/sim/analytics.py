
"""User-value analytics for simulation event streams and knowledge board data."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

USER_VALUE_KPI_PAYLOAD_VERSION = 2
STRATEGIC_ADHERENCE_PAYLOAD_VERSION = 1


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


@dataclass(frozen=True, slots=True)
class StrategicPillarThreshold:
    """Pass/fail bands for a strategic pillar score."""

    warning_floor: float
    pass_floor: float
    critical: bool = False

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "warning_floor": self.warning_floor,
            "pass_floor": self.pass_floor,
            "critical": self.critical,
        }


@dataclass(frozen=True, slots=True)
class StrategicAdherenceThresholds:
    """Pass/fail configuration for strategic adherence evaluation."""

    aggregate_regression_tolerance: float = 0.0
    pillars: dict[str, StrategicPillarThreshold] | None = None

    def __post_init__(self) -> None:
        if self.pillars is None:
            object.__setattr__(
                self,
                'pillars',
                {
                    'social_emergence': StrategicPillarThreshold(
                        warning_floor=0.65, pass_floor=0.75, critical=True
                    ),
                    'user_intervention': StrategicPillarThreshold(
                        warning_floor=0.65, pass_floor=0.75, critical=False
                    ),
                    'governance': StrategicPillarThreshold(
                        warning_floor=0.7, pass_floor=0.8, critical=True
                    ),
                    'long_run_persistence': StrategicPillarThreshold(
                        warning_floor=0.7, pass_floor=0.8, critical=True
                    ),
                },
            )

    def as_dict(self) -> dict[str, Any]:
        pillars = self.pillars or {}
        return {
            'aggregate_regression_tolerance': self.aggregate_regression_tolerance,
            'pillars': {name: threshold.as_dict() for name, threshold in pillars.items()},
        }


@dataclass(frozen=True, slots=True)
class StrategicPillarScore:
    """A scored strategic pillar with band evaluation metadata."""

    name: str
    score: float
    status: str
    threshold: StrategicPillarThreshold
    metrics: dict[str, float | int]

    def as_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'score': self.score,
            'status': self.status,
            'threshold': self.threshold.as_dict(),
            'metrics': dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class StrategicAdherenceScorecard:
    """Aggregate strategic adherence payload for dashboards and CI gating."""

    payload_version: int
    user_value_kpis: UserValueKPIReport
    additional_metrics: dict[str, float]
    pillar_scores: tuple[StrategicPillarScore, ...]
    aggregate_adherence_score: float
    aggregate_status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            'payload_version': self.payload_version,
            'user_value_kpis': self.user_value_kpis.as_dict(),
            'additional_metrics': dict(self.additional_metrics),
            'pillar_scores': [pillar.as_dict() for pillar in self.pillar_scores],
            'aggregate_adherence_score': self.aggregate_adherence_score,
            'aggregate_status': self.aggregate_status,
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


@dataclass(frozen=True, slots=True)
class StrategicReleaseGateResult:
    """Release gate outcome comparing a current scorecard to a baseline."""

    passed: bool
    aggregate_delta: float
    critical_pillar_regressions: tuple[str, ...]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            'passed': self.passed,
            'aggregate_delta': self.aggregate_delta,
            'critical_pillar_regressions': list(self.critical_pillar_regressions),
            'reasons': list(self.reasons),
        }


def _as_step(entry: Mapping[str, Any]) -> int:
    try:
        return int(entry.get("step", 0))
    except Exception:
        return 0


def _bounded_ratio(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 4)


def _safe_mean(values: Iterable[float]) -> float:
    items = [value for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _step_distance(event: Mapping[str, Any], other: Mapping[str, Any]) -> int:
    return abs(_as_step(event) - _as_step(other))


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


def compute_strategic_adherence_scorecard(
    *,
    events: Iterable[Mapping[str, Any]],
    knowledge_entries: Iterable[Mapping[str, Any]],
    thresholds: StrategicAdherenceThresholds | None = None,
) -> StrategicAdherenceScorecard:
    """Compute strategic adherence scores for dashboarding and release gates."""

    cfg = thresholds or StrategicAdherenceThresholds()
    event_rows = [dict(event) for event in events]
    entries = [dict(entry) for entry in knowledge_entries]
    user_value_kpis = compute_user_value_kpis(events=event_rows, knowledge_entries=entries)

    latency_values = [float(event['latency_ms']) for event in event_rows if 'latency_ms' in event]
    latency_score = _bounded_ratio(1.0 - (_safe_mean(latency_values) / 1000.0)) if latency_values else 0.0

    recovery_events = [event for event in event_rows if 'recovery_success' in event]
    recovery_success = _bounded_ratio(
        _safe_mean(float(bool(event.get('recovery_success'))) for event in recovery_events)
    ) if recovery_events else 0.0

    command_events = [event for event in event_rows if str(event.get('type')) == 'human_command']
    command_responses = 0
    for command in command_events:
        if any(
            str(event.get('type')) in {'command_ack', 'agent_action', 'governance_vote'}
            and _step_distance(command, event) <= 1
            for event in event_rows
            if event is not command
        ):
            command_responses += 1
    command_responsiveness = _bounded_ratio(
        command_responses / len(command_events) if command_events else 0.0
    )

    memory_scores = [
        float(event['memory_recall_quality'])
        for event in event_rows
        if 'memory_recall_quality' in event
    ] + [
        float(entry['memory_recall_quality'])
        for entry in entries
        if 'memory_recall_quality' in entry
    ]
    memory_recall_quality = _bounded_ratio(_safe_mean(memory_scores)) if memory_scores else 0.0

    additional_metrics = {
        'latency_score': latency_score,
        'recovery_success': recovery_success,
        'command_responsiveness': command_responsiveness,
        'memory_recall_quality': memory_recall_quality,
    }

    pillar_inputs = {
        'social_emergence': {
            'interaction_diversity': user_value_kpis.cross_agent_interaction_diversity,
            'novelty_score': user_value_kpis.novelty_score,
            'social_graph_change_score': _bounded_ratio(user_value_kpis.social_graph_change_count / 3.0),
        },
        'user_intervention': {
            'user_intervention_coverage': _bounded_ratio(user_value_kpis.user_intervention_rate * 4.0),
            'command_responsiveness': command_responsiveness,
            'latency_score': latency_score,
        },
        'governance': {
            'conflict_resolution_score': _bounded_ratio(1.0 / (1 + user_value_kpis.unresolved_conflict_count)),
            'recovery_success': recovery_success,
            'narrative_continuity_score': user_value_kpis.narrative_continuity_score,
        },
        'long_run_persistence': {
            'return_session_continuity': user_value_kpis.return_session_continuity,
            'memory_recall_quality': memory_recall_quality,
            'narrative_continuity_score': user_value_kpis.narrative_continuity_score,
        },
    }

    pillar_scores: list[StrategicPillarScore] = []
    for pillar_name, metrics in pillar_inputs.items():
        threshold = (cfg.pillars or {})[pillar_name]
        score = _bounded_ratio(_safe_mean(float(value) for value in metrics.values()))
        if score >= threshold.pass_floor:
            status = 'pass'
        elif score >= threshold.warning_floor:
            status = 'warning'
        else:
            status = 'fail'
        pillar_scores.append(
            StrategicPillarScore(
                name=pillar_name,
                score=score,
                status=status,
                threshold=threshold,
                metrics={key: round(float(value), 4) for key, value in metrics.items()},
            )
        )

    aggregate_adherence_score = _bounded_ratio(
        _safe_mean(pillar.score for pillar in pillar_scores)
    )
    aggregate_status = 'pass' if all(p.status == 'pass' for p in pillar_scores) else (
        'fail' if any(p.status == 'fail' for p in pillar_scores) else 'warning'
    )
    return StrategicAdherenceScorecard(
        payload_version=STRATEGIC_ADHERENCE_PAYLOAD_VERSION,
        user_value_kpis=user_value_kpis,
        additional_metrics=additional_metrics,
        pillar_scores=tuple(pillar_scores),
        aggregate_adherence_score=aggregate_adherence_score,
        aggregate_status=aggregate_status,
    )


def evaluate_release_gate(
    current: StrategicAdherenceScorecard,
    baseline: StrategicAdherenceScorecard,
    *,
    thresholds: StrategicAdherenceThresholds | None = None,
) -> StrategicReleaseGateResult:
    """Gate releases on aggregate regression and critical-pillar drops."""

    cfg = thresholds or StrategicAdherenceThresholds()
    aggregate_delta = round(
        current.aggregate_adherence_score - baseline.aggregate_adherence_score,
        4,
    )
    current_pillars = {pillar.name: pillar for pillar in current.pillar_scores}
    baseline_pillars = {pillar.name: pillar for pillar in baseline.pillar_scores}
    critical_regressions: list[str] = []
    for name, threshold in (cfg.pillars or {}).items():
        if not threshold.critical:
            continue
        current_pillar = current_pillars.get(name)
        baseline_pillar = baseline_pillars.get(name)
        if current_pillar is None or baseline_pillar is None:
            continue
        if current_pillar.score < baseline_pillar.score:
            critical_regressions.append(name)

    reasons: list[str] = []
    if aggregate_delta < -cfg.aggregate_regression_tolerance:
        reasons.append('aggregate_score_regressed')
    if critical_regressions:
        reasons.append('critical_pillar_drop')
    return StrategicReleaseGateResult(
        passed=not reasons,
        aggregate_delta=aggregate_delta,
        critical_pillar_regressions=tuple(critical_regressions),
        reasons=tuple(reasons),
    )


def strategic_adherence_scorecard_from_dict(payload: Mapping[str, Any]) -> StrategicAdherenceScorecard:
    """Rehydrate a scorecard from JSON payload data."""

    user_payload = payload.get('user_value_kpis', {})
    thresholds_payload = user_payload.get('thresholds', {})
    user_value = UserValueKPIReport(
        payload_version=int(user_payload.get('payload_version', USER_VALUE_KPI_PAYLOAD_VERSION)),
        thresholds=SimulationStagnationThresholds(**thresholds_payload),
        narrative_continuity_score=float(user_payload.get('narrative_continuity_score', 0.0)),
        unresolved_conflict_count=int(user_payload.get('unresolved_conflict_count', 0)),
        cross_agent_interaction_diversity=float(user_payload.get('cross_agent_interaction_diversity', 0.0)),
        user_intervention_rate=float(user_payload.get('user_intervention_rate', 0.0)),
        return_session_continuity=float(user_payload.get('return_session_continuity', 0.0)),
        novelty_score=float(user_payload.get('novelty_score', 0.0)),
        repetitive_intents_ratio=float(user_payload.get('repetitive_intents_ratio', 0.0)),
        social_graph_change_count=int(user_payload.get('social_graph_change_count', 0)),
        stagnation_alerts=tuple(str(value) for value in user_payload.get('stagnation_alerts', [])),
    )
    pillars: list[StrategicPillarScore] = []
    for pillar in payload.get('pillar_scores', []):
        threshold_payload = pillar.get('threshold', {})
        pillars.append(
            StrategicPillarScore(
                name=str(pillar.get('name', '')),
                score=float(pillar.get('score', 0.0)),
                status=str(pillar.get('status', 'fail')),
                threshold=StrategicPillarThreshold(
                    warning_floor=float(threshold_payload.get('warning_floor', 0.0)),
                    pass_floor=float(threshold_payload.get('pass_floor', 0.0)),
                    critical=bool(threshold_payload.get('critical', False)),
                ),
                metrics={
                    str(key): float(value)
                    for key, value in dict(pillar.get('metrics', {})).items()
                },
            )
        )
    return StrategicAdherenceScorecard(
        payload_version=int(payload.get('payload_version', STRATEGIC_ADHERENCE_PAYLOAD_VERSION)),
        user_value_kpis=user_value,
        additional_metrics={
            str(key): float(value)
            for key, value in dict(payload.get('additional_metrics', {})).items()
        },
        pillar_scores=tuple(pillars),
        aggregate_adherence_score=float(payload.get('aggregate_adherence_score', 0.0)),
        aggregate_status=str(payload.get('aggregate_status', 'fail')),
    )


__all__ = [
    'STRATEGIC_ADHERENCE_PAYLOAD_VERSION',
    'USER_VALUE_KPI_PAYLOAD_VERSION',
    'SimulationStagnationThresholds',
    'StrategicAdherenceScorecard',
    'StrategicAdherenceThresholds',
    'StrategicPillarScore',
    'StrategicPillarThreshold',
    'StrategicReleaseGateResult',
    'UserValueKPIReport',
    'compute_strategic_adherence_scorecard',
    'compute_user_value_kpis',
    'evaluate_release_gate',
    'strategic_adherence_scorecard_from_dict',
]
