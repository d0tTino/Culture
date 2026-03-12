from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .personality_transition import PersonalityTransition, TraitTransitionLog


def _coerce_transition(entry: Mapping[str, Any] | PersonalityTransition) -> PersonalityTransition:
    if isinstance(entry, PersonalityTransition):
        return entry
    if hasattr(PersonalityTransition, "model_validate"):
        return PersonalityTransition.model_validate(entry)
    return PersonalityTransition.parse_obj(entry)


def personality_timeline(
    trait_log: TraitTransitionLog,
    lifecycle_history: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a merged, ordered timeline of personality and lifecycle transitions."""

    timeline: list[dict[str, Any]] = []
    for idx, transition in enumerate(trait_log.transitions):
        timeline.append(
            {
                "kind": "personality_transition",
                "step": int(transition.step),
                "index": idx,
                "hash": trait_log.hash_chain[idx] if idx < len(trait_log.hash_chain) else None,
                "cause": transition.cause,
                "source": transition.source,
                "deltas": [
                    {
                        "trait": delta.trait,
                        "before": float(delta.before),
                        "after": float(delta.after),
                        "delta": float(delta.bounded_delta),
                    }
                    for delta in transition.deltas
                ],
                "input_signals": dict(transition.input_signals),
            }
        )

    for item in lifecycle_history or []:
        timeline.append(
            {
                "kind": "lifecycle_transition",
                "step": int(item.get("step", 0)),
                "from": str(item.get("from", "")),
                "to": str(item.get("to", "")),
                "reason": str(item.get("reason", "")),
            }
        )

    timeline.sort(key=lambda item: (int(item.get("step", 0)), str(item.get("kind", ""))))
    return timeline


def build_character_arc_summaries(
    trait_log: TraitTransitionLog,
    *,
    lifecycle_history: Sequence[Mapping[str, Any]] | None = None,
    window_size: int = 20,
) -> list[dict[str, Any]]:
    """Generate periodic summaries over transition windows."""

    transitions = list(trait_log.transitions)
    if not transitions:
        return []

    summaries: list[dict[str, Any]] = []
    for start in range(0, len(transitions), max(1, window_size)):
        window = transitions[start : start + max(1, window_size)]
        if not window:
            continue
        deltas_by_trait: dict[str, float] = defaultdict(float)
        social_events: dict[str, int] = defaultdict(int)
        for transition in window:
            for delta in transition.deltas:
                deltas_by_trait[delta.trait] += float(delta.bounded_delta)
            social_signal = float(transition.input_signals.get("social_outcome", 0.0))
            if social_signal > 0.15:
                social_events["positive_social_outcomes"] += 1
            if social_signal < -0.15:
                social_events["negative_social_outcomes"] += 1
            if float(transition.input_signals.get("conflict_outcome", 0.0)) < -0.2:
                social_events["conflict_events"] += 1

        top_changes = sorted(deltas_by_trait.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        if not top_changes:
            continue
        headline_parts = [
            f"{trait} {'increased' if value >= 0 else 'decreased'} by {abs(value):.3f}"
            for trait, value in top_changes
        ]

        step_start = int(window[0].step)
        step_end = int(window[-1].step)
        lifecycle_events = [
            event
            for event in (lifecycle_history or [])
            if step_start <= int(event.get("step", 0)) <= step_end
        ]

        summaries.append(
            {
                "window_start_step": step_start,
                "window_end_step": step_end,
                "top_trait_changes": [
                    {"trait": trait, "delta": float(value)} for trait, value in top_changes
                ],
                "social_event_counts": dict(social_events),
                "lifecycle_events": [dict(event) for event in lifecycle_events],
                "summary": "; ".join(headline_parts),
            }
        )

    return summaries


def top_trait_changes(
    transitions: Sequence[Mapping[str, Any] | PersonalityTransition],
    *,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Return top absolute trait movements with source/cause breadcrumbs."""

    movements: list[dict[str, Any]] = []
    for index, entry in enumerate(transitions):
        transition = _coerce_transition(entry)
        for delta in transition.deltas:
            movements.append(
                {
                    "trait": delta.trait,
                    "delta": float(delta.bounded_delta),
                    "before": float(delta.before),
                    "after": float(delta.after),
                    "step": int(transition.step),
                    "cause": transition.cause,
                    "source": transition.source,
                    "transition_index": index,
                }
            )
    ranked = sorted(movements, key=lambda item: abs(float(item["delta"])), reverse=True)
    return ranked[: max(1, top_k)]


def smooth_trait_deltas(
    deltas: Mapping[str, float],
    *,
    recent_transitions: Sequence[Mapping[str, Any] | PersonalityTransition],
    window_size: int,
    alpha: float,
    hysteresis_threshold: float,
) -> dict[str, float]:
    """Apply smoothing + hysteresis against short sign-flip oscillations."""

    bounded_alpha = min(1.0, max(0.0, float(alpha)))
    threshold = max(0.0, float(hysteresis_threshold))
    if window_size <= 0:
        return {str(k): float(v) for k, v in deltas.items()}

    recent = list(recent_transitions)[-window_size:]
    history: dict[str, list[float]] = defaultdict(list)
    for entry in recent:
        transition = _coerce_transition(entry)
        for delta in transition.deltas:
            history[delta.trait].append(float(delta.bounded_delta))

    smoothed: dict[str, float] = {}
    for trait, proposed in deltas.items():
        value = float(proposed)
        hist = history.get(str(trait), [])
        if not hist:
            smoothed[str(trait)] = value
            continue
        avg = sum(hist) / len(hist)
        candidate = bounded_alpha * value + (1.0 - bounded_alpha) * avg
        if avg * value < 0 and abs(value) < threshold:
            candidate = 0.0
        smoothed[str(trait)] = candidate

    return smoothed


def discord_trait_shift_message(
    *,
    agent_name: str,
    summary: Mapping[str, Any],
) -> str:
    """Render a concise Discord update for notable personality drift."""

    changes = list(summary.get("top_trait_changes", []))
    if not changes:
        return ""
    strongest = changes[0]
    trait = str(strongest.get("trait", "trait"))
    delta = float(strongest.get("delta", 0.0))
    direction = "more" if delta >= 0 else "less"
    collaborative_hint = "collaborative" if trait in {"empathy", "trust_baseline"} and delta >= 0 else trait
    social = summary.get("social_event_counts", {})
    positive = int(dict(social).get("positive_social_outcomes", 0))
    cause = "after repeated positive outcomes" if positive > 0 else "as recent events accumulated"
    return (
        f"Agent {agent_name} has become {direction} {collaborative_hint} {cause} "
        f"(window {summary.get('window_start_step')}→{summary.get('window_end_step')})."
    )
