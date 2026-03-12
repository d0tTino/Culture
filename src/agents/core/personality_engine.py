from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import BaseModel

from src.infra.config import get_config
from src.sim.engines.domain_events import TraitDriftApplied

from .agent_state import AgentState
from .personality_insights import smooth_trait_deltas
from .personality_transition import PersonalityTransition, TraitDelta, TraitTransitionLog
from .trait_policy import (
    action_intent_biasing,
    merge_trait_policy_coefficients,
    mood_update_multiplier,
    normalize_trait_projection,
    relationship_update_sensitivity,
    trait_drift_from_experience,
)


class ExperienceSignal(BaseModel):
    """Structured per-turn signal envelope used for deterministic trait drift."""

    social_outcome: float = 0.0
    conflict_outcome: float = 0.0
    task_outcome: float = 0.0
    mood_trajectory: float = 0.0
    governance_participation: float = 0.0


class PersonalityEngine:
    """Sole owner of personality-dependent strategy and trait drift updates."""

    def _coefficients(self, state: AgentState) -> dict[str, float]:
        coefficients = merge_trait_policy_coefficients(state.trait_policy_coefficients)
        return cast(dict[str, float], coefficients)

    def trait_projection(self, state: AgentState) -> dict[str, dict[str, float]]:
        """Return normalized trait projection consumed by influence policy points."""

        return cast(dict[str, dict[str, float]], normalize_trait_projection(state.traits))

    def _build_transition(
        self,
        state: AgentState,
        deltas: Mapping[str, float],
        *,
        max_step: float,
        cause: str,
        source: str,
        input_signals: Mapping[str, float] | None = None,
    ) -> PersonalityTransition:
        transition_deltas: list[TraitDelta] = []
        for trait, proposed_delta in deltas.items():
            if not hasattr(state.traits, trait):
                continue
            before = float(getattr(state.traits, trait))
            bounded_delta = max(-max_step, min(max_step, float(proposed_delta)))
            after = max(0.0, min(1.0, before + bounded_delta))
            applied_delta = after - before
            if abs(applied_delta) < 1e-12:
                continue
            transition_deltas.append(
                TraitDelta(
                    trait=trait,
                    before=before,
                    proposed_delta=float(proposed_delta),
                    bounded_delta=applied_delta,
                    after=after,
                )
            )

        return PersonalityTransition(
            step=int(state.step_counter),
            cause=cause,
            source=source,
            max_step=float(max_step),
            input_signals={str(k): float(v) for k, v in (input_signals or {}).items()},
            deltas=transition_deltas,
        )

    def _reduce_transition(
        self,
        state: AgentState,
        transition: PersonalityTransition,
    ) -> list[dict[str, float | str | int]]:
        """Apply transition event through a single reducer and persist event history."""

        records: list[dict[str, float | str | int]] = []
        for delta in transition.deltas:
            setattr(state.traits, delta.trait, delta.after)
            record: dict[str, float | str | int] = {
                "step": transition.step,
                "trait": delta.trait,
                "before": delta.before,
                "delta": delta.bounded_delta,
                "after": delta.after,
                "cause": transition.cause,
                "source": transition.source,
            }
            if transition.input_signals:
                record.update(transition.input_signals)
            records.append(record)

        transition.resulting_traits = self.trait_projection(state)["raw"]
        if not state.trait_transition_log.seed_traits:
            state.trait_transition_log.seed_traits = {
                delta.trait: delta.before for delta in transition.deltas
            } or dict(transition.resulting_traits)
        state.trait_transition_log.append(transition)
        dumped = transition.model_dump(mode="python") if hasattr(transition, "model_dump") else transition.dict()
        state.personality_transition_events.append(dumped)
        if records:
            state.trait_change_audit.extend(records)
        return records

    def replay_transitions(
        self,
        initial_traits: Mapping[str, float],
        transitions: Sequence[Mapping[str, object] | PersonalityTransition] | TraitTransitionLog,
    ) -> dict[str, float]:
        """Replay transitions deterministically from a seed trait state."""
        transition_stream: Sequence[Mapping[str, object] | PersonalityTransition]
        if isinstance(transitions, TraitTransitionLog):
            if not transitions.verify_hash_chain():
                raise ValueError("TraitTransitionLog hash chain verification failed")
            transition_stream = transitions.transitions
        else:
            transition_stream = transitions
        replayed = {str(k): float(v) for k, v in initial_traits.items()}
        for entry in transition_stream:
            transition = (
                entry
                if isinstance(entry, PersonalityTransition)
                else (
                    PersonalityTransition.model_validate(entry)
                    if hasattr(PersonalityTransition, "model_validate")
                    else PersonalityTransition.parse_obj(entry)
                )
            )
            for delta in transition.deltas:
                replayed[delta.trait] = float(delta.after)
        return replayed

    def action_biases(self, state: AgentState, available_actions: Sequence[str]) -> dict[str, float]:
        return cast(
            dict[str, float],
            action_intent_biasing(
                self.trait_projection(state),
                available_actions,
                self._coefficients(state),
            ),
        )

    def mood_multiplier(self, state: AgentState) -> float:
        return float(mood_update_multiplier(self.trait_projection(state), self._coefficients(state)))

    def relationship_sensitivity(self, state: AgentState, *, is_targeted: bool) -> float:
        return float(
            relationship_update_sensitivity(
                self.trait_projection(state),
                is_targeted=is_targeted,
                coefficients=self._coefficients(state),
            )
        )

    def reduce_trait_drift_event(
        self,
        state: AgentState,
        event: TraitDriftApplied,
    ) -> list[dict[str, float | str | int]]:
        """Apply trait drift exclusively from a domain event payload."""

        transition = self._build_transition(
            state,
            event.deltas,
            max_step=event.max_step,
            cause=event.cause,
            source=event.source,
            input_signals=event.input_signals,
        )
        return self._reduce_transition(state, transition)

    def build_experience_drift_event(
        self,
        state: AgentState,
        signals: ExperienceSignal,
        *,
        max_step: float = 0.01,
        source: str = "simulation.turn",
    ) -> TraitDriftApplied:
        projection = self.trait_projection(state)
        drift = trait_drift_from_experience(
            {
                "social_outcome": signals.social_outcome,
                "mood_level": signals.mood_trajectory,
                "adaptability": projection["raw"]["adaptability"],
            },
            self._coefficients(state),
        )
        drift["resilience"] += 0.003 * signals.task_outcome
        drift["adaptability"] += 0.003 * signals.governance_participation
        drift["assertiveness"] += 0.002 * signals.conflict_outcome

        smoothing_window = int(get_config("TRAIT_DRIFT_SMOOTHING_WINDOW") or 5)
        smoothing_alpha = float(get_config("TRAIT_DRIFT_SMOOTHING_ALPHA") or 0.65)
        hysteresis = float(get_config("TRAIT_DRIFT_HYSTERESIS_THRESHOLD") or 0.0025)
        drift = smooth_trait_deltas(
            drift,
            recent_transitions=state.trait_transition_log.transitions,
            window_size=smoothing_window,
            alpha=smoothing_alpha,
            hysteresis_threshold=hysteresis,
        )

        return TraitDriftApplied(
            agent_id=str(getattr(state, "agent_id", "")),
            step=int(state.step_counter),
            source=source,
            cause="experience_drift",
            deltas={str(k): float(v) for k, v in drift.items()},
            max_step=float(max_step),
            input_signals=signals.model_dump(),
        )

    def apply_experience_drift(
        self,
        state: AgentState,
        signals: ExperienceSignal,
        *,
        max_step: float = 0.01,
        source: str = "simulation.turn",
    ) -> list[dict[str, float | str | int]]:
        """Apply bounded per-turn drift updates and persist an audit trail on ``state``."""

        drift_event = self.build_experience_drift_event(
            state,
            signals,
            max_step=max_step,
            source=source,
        )
        return self.reduce_trait_drift_event(state, drift_event)

    def apply_role_transition_blend(
        self,
        state: AgentState,
        *,
        target_traits: dict[str, float],
        blend_ratio: float = 0.25,
        max_step: float = 0.03,
        source: str = "role_transition",
    ) -> list[dict[str, float | str | int]]:
        """Blend current traits toward a target template when transitioning roles."""

        deltas = {
            trait: (float(target) - float(getattr(state.traits, trait))) * blend_ratio
            for trait, target in target_traits.items()
            if hasattr(state.traits, trait)
        }
        event = TraitDriftApplied(
            agent_id=str(getattr(state, "agent_id", "")),
            step=int(state.step_counter),
            source=source,
            cause="role_transition_blend",
            deltas={str(k): float(v) for k, v in deltas.items()},
            max_step=float(max_step),
            input_signals={"blend_ratio": float(blend_ratio)},
        )
        return self.reduce_trait_drift_event(state, event)

    def apply_exogenous_trait_intervention(
        self,
        state: AgentState,
        trait_updates: dict[str, float],
        *,
        max_step: float = 0.05,
        source: str,
        cause: str = "exogenous_intervention",
    ) -> list[dict[str, float | str | int]]:
        """Apply external/admin trait edits through the same bounded/audited engine path."""

        event = TraitDriftApplied(
            agent_id=str(getattr(state, "agent_id", "")),
            step=int(state.step_counter),
            source=source,
            cause=cause,
            deltas={str(k): float(v) for k, v in trait_updates.items()},
            max_step=float(max_step),
            input_signals={},
        )
        return self.reduce_trait_drift_event(state, event)
