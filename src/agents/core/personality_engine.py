from __future__ import annotations

from collections.abc import Sequence
from typing import cast
from warnings import warn

from pydantic import BaseModel

from .agent_state import AgentState
from .trait_policy import (
    action_intent_biasing,
    merge_trait_policy_coefficients,
    mood_update_multiplier,
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

    def _apply_trait_adjustments(
        self,
        state: AgentState,
        deltas: dict[str, float],
        *,
        max_step: float,
        cause: str,
        source: str,
        context: dict[str, float | str] | None = None,
    ) -> list[dict[str, float | str | int]]:
        records: list[dict[str, float | str | int]] = []
        for trait, proposed_delta in deltas.items():
            if not hasattr(state.traits, trait):
                continue
            before = float(getattr(state.traits, trait))
            bounded_delta = max(-max_step, min(max_step, float(proposed_delta)))
            after = max(0.0, min(1.0, before + bounded_delta))
            applied_delta = after - before
            if abs(applied_delta) < 1e-12:
                continue
            setattr(state.traits, trait, after)
            record: dict[str, float | str | int] = {
                "step": int(state.step_counter),
                "trait": trait,
                "before": before,
                "delta": applied_delta,
                "after": after,
                "cause": cause,
                "source": source,
            }
            if context:
                record.update(context)
            records.append(record)

        if records:
            state.trait_change_audit.extend(records)
        return records

    def _coefficients(self, state: AgentState) -> dict[str, float]:
        coefficients = merge_trait_policy_coefficients(state.trait_policy_coefficients)
        return cast(dict[str, float], coefficients)

    def action_biases(self, state: AgentState, available_actions: Sequence[str]) -> dict[str, float]:
        return cast(
            dict[str, float],
            action_intent_biasing(
                state.traits,
                available_actions,
                self._coefficients(state),
            ),
        )

    def mood_multiplier(self, state: AgentState) -> float:
        return float(mood_update_multiplier(state.traits, self._coefficients(state)))

    def relationship_sensitivity(self, state: AgentState, *, is_targeted: bool) -> float:
        return float(
            relationship_update_sensitivity(
                state.traits,
                is_targeted=is_targeted,
                coefficients=self._coefficients(state),
            )
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

        drift = trait_drift_from_experience(
            {
                "social_outcome": signals.social_outcome,
                "mood_level": signals.mood_trajectory,
                "adaptability": state.traits.adaptability,
            },
            self._coefficients(state),
        )
        drift["resilience"] += 0.003 * signals.task_outcome
        drift["adaptability"] += 0.003 * signals.governance_participation
        drift["assertiveness"] += 0.002 * signals.conflict_outcome

        return self._apply_trait_adjustments(
            state,
            drift,
            max_step=max_step,
            cause="experience_drift",
            source=source,
            context={
                "social_outcome": signals.social_outcome,
                "conflict_outcome": signals.conflict_outcome,
                "task_outcome": signals.task_outcome,
                "mood_trajectory": signals.mood_trajectory,
                "governance_participation": signals.governance_participation,
            },
        )

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
        return self._apply_trait_adjustments(
            state,
            deltas,
            max_step=max_step,
            cause="role_transition_blend",
            source=source,
            context={"blend_ratio": blend_ratio},
        )

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

        return self._apply_trait_adjustments(
            state,
            trait_updates,
            max_step=max_step,
            cause=cause,
            source=source,
        )

    def update_traits(
        self,
        state: AgentState,
        signals: ExperienceSignal,
        *,
        max_step: float = 0.01,
    ) -> list[dict[str, float | str | int]]:
        warn(
            "PersonalityEngine.update_traits is deprecated; use apply_experience_drift instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.apply_experience_drift(
            state,
            signals,
            max_step=max_step,
            source="legacy.update_traits",
        )
