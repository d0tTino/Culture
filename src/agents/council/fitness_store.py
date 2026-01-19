"""Fitness tracking for council outcomes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

from src.agents.council.types import CouncilOutcome, CouncilQuestion, MemberAnswer

if TYPE_CHECKING:  # pragma: no cover - avoid circular imports during runtime
    pass


@dataclass
class MemberFitness:
    """Aggregated performance metrics for a council member."""

    wins: int = 0
    appearances: int = 0
    win_rate: float = 0.0
    agreement_score: float = 0.0


class CouncilFitnessStore:
    """Store and update fitness metrics for council participants."""

    def __init__(
        self,
        *,
        agreement_threshold: float = 0.75,
        min_samples: int = 3,
        score_tolerance: float = 0.05,
    ) -> None:
        self._wins: Counter[str] = Counter()
        self._appearances: Counter[str] = Counter()
        self._pair_counts: Counter[str] = Counter()
        self._pair_agreements: Counter[str] = Counter()
        self._agreement_scores: list[float] = []
        self.agreement_threshold = float(agreement_threshold)
        self.min_samples = int(min_samples)
        self.score_tolerance = float(score_tolerance)

    def reset(self) -> None:
        """Clear accumulated fitness statistics."""

        self._wins.clear()
        self._appearances.clear()
        self._pair_counts.clear()
        self._pair_agreements.clear()
        self._agreement_scores.clear()

    @staticmethod
    def _coerce_score(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _extract_total_scores(self, vote: Any) -> dict[str, float]:
        totals: dict[str, float] = {}
        metrics = getattr(vote, "metrics", None)
        if isinstance(metrics, Mapping):
            member_scores = metrics.get("member_scores")
            if isinstance(member_scores, Mapping):
                for member_id, score_map in member_scores.items():
                    if not isinstance(score_map, Mapping):
                        continue
                    total_value = self._coerce_score(score_map.get("total"))
                    if total_value is not None:
                        totals[str(member_id)] = total_value

        def _update_from_mapping(raw: Mapping[str, Any]) -> None:
            for member_id, value in raw.items():
                total_value: float | None = None
                if isinstance(value, Mapping):
                    total_value = self._coerce_score(value.get("total"))
                    if total_value is None:
                        numeric_values = [
                            score
                            for score in (self._coerce_score(item) for item in value.values())
                            if score is not None
                        ]
                        if numeric_values:
                            total_value = sum(numeric_values) / len(numeric_values)
                else:
                    total_value = self._coerce_score(value)
                if total_value is not None:
                    totals.setdefault(str(member_id), total_value)

        scores = getattr(vote, "scores", None)
        if isinstance(scores, Mapping):
            _update_from_mapping(scores)

        votes = getattr(vote, "votes", None)
        if isinstance(votes, Mapping):
            _update_from_mapping(votes)

        return totals

    def update_from_vote(
        self, question: CouncilQuestion, answers: Sequence[MemberAnswer], vote: Any
    ) -> dict[str, Any]:
        """Update metrics from a council vote and return a serializable snapshot."""

        winners = {getattr(vote, "winning_member_id", "")}
        winners.discard("")
        total_scores = self._extract_total_scores(vote)

        for answer in answers:
            member_id = str(answer.member_id)
            self._appearances[member_id] += 1
            if member_id in winners:
                self._wins[member_id] += 1

        for left, right in combinations(answers, 2):
            key = "|".join(sorted((left.member_id, right.member_id)))
            self._pair_counts[key] += 1
            left_score = total_scores.get(left.member_id)
            right_score = total_scores.get(right.member_id)
            if (
                left_score is not None
                and right_score is not None
                and abs(left_score - right_score) <= self.score_tolerance
            ):
                self._pair_agreements[key] += 1

        members_snapshot: dict[str, Any] = {}
        for member_id in set(self._appearances.keys()) | winners:
            appearances = self._appearances.get(member_id, 0)
            wins = self._wins.get(member_id, 0)
            members_snapshot[member_id] = {
                "wins": wins,
                "participations": appearances,
                "win_rate": float(wins / appearances) if appearances else 0.0,
                "agreement_score": self.average_agreement_score,
            }

        pairs_snapshot: dict[str, Any] = {}
        warnings: list[str] = []
        for pair, together in self._pair_counts.items():
            agreements = self._pair_agreements.get(pair, 0)
            rate = float(agreements / together) if together else 0.0
            pairs_snapshot[pair] = {
                "questions_together": together,
                "top_agreements": agreements,
                "agreement_rate": rate,
            }
            if together >= self.min_samples and rate >= self.agreement_threshold:
                warnings.append(
                    f"Repeated high agreement detected between {pair} (agreement rate {rate:.2f})"
                )

        return {"members": members_snapshot, "pairs": pairs_snapshot, "warnings": warnings}

    def record(self, outcome: CouncilOutcome) -> None:
        """Record the results of a completed council round."""

        winners = set(outcome.winning_member_ids)
        agreement_score = self._extract_agreement(outcome.metadata)
        if agreement_score is not None:
            self._agreement_scores.append(agreement_score)

        for member_id in self._iter_participants(outcome.answers):
            self._appearances[member_id] += 1
            if member_id in winners:
                self._wins[member_id] += 1

    @staticmethod
    def _iter_participants(answers: Sequence) -> Iterable[str]:
        for answer in answers:
            member_id = getattr(answer, "member_id", None)
            if member_id:
                yield str(member_id)

    @staticmethod
    def _extract_agreement(metadata: Mapping[str, object] | None) -> float | None:
        if not isinstance(metadata, Mapping):
            return None
        raw_score = metadata.get("agreement_score")
        if isinstance(raw_score, (int, float)):
            return float(raw_score)
        return None

    def get_member_fitness(self, member_id: str) -> MemberFitness:
        """Return the fitness metrics for the provided member."""

        appearances = self._appearances.get(member_id, 0)
        wins = self._wins.get(member_id, 0)
        win_rate = float(wins / appearances) if appearances else 0.0
        return MemberFitness(
            wins=wins,
            appearances=appearances,
            win_rate=win_rate,
            agreement_score=self.average_agreement_score,
        )

    @property
    def average_agreement_score(self) -> float:
        """Average agreement score across recorded outcomes."""

        if not self._agreement_scores:
            return 0.0
        return sum(self._agreement_scores) / len(self._agreement_scores)


council_fitness_store = CouncilFitnessStore()

__all__ = ["CouncilFitnessStore", "council_fitness_store"]
