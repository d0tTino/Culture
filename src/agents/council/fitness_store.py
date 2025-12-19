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
    ) -> None:
        self._wins: Counter[str] = Counter()
        self._appearances: Counter[str] = Counter()
        self._pair_counts: Counter[str] = Counter()
        self._pair_agreements: Counter[str] = Counter()
        self._agreement_scores: list[float] = []
        self.agreement_threshold = float(agreement_threshold)
        self.min_samples = int(min_samples)

    def reset(self) -> None:
        """Clear accumulated fitness statistics."""

        self._wins.clear()
        self._appearances.clear()
        self._pair_counts.clear()
        self._pair_agreements.clear()
        self._agreement_scores.clear()

    @staticmethod
    def _normalize_answer(answer: str | None) -> str:
        return (answer or "").strip().lower()

    def update_from_vote(
        self, question: CouncilQuestion, answers: Sequence[MemberAnswer], vote: Any
    ) -> dict[str, Any]:
        """Update metrics from a council vote and return a serializable snapshot."""

        winners = {getattr(vote, "winning_member_id", "")}
        winners.discard("")

        for answer in answers:
            member_id = str(answer.member_id)
            self._appearances[member_id] += 1
            if member_id in winners:
                self._wins[member_id] += 1

        for left, right in combinations(answers, 2):
            key = "|".join(sorted((left.member_id, right.member_id)))
            self._pair_counts[key] += 1
            if self._normalize_answer(left.answer) == self._normalize_answer(right.answer):
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
                    f"Potential collusion detected between {pair} (agreement rate {rate:.2f})"
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
