"""Fitness tracking for council outcomes."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

from src.agents.council.types import CouncilOutcome, CouncilQuestion, MemberAnswer

if TYPE_CHECKING:  # pragma: no cover - avoid circular imports during runtime
    from src.agents.council.orchestrator import CouncilVoteModel


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

    def update_from_vote(
        self,
        question: CouncilQuestion,
        answers: Sequence[MemberAnswer],
        vote: CouncilVoteModel,
    ) -> dict[str, Any]:
        """Update counters based on a council ``vote`` and return a snapshot."""

        del question  # unused placeholder for future metadata alignment

        answers_by_id = {answer.member_id: answer for answer in answers}
        winner_ids = {vote.winning_member_id}
        scores = vote.scores or {}

        for member_id in answers_by_id:
            self._appearances[member_id] += 1
            if member_id in winner_ids:
                self._wins[member_id] += 1

        for left, right in combinations(sorted(answers_by_id), 2):
            pair_key = self._pair_key(left, right)
            self._pair_counts[pair_key] += 1
            if self._members_agree(left, right, answers_by_id, scores):
                self._pair_agreements[pair_key] += 1

        snapshot = {
            "members": self._serialize_members(),
            "pairs": self._serialize_pairs(),
            "warnings": [],
        }

        for pair_key, stats in snapshot["pairs"].items():
            if stats["questions_together"] < self.min_samples:
                continue
            if stats["agreement_rate"] < self.agreement_threshold:
                continue
            snapshot["warnings"].append(

                    f"Pair {pair_key} shows high agreement "
                    f"({stats['agreement_rate']:.2f}) across {stats['questions_together']} questions."

            )

        return snapshot

    @staticmethod
    def _pair_key(left: str, right: str) -> str:
        return "|".join(sorted([left, right]))

    @staticmethod
    def _members_agree(
        left: str,
        right: str,
        answers_by_id: Mapping[str, MemberAnswer],
        scores: Mapping[str, float],
    ) -> bool:
        left_answer = answers_by_id.get(left)
        right_answer = answers_by_id.get(right)
        if left_answer and right_answer and left_answer.answer == right_answer.answer:
            return True

        if not scores:
            return False

        left_score = scores.get(left)
        right_score = scores.get(right)
        if left_score is None or right_score is None:
            return False

        if left_score == right_score:
            return True

        top_score = max(scores.values())
        return bool(top_score and left_score >= top_score and right_score >= top_score)

    def _serialize_members(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for member_id, appearances in self._appearances.items():
            wins = self._wins.get(member_id, 0)
            win_rate = float(wins / appearances) if appearances else 0.0
            stats[member_id] = {
                "wins": wins,
                "appearances": appearances,
                "win_rate": win_rate,
            }
        return stats

    def _serialize_pairs(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for pair_key, total in self._pair_counts.items():
            agreements = self._pair_agreements.get(pair_key, 0)
            agreement_rate = float(agreements / total) if total else 0.0
            stats[pair_key] = {
                "questions_together": total,
                "top_agreements": agreements,
                "agreement_rate": agreement_rate,
            }
        return stats


council_fitness_store = CouncilFitnessStore()

__all__ = ["CouncilFitnessStore", "MemberFitness", "council_fitness_store"]

