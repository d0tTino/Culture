"""Fitness tracking for council outcomes."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

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

    def reset(self) -> None:
        """Clear tracked metrics so repeated tests start from a clean slate."""

        self._wins.clear()
        self._appearances.clear()
        self._pair_counts.clear()
        self._pair_agreements.clear()
        self._agreement_scores.clear()

    def update_from_vote(
        self,
        question: CouncilQuestion,
        answers: Sequence[MemberAnswer],
        vote: CouncilVoteModel,
    ) -> dict[str, object]:
        """Update aggregate metrics based on the latest council vote."""

        top_score = max(vote.scores.values()) if vote.scores else 0.0
        winners = {
            member_id
            for member_id, score in vote.scores.items()
            if score >= top_score
        }

        for answer in answers:
            member_id = answer.member_id
            self._appearances[member_id] += 1
            if member_id in winners:
                self._wins[member_id] += 1

        for idx, left in enumerate(answers):
            for right in answers[idx + 1 :]:
                member_a, member_b = sorted((left.member_id, right.member_id))
                key = f"{member_a}|{member_b}"
                self._pair_counts[key] += 1
                if (left.answer or "").strip().lower() == (right.answer or "").strip().lower():
                    self._pair_agreements[key] += 1

        members_snapshot: dict[str, dict[str, float]] = {}
        for member_id in self._appearances:
            appearances = float(self._appearances[member_id])
            wins = float(self._wins.get(member_id, 0))
            members_snapshot[member_id] = {
                "wins": wins,
                "appearances": appearances,
                "win_rate": wins / appearances if appearances else 0.0,
            }

        pairs_snapshot: dict[str, dict[str, float]] = {}
        warnings: list[str] = []
        for key, count in self._pair_counts.items():
            agreements = float(self._pair_agreements.get(key, 0))
            agreement_rate = agreements / count if count else 0.0
            pairs_snapshot[key] = {
                "questions_together": float(count),
                "top_agreements": agreements,
                "agreement_rate": agreement_rate,
            }
            if count >= self.min_samples and agreement_rate >= self.agreement_threshold:
                warnings.append(
                    f"Pair {key} showing high agreement {agreement_rate:.2f} over {count} questions"
                )

        return {
            "question_id": question.question_id,
            "members": members_snapshot,
            "pairs": pairs_snapshot,
            "warnings": warnings,
        }

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
