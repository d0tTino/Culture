"""In-memory tracker for council fitness statistics and collusion signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

from .types import CouncilQuestion, MemberAnswer


@dataclass
class MemberFitness:
    """Aggregated performance metrics for a council member."""

    questions: int = 0
    wins: int = 0
    win_rate: float = 0.0


@dataclass
class PairFitness:
    """Agreement tracking for a pair of council members."""

    questions_together: int = 0
    top_agreements: int = 0
    agreement_rate: float = 0.0


def _pair_key(member_a: str, member_b: str) -> tuple[str, str]:
    return tuple(sorted((member_a, member_b)))


class CouncilFitnessStore:
    """Track member win rates and agreement patterns across council runs."""

    def __init__(
        self,
        *,
        agreement_threshold: float = 0.8,
        min_samples: int = 3,
    ) -> None:
        self.agreement_threshold = agreement_threshold
        self.min_samples = min_samples
        self._member_stats: dict[str, MemberFitness] = {}
        self._pair_stats: dict[tuple[str, str], PairFitness] = {}

    def reset(self) -> None:
        """Clear accumulated statistics (useful for tests)."""

        self._member_stats.clear()
        self._pair_stats.clear()

    def _ensure_member(self, member_id: str) -> MemberFitness:
        if member_id not in self._member_stats:
            self._member_stats[member_id] = MemberFitness()
        return self._member_stats[member_id]

    def _ensure_pair(self, member_a: str, member_b: str) -> PairFitness:
        key = _pair_key(member_a, member_b)
        if key not in self._pair_stats:
            self._pair_stats[key] = PairFitness()
        return self._pair_stats[key]

    def _record_member_results(
        self, member_ids: Sequence[str], winning_member_id: str | None
    ) -> None:
        for member_id in member_ids:
            stats = self._ensure_member(member_id)
            stats.questions += 1

        if winning_member_id:
            winner_stats = self._ensure_member(winning_member_id)
            winner_stats.wins += 1

        for stats in self._member_stats.values():
            if stats.questions:
                stats.win_rate = stats.wins / stats.questions

    def _record_pair_results(
        self,
        member_ids: Sequence[str],
        scores: Mapping[str, float] | None,
    ) -> None:
        participants = sorted(set(member_ids))
        for member_a, member_b in combinations(participants, 2):
            pair_stats = self._ensure_pair(member_a, member_b)
            pair_stats.questions_together += 1

        if not scores:
            return

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not sorted_scores:
            return

        top_score = sorted_scores[0][1]
        top_members: list[str] = [
            member_id for member_id, score in sorted_scores if score == top_score
        ]
        if len(top_members) < 2 and len(sorted_scores) > 1:
            top_members.append(sorted_scores[1][0])

        for member_a, member_b in combinations(sorted(set(top_members)), 2):
            pair_stats = self._ensure_pair(member_a, member_b)
            pair_stats.top_agreements += 1

        for stats in self._pair_stats.values():
            if stats.questions_together:
                stats.agreement_rate = stats.top_agreements / stats.questions_together

    def _build_warnings(self) -> list[str]:
        warnings: list[str] = []
        for (member_a, member_b), stats in self._pair_stats.items():
            if stats.questions_together < self.min_samples:
                continue
            if stats.agreement_rate < self.agreement_threshold:
                continue
            warnings.append(
                (
                    f"High agreement detected between {member_a} and {member_b}: "
                    f"{stats.agreement_rate:.2f} over {stats.questions_together} questions"
                )
            )
        return warnings

    def update_from_vote(
        self,
        question: CouncilQuestion,
        answers: Sequence[MemberAnswer],
        vote: Any,
    ) -> dict[str, Any]:
        """Update fitness metrics after a judge vote and return a snapshot."""

        _ = question.question_id  # Ensures the question object was provided
        member_ids = [answer.member_id for answer in answers]
        if not member_ids:
            return self.snapshot()

        winning_member_id = getattr(vote, "winning_member_id", None)
        scores = getattr(vote, "scores", None)

        self._record_member_results(member_ids, winning_member_id)
        self._record_pair_results(member_ids, scores)

        warnings = self._build_warnings()

        return {
            "members": {member_id: asdict(stats) for member_id, stats in self._member_stats.items()},
            "pairs": {
                f"{member_a}|{member_b}": asdict(stats)
                for (member_a, member_b), stats in self._pair_stats.items()
            },
            "warnings": warnings,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return the current fitness state without mutating it."""

        return {
            "members": {member_id: asdict(stats) for member_id, stats in self._member_stats.items()},
            "pairs": {
                f"{member_a}|{member_b}": asdict(stats)
                for (member_a, member_b), stats in self._pair_stats.items()
            },
            "warnings": self._build_warnings(),
        }


council_fitness_store = CouncilFitnessStore()
