import pytest

from src.agents.council.fitness_store import CouncilFitnessStore
from src.agents.council.orchestrator import CouncilVoteModel
from src.agents.council.types import CouncilQuestion, MemberAnswer


pytestmark = pytest.mark.unit


def _build_answers(member_ids: list[str]) -> list[MemberAnswer]:
    return [MemberAnswer(member_id=member_id, answer="a") for member_id in member_ids]


def _build_vote(winner: str, scores: dict[str, float]) -> CouncilVoteModel:
    return CouncilVoteModel(
        winning_member_id=winner,
        scores=scores,
        metrics={},
        summary="",
        reasoning=None,
    )


def test_fitness_store_tracks_win_rates_and_pairs() -> None:
    store = CouncilFitnessStore()
    question = CouncilQuestion(question_id="q1", prompt="p")
    answers = _build_answers(["a", "b", "c"])
    vote = _build_vote("a", {"a": 0.9, "b": 0.4, "c": 0.4})

    snapshot = store.update_from_vote(question, answers, vote)

    member_stats = snapshot["members"]
    assert member_stats["a"]["wins"] == 1
    assert member_stats["a"]["win_rate"] == pytest.approx(1.0)
    assert member_stats["b"]["win_rate"] == pytest.approx(0.0)

    pair_stats = snapshot["pairs"]
    assert pair_stats["a|b"]["questions_together"] == 1
    assert pair_stats["a|b"]["top_agreements"] == 1
    assert pair_stats["a|b"]["agreement_rate"] == pytest.approx(1.0)


def test_fitness_store_flags_collusion_when_pairs_align_repeatedly() -> None:
    store = CouncilFitnessStore(agreement_threshold=0.6, min_samples=2)
    question = CouncilQuestion(question_id="q2", prompt="p")
    answers = _build_answers(["x", "y"])
    vote = _build_vote("x", {"x": 0.8, "y": 0.8})

    store.update_from_vote(question, answers, vote)
    snapshot = store.update_from_vote(question, answers, vote)

    assert snapshot["warnings"]
    assert "x" in snapshot["warnings"][0]
