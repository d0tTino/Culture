from pathlib import Path

import pytest

from src.agents.council.stats_store import CouncilStatsStore
from src.agents.council.types import CouncilOutcome, CouncilQuestion, MemberAnswer

pytestmark = pytest.mark.unit


@pytest.fixture()
def store(tmp_path: Path) -> CouncilStatsStore:
    db_dir = tmp_path / "council"
    db_dir.mkdir(parents=True, exist_ok=True)
    return CouncilStatsStore(db_path=db_dir / "stats.sqlite3")


def _build_outcome() -> CouncilOutcome:
    question = CouncilQuestion(question_id="q-1", prompt="What now?")
    answers = [
        MemberAnswer(member_id="alpha", answer="Option A", confidence=0.8),
        MemberAnswer(member_id="beta", answer="Option A", confidence=0.6),
        MemberAnswer(member_id="gamma", answer="Option B", confidence=0.4),
    ]
    return CouncilOutcome(
        question=question,
        answers=answers,
        resolution="Option A",
        winning_member_ids=["alpha"],
    )


def test_stats_store_records_member_and_pairwise_metrics(
    store: CouncilStatsStore,
) -> None:
    outcome = _build_outcome()

    store.record_outcome(
        outcome,
        score_map={"alpha": 1.0, "beta": 0.5, "gamma": 0.0},
        ema_alpha=0.5,
    )

    alpha_stats = store.get_member_stats("alpha")
    assert alpha_stats["participations"] == 1
    assert alpha_stats["wins"] == 1
    assert alpha_stats["avg_confidence"] == pytest.approx(0.8)

    beta_stats = store.get_member_stats("beta")
    assert beta_stats["participations"] == 1
    assert beta_stats["wins"] == 0
    assert beta_stats["win_rate"] == 0.0

    pairwise = store.get_pairwise_agreement("alpha", "beta")
    assert pairwise["agreements"] == 1
    assert pairwise["disagreements"] == 0
    assert pairwise["ema_agreement"] == pytest.approx(0.5)
    assert pairwise["ema_last_updated"]
    assert pairwise["ema_high_agreement"] is False
    divergent = store.get_pairwise_agreement("alpha", "gamma")
    assert divergent["agreements"] == 0
    assert divergent["disagreements"] == 1


def test_stats_store_serializes_aggregates(store: CouncilStatsStore) -> None:
    store.record_batch([_build_outcome(), _build_outcome()])

    snapshot = store.serialize_metrics()

    assert snapshot["members"]
    alpha_entry = next(m for m in snapshot["members"] if m["member_id"] == "alpha")
    assert alpha_entry["participations"] == 2
    assert alpha_entry["wins"] == 2
    assert alpha_entry["win_rate"] == 1.0

    pairwise_entries = {(p["member_a"], p["member_b"]): p for p in snapshot["pairwise"]}
    assert pairwise_entries[("alpha", "beta")]["agreement_rate"] == 1.0
    assert pairwise_entries[("alpha", "beta")]["ema_agreement"] == 1.0
    assert pairwise_entries[("alpha", "beta")]["ema_high_agreement"] is True
    assert pairwise_entries[("alpha", "beta")]["ema_last_updated"]
    assert pairwise_entries[("alpha", "gamma")]["agreements"] == 0


def test_stats_store_updates_pairwise_ema(store: CouncilStatsStore) -> None:
    outcome = _build_outcome()

    store.record_outcome(
        outcome,
        score_map={"alpha": 1.0, "beta": 0.5, "gamma": 0.0},
        ema_alpha=0.5,
    )
    store.record_outcome(
        outcome,
        score_map={"alpha": 1.0, "beta": 1.0, "gamma": 0.0},
        ema_alpha=0.5,
    )

    pairwise = store.get_pairwise_agreement("alpha", "beta")
    assert pairwise["ema_agreement"] == pytest.approx(0.75)


def test_stats_store_updates_pairwise_ema_across_outcomes(
    store: CouncilStatsStore,
) -> None:
    outcome = _build_outcome()

    store.record_outcome(
        outcome,
        score_map={"alpha": 1.0, "beta": 1.0, "gamma": 1.0},
        ema_alpha=0.5,
    )
    store.record_outcome(
        outcome,
        score_map={"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
        ema_alpha=0.5,
    )
    store.record_outcome(
        outcome,
        score_map={"alpha": 1.0, "beta": 0.8, "gamma": 0.0},
        ema_alpha=0.5,
    )

    pairwise = store.get_pairwise_agreement("alpha", "beta")
    assert pairwise["ema_agreement"] == pytest.approx(0.65)


def test_stats_store_serializes_ema_flags(store: CouncilStatsStore) -> None:
    outcome = _build_outcome()

    store.record_outcome(
        outcome,
        score_map={"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
        ema_alpha=0.5,
    )

    snapshot = store.serialize_metrics()

    pairwise_entries = {(p["member_a"], p["member_b"]): p for p in snapshot["pairwise"]}
    alpha_beta = pairwise_entries[("alpha", "beta")]
    assert alpha_beta["ema_agreement"] == pytest.approx(0.0)
    assert alpha_beta["ema_high_agreement"] is False
    assert alpha_beta["ema_low_agreement"] is True
    assert alpha_beta["ema_last_updated"]
