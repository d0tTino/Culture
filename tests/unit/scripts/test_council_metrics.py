import pytest

from scripts import council_metrics

pytestmark = pytest.mark.unit


def test_council_metrics_reports_stats(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    mock_metrics = {
        "members": [
            {
                "member_id": "alpha",
                "participations": 12,
                "wins": 3,
                "win_rate": 0.1,
                "avg_confidence": 0.8,
            },
            {
                "member_id": "bravo",
                "participations": 0,
                "wins": 0,
                "win_rate": 0.0,
                "avg_confidence": 0.0,
            },
            {
                "member_id": "charlie",
                "participations": 20,
                "wins": 1,
                "win_rate": 0.05,
                "avg_confidence": 0.4,
            },
        ],
        "pairwise": [
            {
                "member_a": "alpha",
                "member_b": "bravo",
                "agreements": 2,
                "disagreements": 1,
                "agreement_rate": 2 / 3,
                "flagged": True,
                "collusion_warning": "alpha vs bravo agreeing too often",
            }
        ],
        "collusion_warnings": ["Global collusion alert"],
    }

    class FakeOrchestrator:
        def __init__(self) -> None:
            self.received_question = None

        def serialize_metrics(self, *, question_id: str | None = None) -> dict[str, list[dict[str, float]]]:
            self.received_question = question_id
            return mock_metrics

    orchestrator = FakeOrchestrator()
    monkeypatch.setattr(council_metrics, "CouncilOrchestrator", lambda: orchestrator)

    council_metrics.main(["--show-collusion-flags"])

    output = capsys.readouterr().out
    assert orchestrator.received_question is None
    assert "Council Metrics" in output
    assert "- alpha: 10.00% win rate (3/12 wins, avg confidence 0.80)" in output
    assert "- bravo: 0.00% win rate (0/0 wins, avg confidence 0.00)" in output
    assert "- charlie: 5.00% win rate (1/20 wins, avg confidence 0.40)" in output

    assert "Pairwise Agreement Rates:" in output
    assert "- alpha vs bravo: 66.67% agreement [FLAGGED] (2 agreements / 1 disagreements) - alpha vs bravo agreeing too often" in output

    assert "Warnings:" in output
    assert "Member bravo has no recorded participations." in output
    assert "Member charlie is a candidate for deactivation (win rate 5.00% below 20.00% after 20 participations)." in output
    assert "alpha vs bravo agreeing too often" in output
    assert "Global collusion alert" in output
    assert "Member alpha is a candidate for deactivation" in output


def test_council_metrics_filters_questions(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    mock_metrics = {
        "members": [],
        "pairwise": [],
        "questions": [
            {
                "question_id": "target-question",
                "prompt": "What now?",
                "members": [
                    {
                        "member_id": "alpha",
                        "participations": 12,
                        "wins": 1,
                        "win_rate": 1 / 12,
                        "avg_confidence": 0.9,
                    }
                ],
                "pairwise": [
                    {
                        "member_a": "alpha",
                        "member_b": "bravo",
                        "agreements": 3,
                        "disagreements": 0,
                        "agreement_rate": 1.0,
                        "flagged": True,
                        "collusion_warning": "Potential collusion detected",
                    }
                ],
                "warnings": ["Custom question warning"],
            },
            {
                "question_id": "other-question",
                "members": [],
                "pairwise": [],
            },
        ],
    }

    class FakeOrchestrator:
        def __init__(self) -> None:
            self.received_question: str | None = None

        def serialize_metrics(self, *, question_id: str | None = None) -> dict[str, list[dict[str, float]]]:
            self.received_question = question_id
            return mock_metrics

    orchestrator = FakeOrchestrator()
    monkeypatch.setattr(council_metrics, "CouncilOrchestrator", lambda: orchestrator)

    council_metrics.main(["--question-id", "target-question", "--show-collusion-flags"])

    output = capsys.readouterr().out
    assert orchestrator.received_question == "target-question"
    assert "Question: target-question" in output
    assert "Prompt: What now?" in output
    assert "Custom question warning" in output
    assert "Member alpha is a candidate for deactivation" in output
    assert "Potential collusion detected" in output
    assert "other-question" not in output
