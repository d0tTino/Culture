import pytest

from scripts import council_metrics

pytestmark = pytest.mark.unit


def test_council_metrics_reports_stats(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    mock_metrics = {
        "members": [
            {
                "member_id": "alpha",
                "participations": 5,
                "wins": 3,
                "win_rate": 0.6,
                "avg_confidence": 0.8,
            },
            {
                "member_id": "bravo",
                "participations": 0,
                "wins": 0,
                "win_rate": 0.0,
                "avg_confidence": 0.0,
            },
        ],
        "pairwise": [
            {
                "member_a": "alpha",
                "member_b": "bravo",
                "agreements": 2,
                "disagreements": 1,
                "agreement_rate": 2 / 3,
            }
        ],
    }

    class FakeOrchestrator:
        def serialize_metrics(self) -> dict[str, list[dict[str, float]]]:
            return mock_metrics

    monkeypatch.setattr(council_metrics, "CouncilOrchestrator", FakeOrchestrator)

    council_metrics.main()

    output = capsys.readouterr().out
    assert "Member Win Rates:" in output
    assert "- alpha: 60.00% win rate (3/5 wins, avg confidence 0.80)" in output
    assert "- bravo: 0.00% win rate (0/0 wins, avg confidence 0.00)" in output

    assert "Pairwise Agreement Rates:" in output
    assert "- alpha vs bravo: 66.67% agreement (2 agreements / 1 disagreements)" in output

    assert "Warnings:" in output
    assert "Member bravo has no recorded participations." in output
