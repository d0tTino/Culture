import pytest
from typer.testing import CliRunner

from scripts import council_cli
from src.agents.council.types import CouncilOutcome, CouncilQuestion, MemberAnswer

pytestmark = pytest.mark.unit


def test_council_cli_reports_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    recorded_calls: list[tuple[CouncilQuestion, str | None, list[str] | None]] = []

    def fake_run_council(
        question: CouncilQuestion,
        *,
        extra_context: str | None,
        rag_docs: list[str] | None,
    ) -> CouncilOutcome:
        recorded_calls.append((question, extra_context, rag_docs))
        return CouncilOutcome(
            question=CouncilQuestion(
                question_id="q-123",
                prompt="Mocked prompt",
                context="Mock context",
                rag_documents=rag_docs or [],
            ),
            answers=[MemberAnswer(member_id="alpha", answer="Alpha answer")],
            resolution="Mock resolution",
            winning_member_ids=["alpha"],
            summary="Mock summary",
            metadata={"fitness": {"scores": []}},
        )

    monkeypatch.setattr(council_cli, "run_council", fake_run_council)

    result = runner.invoke(
        council_cli.app,
        [
            "What should we do?",
            "--context",
            "Extra context",
            "--rag-doc",
            "doc-a",
            "--rag-doc",
            "doc-b",
        ],
    )

    assert result.exit_code == 0
    assert "Question: Mocked prompt" in result.output
    assert "Context: Mock context" in result.output
    assert "RAG Docs:" in result.output
    assert "- doc-a" in result.output
    assert "- doc-b" in result.output
    assert "Resolution: Mock resolution" in result.output
    assert "Summary: Mock summary" in result.output
    assert "Winners: alpha" in result.output
    assert "- alpha: Alpha answer" in result.output

    question, extra_context, rag_docs = recorded_calls[0]
    assert question.prompt == "What should we do?"
    assert question.context == "Extra context"
    assert rag_docs == ["doc-a", "doc-b"]
    assert extra_context == "Extra context"
