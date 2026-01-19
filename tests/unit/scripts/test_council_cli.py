import asyncio

import pytest
from typer.testing import CliRunner

from scripts import council_cli
from src.agents.council import orchestrator
from src.agents.council.types import CouncilOutcome, CouncilQuestion, MemberAnswer

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def stub_council_config(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_settings() -> tuple[dict, dict[str, str]]:
        config = {
            "members": [
                {"member_id": "alpha-id", "id": "alpha", "display_name": "Alpha Prime"},
                {"member_id": "bravo-id", "id": "bravo", "display_name": "Bravo Squad"},
            ],
            "enabled": True,
        }
        members = {"alpha-id": "Alpha Prime", "bravo-id": "Bravo Squad"}
        return config, members

    monkeypatch.setattr(council_cli, "_load_council_settings", fake_load_settings)
    monkeypatch.setattr(council_cli, "get_config", lambda *_args, **_kwargs: True)


def test_council_cli_reports_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    recorded_calls: list[tuple[CouncilQuestion, str | None, list[str] | None]] = []

    def fake_run_council(
        question: CouncilQuestion,
        *,
        extra_context: str | None,
        rag_docs: list[str] | None,
        allow_disabled_mode: bool,
    ) -> CouncilOutcome:
        recorded_calls.append((question, extra_context, rag_docs))
        return CouncilOutcome(
            question=CouncilQuestion(
                question_id="q-123",
                prompt="Mocked prompt",
                context="Mock context",
                rag_documents=rag_docs or [],
            ),
            answers=[
                MemberAnswer(
                    member_id="alpha-id",
                    answer="Alpha answer",
                    confidence=0.7,
                    reasoning="Alpha reasoning",
                )
            ],
            resolution="Mock resolution",
            winning_member_ids=["alpha-id"],
            summary="Mock summary",
            metrics={"fitness": {"score": 1}},
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
    assert "Winners: Alpha Prime (alpha-id)" in result.output
    assert "Answers:" in result.output
    assert "- Alpha Prime (alpha-id)" in result.output
    assert "Confidence: 0.7" in result.output
    assert "Reasoning: Alpha reasoning" in result.output
    assert "Answer: Alpha answer" in result.output

    question, extra_context, rag_docs = recorded_calls[0]
    assert question.prompt == "What should we do?"
    assert question.context == "Extra context"
    assert rag_docs == ["doc-a", "doc-b"]
    assert extra_context == {"text": "Extra context"}


def test_council_cli_sets_agent_metadata_and_invokes_rag_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    recorded_question: CouncilQuestion | None = None

    class FakeRetriever:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, int, int | None]] = []

        async def retrieve(
            self, agent_identifier: str, query: str, k: int = 5, token_budget: int | None = None
        ) -> list[str]:
            self.calls.append((agent_identifier, query, k, token_budget))
            return []

    retriever = FakeRetriever()

    def fake_run_council(
        question: CouncilQuestion,
        *,
        extra_context: str | None,
        rag_docs: list[str] | None,
        allow_disabled_mode: bool,
    ) -> CouncilOutcome:
        nonlocal recorded_question
        recorded_question = question
        asyncio.run(
            orchestrator._populate_question_rag_documents(
                question,
                base_documents=rag_docs,
                memory_retriever=retriever,
                top_k=1,
            )
        )
        return CouncilOutcome(
            question=question,
            answers=[],
            resolution="Mock resolution",
            winning_member_ids=[],
            summary=None,
            metadata={"fitness": {"scores": []}},
        )

    monkeypatch.setattr(council_cli, "run_council", fake_run_council)

    result = runner.invoke(
        council_cli.app,
        ["Prompt", "--context", "Context", "--agent-id", "cli-agent"],
    )

    assert result.exit_code == 0
    assert recorded_question is not None
    assert recorded_question.metadata == {"agent_id": "cli-agent"}
    assert retriever.calls == [("cli-agent", "Prompt\n\nContext", 1, None)]


def test_council_cli_accepts_question_option(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    recorded_calls: list[tuple[CouncilQuestion, str | None, list[str] | None]] = []

    def fake_run_council(
        question: CouncilQuestion,
        *,
        extra_context: str | None,
        rag_docs: list[str] | None,
        allow_disabled_mode: bool,
    ) -> CouncilOutcome:
        recorded_calls.append((question, extra_context, rag_docs))
        return CouncilOutcome(
            question=question,
            answers=[],
            resolution="Mock resolution",
            winning_member_ids=[],
            summary=None,
            metadata={"fitness": {"scores": []}},
        )

    monkeypatch.setattr(council_cli, "run_council", fake_run_council)

    result = runner.invoke(council_cli.app, ["--question", "What now?"])

    assert result.exit_code == 0
    question, extra_context, rag_docs = recorded_calls[0]
    assert question.prompt == "What now?"
    assert extra_context is None
    assert rag_docs == []


def test_council_cli_show_all_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()

    def fake_run_council(
        question: CouncilQuestion,
        *,
        extra_context: str | None,
        rag_docs: list[str] | None,
        allow_disabled_mode: bool,
    ) -> CouncilOutcome:
        return CouncilOutcome(
            question=question,
            answers=[
                MemberAnswer(member_id="alpha-id", answer="Alpha answer"),
                MemberAnswer(member_id="bravo-id", answer="Bravo answer"),
            ],
            resolution="Mock resolution",
            winning_member_ids=["alpha-id"],
            summary=None,
            metadata={"fitness": {"scores": []}},
        )

    monkeypatch.setattr(council_cli, "run_council", fake_run_council)

    result = runner.invoke(council_cli.app, ["Prompt"])

    assert result.exit_code == 0
    assert "- Alpha Prime (alpha-id)" in result.output
    assert "- Bravo Squad (bravo-id)" not in result.output

    result = runner.invoke(council_cli.app, ["Prompt", "--show-all"])

    assert result.exit_code == 0
    assert "- Alpha Prime (alpha-id)" in result.output
    assert "- Bravo Squad (bravo-id)" in result.output


def test_council_cli_show_votes(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()

    def fake_run_council(
        question: CouncilQuestion,
        *,
        extra_context: str | None,
        rag_docs: list[str] | None,
        allow_disabled_mode: bool,
    ) -> CouncilOutcome:
        return CouncilOutcome(
            question=question,
            answers=[
                MemberAnswer(
                    member_id="alpha-id", answer="Alpha answer", votes={"bravo-id": 0.6}
                ),
                MemberAnswer(
                    member_id="bravo-id", answer="Bravo answer", votes={"alpha-id": 0.4}
                ),
            ],
            resolution="Mock resolution",
            winning_member_ids=["alpha-id"],
            votes={"alpha-id": 0.55, "bravo-id": 0.45},
            summary=None,
            metrics={"fitness": {"score": 0.95}},
        )

    monkeypatch.setattr(council_cli, "run_council", fake_run_council)

    result = runner.invoke(council_cli.app, ["Prompt", "--show-votes", "--show-all"])

    assert result.exit_code == 0
    assert "Judge Scores:" in result.output
    assert "- Alpha Prime (alpha-id): 0.55" in result.output
    assert "- Bravo Squad (bravo-id): 0.45" in result.output
    assert "Votes:" in result.output
    assert "- Alpha Prime (alpha-id):" in result.output
    assert "  - Bravo Squad (bravo-id): 0.6" in result.output
    assert "- Bravo Squad (bravo-id):" in result.output
    assert "  - Alpha Prime (alpha-id): 0.4" in result.output

    result_default = runner.invoke(council_cli.app, ["Prompt"])

    assert "Judge Scores:" not in result_default.output
    assert "Votes:" not in result_default.output


def test_council_cli_show_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()

    def fake_run_council(
        question: CouncilQuestion,
        *,
        extra_context: str | None,
        rag_docs: list[str] | None,
        allow_disabled_mode: bool,
    ) -> CouncilOutcome:
        return CouncilOutcome(
            question=question,
            answers=[MemberAnswer(member_id="alpha-id", answer="Alpha answer")],
            resolution="Mock resolution",
            winning_member_ids=["alpha-id"],
            summary=None,
            metrics={"fitness": {"score": 0.87}, "collusion": {"flagged": False}},
        )

    monkeypatch.setattr(council_cli, "run_council", fake_run_council)

    result = runner.invoke(council_cli.app, ["Prompt", "--show-metrics"])

    assert result.exit_code == 0
    assert "Key metrics:" in result.output
    assert "- fitness: {'score': 0.87}" in result.output
    assert "- collusion: {'flagged': False}" in result.output

    result_default = runner.invoke(council_cli.app, ["Prompt"])

    assert "Key metrics:" not in result_default.output


def test_council_cli_honors_env_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(council_cli, "get_config", lambda *_args, **_kwargs: False)

    result = runner.invoke(council_cli.app, ["Prompt", "--enforce-env-guard"])

    assert result.exit_code == 1
    assert "Council Mode is disabled" in result.output


def test_council_cli_supports_legacy_short_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()

    def fake_run_council(
        question: CouncilQuestion,
        *,
        extra_context: str | None,
        rag_docs: list[str] | None,
        allow_disabled_mode: bool,
    ) -> CouncilOutcome:
        return CouncilOutcome(
            question=question,
            answers=[
                MemberAnswer(member_id="alpha-id", answer="Alpha answer"),
                MemberAnswer(member_id="bravo-id", answer="Bravo answer"),
            ],
            resolution="Mock resolution",
            winning_member_ids=["alpha-id"],
            summary=None,
            metadata={"fitness": {"scores": []}},
        )

    monkeypatch.setattr(council_cli, "run_council", fake_run_council)

    result = runner.invoke(council_cli.app, ["-q", "Prompt", "-a"])

    assert result.exit_code == 0
    assert "- Alpha Prime (alpha-id)" in result.output
    assert "- Bravo Squad (bravo-id)" in result.output
