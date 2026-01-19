import pytest

pytest.importorskip("langgraph")
pytestmark = pytest.mark.unit

from src.agents.council.types import (  # noqa: E402
    CouncilOutcome,
    CouncilQuestion,
    MemberAnswer,
)
from src.agents.graphs import council_graph  # noqa: E402
from src.agents.graphs.council_graph import CouncilGraphState, build_graph  # noqa: E402


@pytest.mark.asyncio
async def test_council_node_sets_outcome_and_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}
    dummy_outcome = CouncilOutcome(
        question=CouncilQuestion(question_id="q-123", prompt="What now?"),
        answers=[MemberAnswer(member_id="alpha", answer="Proceed")],
        resolution="Resolution text",
        winning_member_ids=["alpha"],
        winner_answer="Proceed",
        summary="Summary text",
        metrics={
            "fitness_snapshot": {"members": {}, "pairs": {}, "warnings": ["collusion"]},
            "collusion_warnings": ["collusion"],
        },
        metadata={
            "metrics": {
                "fitness_snapshot": {"members": {}, "pairs": {}, "warnings": ["collusion"]},
                "collusion_warnings": ["collusion"],
            }
        },
    )

    def fake_run_council(
        question: CouncilQuestion,
        *,
        extra_context: dict[str, object] | None,
        rag_docs: list[str] | None,
    ) -> CouncilOutcome:
        recorded["question"] = question
        recorded["extra_context"] = extra_context
        recorded["rag_docs"] = rag_docs
        return dummy_outcome

    monkeypatch.setattr(council_graph, "run_council", fake_run_council)

    graph = build_graph()
    starting_state: CouncilGraphState = {
        "question": "How should we proceed?",
        "question_id": "q-001",
        "context": "Use prior research",
        "rag_documents": ["doc-a", "doc-b"],
    }

    result = await graph.ainvoke(starting_state)

    assert result["council_outcome"] == dummy_outcome
    assert result["final_answer"] == dummy_outcome.summary
    assert {
        "fitness_snapshot",
        "collusion_warnings",
    }.issubset(result["council_outcome"].metrics)
    assert {
        "fitness_snapshot",
        "collusion_warnings",
    }.issubset(result["council_outcome"].metadata["metrics"])
    assert isinstance(recorded["question"], CouncilQuestion)
    assert recorded["question"].prompt == starting_state["question"]
    assert recorded["extra_context"] == {"text": starting_state["context"]}
    assert recorded["rag_docs"] == starting_state["rag_documents"]
