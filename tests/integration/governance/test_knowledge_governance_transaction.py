import pytest

from src.governance.knowledge_governance_transaction import (
    GovernanceMutation,
    KnowledgeGovernanceTransaction,
    make_vote_idempotency_key,
)
from src.sim.knowledge_board import KnowledgeBoard
from src.sim.knowledge_entry import KnowledgeEntry, KnowledgeEntryType


@pytest.mark.integration
@pytest.mark.asyncio
async def test_transaction_rolls_back_board_write_on_emit_failure() -> None:
    board = KnowledgeBoard()
    tx = KnowledgeGovernanceTransaction(board, step_provider=lambda: 3)

    state = {"votes": 0}

    def apply() -> int:
        state["votes"] += 1
        return 1

    def rollback(token: object) -> None:
        if token:
            state["votes"] -= int(token)

    async def failing_emit(_payload: dict[str, object]) -> None:
        raise RuntimeError("simulated crash")

    ok = await tx.execute(
        actor_id="a1",
        board_entry=KnowledgeEntry(
            content_full="Vote",
            entry_type=KnowledgeEntryType.VOTE,
            reference_metadata={"idempotency_key": "vote:a1:p1:approve:3"},
        ),
        idempotency_key="vote:a1:p1:approve:3",
        governance_mutation=GovernanceMutation(apply=apply, rollback=rollback),
        emit_event=failing_emit,
        event_payload={"type": "governance_vote"},
    )

    assert ok is False
    assert state["votes"] == 0
    assert board.get_full_entries() == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vote_retry_with_same_idempotency_key_is_deduplicated() -> None:
    board = KnowledgeBoard()
    tx = KnowledgeGovernanceTransaction(board, step_provider=lambda: 9)

    mutation_calls = {"count": 0}

    def apply() -> None:
        mutation_calls["count"] += 1

    key = make_vote_idempotency_key(
        voter_id="a2",
        proposal_entry_id="proposal-1",
        approve=True,
        step=9,
    )
    entry = KnowledgeEntry(
        content_full="Vote",
        entry_type=KnowledgeEntryType.VOTE,
        parent_entry_id="proposal-1",
        reference_metadata={"idempotency_key": key},
    )

    first = await tx.execute(
        actor_id="a2",
        board_entry=entry,
        idempotency_key=key,
        governance_mutation=GovernanceMutation(apply=apply, rollback=lambda _token: None),
    )
    second = await tx.execute(
        actor_id="a2",
        board_entry=entry,
        idempotency_key=key,
        governance_mutation=GovernanceMutation(apply=apply, rollback=lambda _token: None),
    )

    assert first is True
    assert second is True
    assert mutation_calls["count"] == 1
    assert len(board.get_full_entries()) == 1
