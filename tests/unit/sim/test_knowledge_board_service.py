import pytest

from src.sim.knowledge_board import KnowledgeBoard
from src.sim.knowledge_board_queries import (
    AgentContributionQueryDTO,
    CausalChainQueryDTO,
    ProposalStatusQueryDTO,
    QueryPagination,
    ThreadQueryDTO,
    TimelineQueryDTO,
)
from src.sim.knowledge_board_service import KnowledgeBoardService
from src.sim.knowledge_entry import KnowledgeEntryType


@pytest.mark.unit
@pytest.mark.asyncio
async def test_post_methods_add_minimal_provenance() -> None:
    step = 12
    board = KnowledgeBoard()
    service = KnowledgeBoardService(
        board,
        step_provider=lambda: step,
        vector_provider=lambda: {"sim": step},
    )

    await service.post_idea(actor_id="agent-1", content="idea", causal_source="test.idea")
    await service.post_vote(
        actor_id="agent-2",
        proposal_id="proposal-1",
        approve=True,
        causal_source="test.vote",
    )
    await service.post_event(actor_id="agent-3", content="event", causal_source="test.event")
    await service.post_lifecycle_transition(
        actor_id="agent-4",
        from_state="active",
        to_state="retired",
        reason="done",
        legacy_artifacts=["artifact"],
        causal_source="test.lifecycle",
    )
    await service.post_human_message(
        actor_id="human",
        content="hello",
        causal_source="test.human",
    )

    assert len(board.entries) == 5
    for entry in board.entries:
        metadata = entry.reference_metadata or {}
        provenance = metadata.get("provenance") or {}
        assert provenance.get("step") == step
        assert provenance.get("actor") == entry.agent_id
        assert isinstance(provenance.get("causal_source"), str)
        assert provenance["causal_source"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_governance_linkage_is_attached_to_entry_and_metadata() -> None:
    board = KnowledgeBoard()
    service = KnowledgeBoardService(
        board,
        step_provider=lambda: 7,
        vector_provider=lambda: {"sim": 7},
    )

    await service.post_event(
        actor_id="agent-1",
        content="governance event",
        event_type=KnowledgeEntryType.GOVERNANCE_DECISION,
        governance_rule_id="rule-123",
        causal_source="test.governance",
    )

    entry = board.entries[-1]
    assert entry.governance_rule_id == "rule-123"
    metadata = entry.reference_metadata or {}
    governance = metadata.get("governance") or {}
    assert governance.get("rule_id") == "rule-123"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_views_and_digest_generation() -> None:
    step = 40
    board = KnowledgeBoard()
    service = KnowledgeBoardService(
        board,
        step_provider=lambda: step,
        vector_provider=lambda: {"sim": step},
    )

    await service.post_proposal(
        actor_id="agent-1",
        content="proposal alpha",
        causal_source="test.proposal",
    )
    proposal_id = board.entries[-1].entry_id
    await service.post_vote(
        actor_id="agent-2",
        proposal_id=proposal_id,
        approve=True,
        causal_source="test.vote",
    )
    await service.post_vote(
        actor_id="agent-3",
        proposal_id=proposal_id,
        approve=False,
        causal_source="test.child",
    )

    timeline = service.query_timeline(TimelineQueryDTO(pagination=QueryPagination(page_size=10)))
    assert timeline.total >= 3

    thread = service.query_thread(
        ThreadQueryDTO(root_entry_id=proposal_id, pagination=QueryPagination(page_size=10))
    )
    assert any(item.parent_entry_id == proposal_id for item in thread.items)

    status = service.query_proposal_status(
        ProposalStatusQueryDTO(proposal_id=proposal_id, pagination=QueryPagination(page_size=10))
    )
    assert status["consensus"]["approvals"] == 1

    contributions = service.query_agent_contribution(
        AgentContributionQueryDTO(agent_id="agent-2", pagination=QueryPagination(page_size=10))
    )
    assert contributions.total >= 1

    chain = service.query_causal_chain(
        CausalChainQueryDTO(entry_id=board.entries[-1].entry_id, depth=4)
    )
    assert chain

    digests = service.generate_story_digests()
    assert set(digests.keys()) == {"daily", "weekly"}
    assert digests["daily"].period == "daily"


class _CapabilityLimitedBoard(KnowledgeBoard):
    supports_threads = False
    supports_causal_chain = False
    supports_votes = False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_methods_return_structured_unsupported_capability_payloads() -> None:
    board = _CapabilityLimitedBoard()
    service = KnowledgeBoardService(
        board,
        step_provider=lambda: 1,
        vector_provider=lambda: {"sim": 1},
    )

    thread = service.query_thread(
        ThreadQueryDTO(root_entry_id="missing", pagination=QueryPagination(page_size=10))
    )
    assert thread.to_dict()["capability"] == "supports_threads"

    proposal = service.query_proposal_status(
        ProposalStatusQueryDTO(proposal_id="proposal-1", pagination=QueryPagination(page_size=10))
    )
    assert proposal["capability"] == "supports_votes"
    assert proposal["unsupported"] is True

    chain = service.query_causal_chain(CausalChainQueryDTO(entry_id="entry-1", depth=4))
    assert chain.to_dict()["capability"] == "supports_causal_chain"
