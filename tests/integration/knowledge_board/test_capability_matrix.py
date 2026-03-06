from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pytest

from src.sim.graph_knowledge_board import GraphKnowledgeBoard
from src.sim.knowledge_board import BoardEntry, KnowledgeBoard
from src.sim.knowledge_board_protocol import (
    EntryStore,
    UnsupportedKnowledgeBoardCapabilityError,
    as_entry_store,
    as_proposal_voting_store,
    as_relationship_store,
    as_semantic_query_store,
)


class DummyResult(list):
    def single(self) -> Any:
        return self[0]


class DummySession:
    def __init__(self, driver: DummyDriver) -> None:
        self.driver = driver

    def run(self, query: str, **params: Any) -> Iterable[Any]:
        if "RETURN count(e) AS cnt" in query:
            return DummyResult([{"cnt": len(self.driver.entries)}])
        if "SET e = $props" in query:
            self.driver.entries.append(dict(params["props"]))
            return DummyResult([])
        if "ORDER BY e.step DESC" in query:
            limit = params["limit"]
            rows = sorted(self.driver.entries, key=lambda entry: entry["step"], reverse=True)[
                :limit
            ]
            return DummyResult([{"e": row} for row in rows])
        if "ORDER BY e.step ASC" in query:
            rows = sorted(self.driver.entries, key=lambda entry: entry["step"])
            return DummyResult([{"e": row} for row in rows])
        if "DETACH DELETE" in query:
            self.driver.entries.clear()
            return DummyResult([])
        if "RETURN p.entry_id AS proposal_id, count(v) AS support_count" in query:
            return DummyResult([{"proposal_id": "p-1", "support_count": 2}])
        if "RETURN i AS entry, endorsements" in query:
            return DummyResult(
                [{"entry": {"entry_id": "idea-1", "entry_type": "idea"}, "endorsements": 2}]
            )
        if "RETURN a.agent_id AS agent_id, e.entry_id AS entry_id" in query:
            return DummyResult(
                [{"agent_id": "agent-1", "entry_id": "idea-1", "entry_type": "idea", "step": 1}]
            )
        return DummyResult([])

    def __enter__(self) -> DummySession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class DummyDriver:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def session(self) -> DummySession:
        return DummySession(self)


def _memory_board() -> KnowledgeBoard:
    return KnowledgeBoard()


def _graph_board() -> GraphKnowledgeBoard:
    return GraphKnowledgeBoard(driver=DummyDriver())


@pytest.mark.integration
@pytest.mark.parametrize("factory", [_memory_board, _graph_board])
def test_entry_and_semantic_capabilities_match_for_shared_contracts(
    factory: Callable[[], EntryStore],
) -> None:
    board = factory()

    as_entry_store(board).add_entry(
        BoardEntry(content_full="seed", entry_type="note"),
        agent_id="agent-1",
        step=1,
    )
    as_entry_store(board).add_entry(
        BoardEntry(
            content_full="proposal",
            entry_type="proposal",
            tags=["governance", "proposal"],
        ),
        agent_id="agent-1",
        step=2,
    )
    as_entry_store(board).add_entry(
        BoardEntry(
            content_full="approve",
            entry_type="vote",
            parent_entry_id=as_entry_store(board).to_snapshot()["entries"][1]["entry_id"],
            reference_metadata={"approve": True, "stance": "approve"},
        ),
        agent_id="agent-2",
        step=3,
    )

    semantic_store = as_semantic_query_store(board)
    proposals = semantic_store.get_active_proposals()

    assert as_entry_store(board).get_recent_entries_for_prompt(1)[0].startswith("[Step 3")
    assert len(proposals) == 1
    assert semantic_store.get_consensus_status(proposals[0]["entry_id"])["consensus"] is True
    assert len(semantic_store.get_agent_stance_history("agent-2")) == 1


@pytest.mark.integration
def test_capability_adapters_fail_explicitly_for_unsupported_memory_features() -> None:
    board = KnowledgeBoard()

    with pytest.raises(UnsupportedKnowledgeBoardCapabilityError) as relationship_error:
        as_relationship_store(board)
    assert relationship_error.value.capability == "RelationshipStore"

    with pytest.raises(UnsupportedKnowledgeBoardCapabilityError) as voting_error:
        as_proposal_voting_store(board)
    assert voting_error.value.capability == "ProposalVotingStore"


@pytest.mark.integration
def test_graph_capability_adapters_expose_extended_features() -> None:
    board = GraphKnowledgeBoard(driver=DummyDriver())

    relationship_store = as_relationship_store(board)
    voting_store = as_proposal_voting_store(board)

    voting_store.record_vote(voter_agent_id="agent-2", proposal_id="p-1", approve=True)

    assert relationship_store.get_endorsed_ideas(min_endorsements=2) == [
        {"entry_id": "idea-1", "entry_type": "idea", "endorsement_count": 2}
    ]
    assert relationship_store.get_agent_contribution_graph() == [
        {"agent_id": "agent-1", "entry_id": "idea-1", "entry_type": "idea", "step": 1}
    ]
    assert voting_store.get_proposal_support_counts(["p-1"]) == {"p-1": 2}
