from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from src.sim.graph_knowledge_board import GraphKnowledgeBoard
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
from src.sim.knowledge_entry import KnowledgeEntry


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
        if "MATCH (p:KBEntry {entry_type: 'proposal'})" in query:
            proposals = [
                entry for entry in self.driver.entries if entry.get("entry_type") == "proposal"
            ]
            return DummyResult(
                [
                    {"proposal": row}
                    for row in sorted(proposals, key=lambda entry: entry["step"], reverse=True)[
                        : params["limit"]
                    ]
                ]
            )
        if "OPTIONAL MATCH (:Agent)-[v:VOTED]->(p)" in query:
            proposal_id = params["proposal_id"]
            votes = [
                entry
                for entry in self.driver.entries
                if entry.get("entry_type") == "vote"
                and entry.get("parent_entry_id") == proposal_id
            ]
            approvals = sum(
                1 for vote in votes if (vote.get("reference_metadata") or {}).get("approve")
            )
            rejections = len(votes) - approvals
            return DummyResult(
                [{"proposal_id": proposal_id, "approvals": approvals, "rejections": rejections}]
            )
        if "MATCH (a:Agent {agent_id: $agent_id})-[:AUTHORED]->(e:KBEntry)" in query:
            agent_id = params["agent_id"]
            rows = [
                entry
                for entry in self.driver.entries
                if entry.get("agent_id") == agent_id
                and entry.get("entry_type") in ["vote", "endorsement"]
            ]
            return DummyResult(
                [
                    {
                        "entry_id": row["entry_id"],
                        "step": row["step"],
                        "entry_type": row["entry_type"],
                        "parent_entry_id": row.get("parent_entry_id"),
                        "target_agent_id": row.get("target_agent_id"),
                    }
                    for row in rows
                ]
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


def _service(board: KnowledgeBoard | GraphKnowledgeBoard) -> KnowledgeBoardService:
    return KnowledgeBoardService(
        board,
        step_provider=lambda: 10,
        vector_provider=lambda: {"sim": 10},
    )


def _seed(service: KnowledgeBoardService) -> str:
    board = service._board
    board.add_entry(
        KnowledgeEntry(
            content_full="proposal alpha", entry_type="proposal", tags=["governance", "proposal"]
        ),
        agent_id="agent-1",
        step=1,
    )
    proposal_id = board.get_full_entries()[-1]["entry_id"]
    board.add_entry(
        KnowledgeEntry(
            content_full="vote yes",
            entry_type="vote",
            parent_entry_id=proposal_id,
            reference_metadata={"approve": True, "stance": "approve"},
        ),
        agent_id="agent-2",
        step=2,
    )
    board.add_entry(
        KnowledgeEntry(
            content_full="vote no",
            entry_type="vote",
            parent_entry_id=proposal_id,
            reference_metadata={"approve": False, "stance": "reject"},
        ),
        agent_id="agent-3",
        step=3,
    )
    board.add_entry(
        KnowledgeEntry(
            content_full="reply to proposal",
            entry_type="note",
            parent_entry_id=proposal_id,
            tags=["discussion"],
        ),
        agent_id="agent-4",
        step=4,
    )
    board.add_entry(
        KnowledgeEntry(
            content_full="reply to reply",
            entry_type="note",
            parent_entry_id=board.get_full_entries()[-1]["entry_id"],
            tags=["discussion"],
        ),
        agent_id="agent-5",
        step=5,
    )
    return proposal_id


@pytest.mark.integration
def test_query_result_parity_for_overlapping_capabilities() -> None:
    memory_service = _service(KnowledgeBoard())
    graph_service = _service(GraphKnowledgeBoard(driver=DummyDriver()))

    proposal_id = _seed(memory_service)
    _seed(graph_service)

    timeline_query = TimelineQueryDTO(pagination=QueryPagination(page_size=10))
    memory_timeline = memory_service.query_timeline(timeline_query).to_dict()
    graph_timeline = graph_service.query_timeline(timeline_query).to_dict()
    assert memory_timeline == graph_timeline

    thread_query = ThreadQueryDTO(
        root_entry_id=proposal_id, pagination=QueryPagination(page_size=10)
    )
    assert (
        memory_service.query_thread(thread_query).to_dict()
        == graph_service.query_thread(thread_query).to_dict()
    )

    proposal_query = ProposalStatusQueryDTO(
        proposal_id=proposal_id, pagination=QueryPagination(page_size=10)
    )
    assert memory_service.query_proposal_status(
        proposal_query
    ) == graph_service.query_proposal_status(proposal_query)

    contribution_query = AgentContributionQueryDTO(
        agent_id="agent-2", pagination=QueryPagination(page_size=10)
    )
    assert (
        memory_service.query_agent_contribution(contribution_query).to_dict()
        == graph_service.query_agent_contribution(contribution_query).to_dict()
    )

    causal_query = CausalChainQueryDTO(
        entry_id=memory_service._board.get_full_entries()[-1]["entry_id"], depth=5
    )
    memory_chain = [item.to_dict() for item in memory_service.query_causal_chain(causal_query)]
    graph_chain = [item.to_dict() for item in graph_service.query_causal_chain(causal_query)]
    assert memory_chain == graph_chain
