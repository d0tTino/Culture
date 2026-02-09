from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from src.sim.graph_knowledge_board import GraphKnowledgeBoard
from src.sim.knowledge_board import BoardEntry

pytestmark = pytest.mark.unit


class DummyResult(list):
    def single(self) -> Any:
        return self[0]


class MockSession:
    def __init__(self, driver: MockDriver) -> None:
        self.driver = driver

    def run(self, query: str, **params: Any) -> Iterable[Any]:
        self.driver.calls.append((query, params))
        if "RETURN count(e) AS cnt" in query:
            return DummyResult([{"cnt": self.driver.entry_count}])
        if "ORDER BY e.step DESC" in query:
            limit = params["limit"]
            entries = list(reversed(self.driver.entries))[:limit]
            return DummyResult([{"e": e} for e in entries])
        if "OPTIONAL MATCH (:Agent)-[v:VOTED {approve: true}]->(e)" in query:
            return DummyResult([{"endorsements": self.driver.endorsements.get(params["entry_id"], 0)}])
        if "RETURN p.entry_id AS proposal_id, count(v) AS support_count" in query:
            return DummyResult(
                [
                    {"proposal_id": proposal_id, "support_count": support_count}
                    for proposal_id, support_count in self.driver.proposal_support.items()
                ]
            )
        if "RETURN i AS entry, endorsements" in query:
            return DummyResult(
                [
                    {"entry": entry, "endorsements": endorsements}
                    for entry, endorsements in self.driver.idea_endorsements
                ]
            )
        if "RETURN a.agent_id AS agent_id" in query:
            return DummyResult(self.driver.contribution_rows)
        if "DETACH DELETE" in query:
            self.driver.entries.clear()
            self.driver.entry_count = 0
            return DummyResult([])
        if "SET e = $props" in query:
            self.driver.entries.append(dict(params["props"]))
            self.driver.entry_count += 1
            return DummyResult([])
        return DummyResult([])

    def __enter__(self) -> MockSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class MockDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.entries: list[dict[str, Any]] = []
        self.entry_count = 0
        self.endorsements: dict[str, int] = {}
        self.proposal_support: dict[str, int] = {}
        self.idea_endorsements: list[tuple[dict[str, Any], int]] = []
        self.contribution_rows: list[dict[str, Any]] = []
        self.closed = False

    def session(self) -> MockSession:
        return MockSession(self)

    def close(self) -> None:
        self.closed = True


def test_add_entry_creates_agent_authorship_and_reference_links() -> None:
    driver = MockDriver()
    board = GraphKnowledgeBoard(driver=driver)

    board.add_entry(
        BoardEntry(
            content_full="new idea",
            entry_type="idea",
            reference_metadata={"references": ["proposal-1"]},
        ),
        agent_id="agent-1",
        step=2,
    )

    queries = [query for query, _ in driver.calls]
    assert any("MERGE (a:Agent" in query and "[:AUTHORED]" in query for query in queries)
    assert any("MERGE (source)-[:REFERENCES]->(target)" in query for query in queries)


def test_query_helpers_and_prompt_relationship_summary() -> None:
    driver = MockDriver()
    board = GraphKnowledgeBoard(driver=driver)

    board.add_entry(BoardEntry(content_full="proposal", entry_type="proposal"), "agent-1", 1)
    proposal_id = driver.entries[0]["entry_id"]

    driver.proposal_support = {proposal_id: 3}
    driver.idea_endorsements = [({"entry_id": "idea-1", "entry_type": "idea", "step": 4}, 2)]
    driver.contribution_rows = [
        {"agent_id": "agent-1", "entry_id": proposal_id, "entry_type": "proposal", "step": 1}
    ]
    driver.endorsements[proposal_id] = 3

    board.record_vote(voter_agent_id="agent-2", proposal_id=proposal_id, approve=True)

    assert board.get_proposal_support_counts() == {proposal_id: 3}
    assert board.get_endorsed_ideas(min_endorsements=2) == [
        {"entry_id": "idea-1", "entry_type": "idea", "step": 4, "endorsement_count": 2}
    ]
    assert board.get_agent_contribution_graph() == [
        {"agent_id": "agent-1", "entry_id": proposal_id, "entry_type": "proposal", "step": 1}
    ]

    prompt_entries = board.get_recent_entries_for_prompt(
        max_entries=1,
        include_relationship_summaries=True,
    )
    assert prompt_entries == ["[Step 1, agent-1]: proposal (endorsements: 3)"]
    assert any("MERGE (a)-[v:VOTED" in query for query, _ in driver.calls)
