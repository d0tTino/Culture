from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from src.sim.graph_knowledge_board import GraphKnowledgeBoard
from src.sim.knowledge_board import KnowledgeBoard
from src.sim.knowledge_board_protocol import (
    GraphKnowledgeBoardAdapter,
    InMemoryKnowledgeBoardAdapter,
    KnowledgeQuery,
    TimeRange,
)
from src.sim.knowledge_entry import KnowledgeEntry

pytestmark = pytest.mark.unit


class DummyResult(list):
    pass


class MockSession:
    def __init__(self, driver: MockDriver) -> None:
        self.driver = driver

    def run(self, query: str, **params: Any) -> Iterable[Any]:
        if "RETURN count(e) AS cnt" in query:
            return DummyResult([{"cnt": len(self.driver.entries)}])
        if "ORDER BY e.step DESC" in query:
            limit = params["limit"]
            rows = sorted(self.driver.entries, key=lambda e: e["step"], reverse=True)[:limit]
            return DummyResult([{"e": row} for row in rows])
        if "ORDER BY e.step ASC" in query:
            rows = sorted(self.driver.entries, key=lambda e: e["step"])
            return DummyResult([{"e": row} for row in rows])
        if "SET e = $props" in query:
            self.driver.entries.append(dict(params["props"]))
            return DummyResult([])
        if "DETACH DELETE" in query:
            if "entry_id" in params:
                self.driver.entries = [
                    e for e in self.driver.entries if e.get("entry_id") != params["entry_id"]
                ]
            else:
                self.driver.entries.clear()
            return DummyResult([])
        return DummyResult([])

    def __enter__(self) -> MockSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class MockDriver:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def session(self) -> MockSession:
        return MockSession(self)


def _fixture_entries() -> list[tuple[str, str, int, str, list[str], dict[str, Any]]]:
    return [
        ("agent-a", "proposal climate", 1, "proposal", ["governance", "proposal"], {}),
        ("agent-b", "support climate", 2, "vote", ["governance", "vote"], {"approve": True}),
        ("agent-c", "reject climate", 3, "vote", ["governance", "vote"], {"approve": False}),
        ("agent-d", "ocean idea", 4, "idea", ["science"], {}),
    ]


def _seed(adapter: InMemoryKnowledgeBoardAdapter | GraphKnowledgeBoardAdapter) -> str:
    proposal_id = ""
    for agent, text, step, entry_type, tags, metadata in _fixture_entries():
        entry = KnowledgeEntry(
            content_full=text,
            entry_type=entry_type,
            tags=tags,
            reference_metadata=metadata,
            parent_entry_id=proposal_id if entry_type == "vote" else None,
        )
        adapter.append_entry(entry, agent_id=agent, step=step)
        if entry_type == "proposal":
            proposal_id = adapter.to_snapshot()["entries"][0]["entry_id"]
    return proposal_id


def test_query_and_vote_parity_across_adapters() -> None:
    memory = InMemoryKnowledgeBoardAdapter(KnowledgeBoard())
    graph = GraphKnowledgeBoardAdapter(GraphKnowledgeBoard(driver=MockDriver()))

    proposal_id = _seed(memory)
    _seed(graph)

    query = KnowledgeQuery(
        semantic="climate",
        topics=("governance",),
        time_range=TimeRange(start_step=1, end_step=3),
        limit=10,
    )

    assert memory.query_entries(query) == graph.query_entries(query)
    assert memory.aggregate_votes([proposal_id]) == graph.aggregate_votes([proposal_id])
