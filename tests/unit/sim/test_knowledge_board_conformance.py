from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pytest

from src.sim.graph_knowledge_board import GraphKnowledgeBoard
from src.sim.knowledge_board import BoardEntry, KnowledgeBoard
from src.sim.knowledge_board_protocol import (
    KnowledgeBoardProtocol,
    supports_graph_queries,
    supports_voting,
)

pytestmark = pytest.mark.unit


class DummyResult(list):
    def single(self) -> Any:
        return self[0]


class MockSession:
    def __init__(self, driver: MockDriver) -> None:
        self.driver = driver

    def run(self, query: str, **params: Any) -> Iterable[Any]:
        if "RETURN count(e) AS cnt" in query:
            return DummyResult([{"cnt": len(self.driver.entries)}])
        if "ORDER BY e.step DESC" in query:
            limit = params["limit"]
            rows = sorted(self.driver.entries, key=lambda entry: entry["step"], reverse=True)[:limit]
            return DummyResult([{"e": row} for row in rows])
        if "ORDER BY e.step ASC" in query:
            rows = sorted(self.driver.entries, key=lambda entry: entry["step"])
            return DummyResult([{"e": row} for row in rows])
        if "DETACH DELETE" in query:
            self.driver.entries.clear()
            return DummyResult([])
        if "SET e = $props" in query:
            self.driver.entries.append(dict(params["props"]))
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


def _memory_factory() -> KnowledgeBoardProtocol:
    return KnowledgeBoard()


def _graph_factory() -> KnowledgeBoardProtocol:
    return GraphKnowledgeBoard(driver=MockDriver())


@pytest.mark.parametrize("factory", [_memory_factory, _graph_factory])
def test_core_operations_match_backend_contract(
    factory: Callable[[], KnowledgeBoardProtocol],
) -> None:
    board = factory()

    board.add_entry(
        BoardEntry(content_full="alpha", entry_type="note"),
        agent_id="agent-1",
        step=1,
    )
    board.add_entry(
        BoardEntry(content_full="beta", entry_type="idea", content_summary="summary-beta"),
        agent_id="agent-2",
        step=2,
    )

    assert board.get_recent_entries_for_prompt(2) == [
        "[Step 1, agent-1]: alpha",
        "[Step 2, agent-2]: summary-beta",
    ]

    snapshot = board.to_snapshot()
    board.replace_entries(snapshot["entries"][:1])
    assert board.get_recent_entries_for_prompt(5) == ["[Step 1, agent-1]: alpha"]

    board.from_snapshot(snapshot)
    assert board.get_recent_entries_for_prompt(5) == [
        "[Step 1, agent-1]: alpha",
        "[Step 2, agent-2]: summary-beta",
    ]


@pytest.mark.parametrize("factory", [_memory_factory, _graph_factory])
def test_snapshot_replay_invariants(
    factory: Callable[[], KnowledgeBoardProtocol],
) -> None:
    source = factory()
    source.add_entry(
        BoardEntry(
            content_full="proposal text",
            entry_type="proposal",
            tags=["governance", "proposal"],
            reference_metadata={"proposal_id": "p-1"},
        ),
        agent_id="agent-3",
        step=5,
    )
    source.add_entry(
        BoardEntry(content_full="follow up", entry_type="note"),
        agent_id="agent-4",
        step=6,
    )

    restored = factory()
    restored.from_snapshot(source.to_snapshot())

    assert restored.get_recent_entries_for_prompt(5) == source.get_recent_entries_for_prompt(5)

    restored_snapshot = restored.to_snapshot()
    source_snapshot = source.to_snapshot()
    assert restored_snapshot["entries"] == source_snapshot["entries"]


@pytest.mark.parametrize(
    ("factory", "expects_extensions"),
    [(_memory_factory, False), (_graph_factory, True)],
)
def test_optional_feature_checks(
    factory: Callable[[], KnowledgeBoardProtocol],
    expects_extensions: bool,
) -> None:
    board = factory()

    assert supports_voting(board) is expects_extensions
    assert supports_graph_queries(board) is expects_extensions
