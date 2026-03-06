from unittest.mock import MagicMock

import pytest

from src.sim.knowledge_board import BoardEntry, KnowledgeBoard
from src.sim.knowledge_entry import KnowledgeEntryType
from src.sim.version_vector import VersionVector

pytestmark = pytest.mark.unit


def _create_board(num: int) -> KnowledgeBoard:
    kb = KnowledgeBoard()
    for i in range(num):
        kb.add_entry(
            BoardEntry(
                content_full=f"entry{i}",
                entry_type=KnowledgeEntryType.NOTE,
                tags=["unit"],
            ),
            agent_id="A",
            step=i,
        )
    return kb


def test_get_state_positive() -> None:
    kb = _create_board(3)
    result = kb.get_state(2)
    assert result == [
        "Step 1 (Agent: A): entry1",
        "Step 2 (Agent: A): entry2",
    ]


@pytest.mark.parametrize("val", [0, -1])
def test_get_state_invalid(val: int) -> None:
    kb = _create_board(1)
    with pytest.raises(ValueError):
        kb.get_state(val)


def test_get_recent_entries_for_prompt_positive() -> None:
    kb = _create_board(2)
    result = kb.get_recent_entries_for_prompt(1)
    assert result == ["[Step 1, A]: entry1"]


@pytest.mark.parametrize("val", [0, -5])
def test_get_recent_entries_for_prompt_invalid(val: int) -> None:
    kb = _create_board(1)
    with pytest.raises(ValueError):
        kb.get_recent_entries_for_prompt(val)


def test_get_recent_entries_with_none_summary() -> None:
    kb = KnowledgeBoard()
    kb.add_entry(
        BoardEntry(
            content_full="entry",
            entry_type=KnowledgeEntryType.NOTE,
            tags=["unit"],
        ),
        agent_id="A",
        step=1,
    )
    result = kb.get_recent_entries_for_prompt(1)
    assert result == ["[Step 1, A]: entry"]


def test_add_entry_calls_increment_when_no_vector() -> None:
    kb = KnowledgeBoard()
    kb.vector.increment = MagicMock()
    kb.vector.merge = MagicMock()

    kb.add_entry(
        BoardEntry(
            content_full="entry",
            entry_type=KnowledgeEntryType.NOTE,
        ),
        agent_id="A",
        step=1,
    )

    kb.vector.increment.assert_called_once_with("A")
    kb.vector.merge.assert_not_called()


def test_add_entry_calls_merge_when_vector_supplied() -> None:
    kb = KnowledgeBoard()
    kb.vector.increment = MagicMock()
    kb.vector.merge = MagicMock()

    vec = {"B": 2}
    kb.add_entry(
        BoardEntry(
            content_full="entry",
            entry_type=KnowledgeEntryType.NOTE,
        ),
        agent_id="A",
        step=1,
        vector=vec,
    )

    kb.vector.merge.assert_called_once()
    arg = kb.vector.merge.call_args.args[0]
    assert isinstance(arg, VersionVector)
    assert arg.clock == vec
    kb.vector.increment.assert_not_called()


def test_add_law_proposal_increment_and_merge() -> None:
    kb = KnowledgeBoard()
    kb.vector.increment = MagicMock()
    kb.vector.merge = MagicMock()

    kb.add_law_proposal("test law", agent_id="A", step=2)
    kb.vector.increment.assert_called_once_with("A")
    kb.vector.merge.assert_not_called()

    kb.vector.increment.reset_mock()
    kb.vector.merge.reset_mock()

    vec = {"A": 3}
    kb.add_law_proposal("another", agent_id="A", step=3, vector=vec)

    kb.vector.merge.assert_called_once()
    arg = kb.vector.merge.call_args.args[0]
    assert isinstance(arg, VersionVector)
    assert arg.clock == vec
    kb.vector.increment.assert_not_called()


def test_typed_entry_persists_metadata_and_references() -> None:
    kb = KnowledgeBoard()
    entry = BoardEntry(
        content_full="Important vote",
        entry_type=KnowledgeEntryType.VOTE,
        tags=["governance", "vote", "vote"],
        parent_entry_id="proposal-1",
        governance_rule_id="rule-1",
        reference_metadata={"approve": True, "stance": "approve"},
    )
    kb.add_entry(entry, agent_id="A", step=3)

    stored = kb.get_full_entries()[-1]
    assert stored["entry_type"] == "vote"
    assert stored["tags"] == ["governance", "vote"]
    assert stored["parent_entry_id"] == "proposal-1"
    assert stored["governance_rule_id"] == "rule-1"
    assert stored["reference_metadata"] == {"approve": True, "stance": "approve"}
    serialized = kb.to_dict()["entries"][-1]
    assert serialized["content_display"].startswith("Step 3 (Agent: A):")


def test_read_models_and_snapshot_migration() -> None:
    kb = KnowledgeBoard()
    kb.from_snapshot(
        {
            "entries": [
                {
                    "entry_id": "p1",
                    "step": 1,
                    "agent_id": "A",
                    "entry_type": "proposal",
                    "content_full": "Do the thing",
                    "content_display": "Do the thing",
                    "content_summary": "Do the thing",
                    "reference_metadata": {"parent": "legacy-parent"},
                },
                {
                    "entry_id": "v1",
                    "step": 2,
                    "agent_id": "A",
                    "entry_type": "vote",
                    "content_full": "approve",
                    "content_display": "approve",
                    "content_summary": "approve",
                    "parent_entry_id": "p1",
                    "reference_metadata": {"approve": True, "stance": "approve"},
                },
            ]
        }
    )

    assert kb.get_full_entries()[0]["parent_entry_id"] == "legacy-parent"
    assert kb.get_active_proposals(limit=10)[0]["entry_id"] == "p1"
    status = kb.get_consensus_status("p1")
    assert status["approvals"] == 1
    assert status["consensus"] is True
    assert kb.get_agent_stance_history("A")[0]["entry_type"] == "vote"


def test_non_typed_entry_is_rejected() -> None:
    kb = KnowledgeBoard()
    ok = kb.add_entry("legacy", agent_id="A", step=1)  # type: ignore[arg-type]
    assert ok is False
