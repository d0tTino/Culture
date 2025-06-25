from unittest.mock import MagicMock

import pytest

from src.sim.knowledge_board import KnowledgeBoard
from src.sim.version_vector import VersionVector

pytestmark = pytest.mark.unit


def _create_board(num: int) -> KnowledgeBoard:
    kb = KnowledgeBoard()
    for i in range(num):
        kb.add_entry(f"entry{i}", agent_id="A", step=i)
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
    kb.add_entry("entry", agent_id="A", step=1)
    kb.entries[-1]["content_summary"] = None
    result = kb.get_recent_entries_for_prompt(1)
    assert result == ["[Step 1, A]: entry"]


def test_add_entry_calls_increment_when_no_vector() -> None:
    kb = KnowledgeBoard()
    kb.vector.increment = MagicMock()
    kb.vector.merge = MagicMock()

    kb.add_entry("entry", agent_id="A", step=1)

    kb.vector.increment.assert_called_once_with("A")
    kb.vector.merge.assert_not_called()


def test_add_entry_calls_merge_when_vector_supplied() -> None:
    kb = KnowledgeBoard()
    kb.vector.increment = MagicMock()
    kb.vector.merge = MagicMock()

    vec = {"B": 2}
    kb.add_entry("entry", agent_id="A", step=1, vector=vec)

    kb.vector.merge.assert_called_once()
    arg = kb.vector.merge.call_args.args[0]
    assert isinstance(arg, VersionVector)
    assert arg.clock == vec
    kb.vector.increment.assert_not_called()


def test_add_law_proposal_increment_and_merge(monkeypatch: pytest.MonkeyPatch) -> None:
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
