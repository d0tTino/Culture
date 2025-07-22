from pathlib import Path

import pytest

from src.infra.ledger import Ledger
from src.sim import quests
from src.sim.quests import Quest
from tests.utils.mock_llm import MockLLM

pytestmark = pytest.mark.unit


def test_generate_quest_adds_to_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    quests.QUESTS.clear()
    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr(quests, "ledger", ledger)
    responses = {
        "structured_output": {
            "id": 1,
            "title": "Quest",
            "description": "Do something",
            "progress": 0,
            "status": "pending",
        }
    }
    with MockLLM(responses):
        quest = quests.generate_quest("Create quest")
    assert quest == Quest(
        id=1, title="Quest", description="Do something", progress=0, status="pending"
    )
    assert quests.QUESTS == [quest]
    assert ledger.get_quests()[0]["title"] == "Quest"
