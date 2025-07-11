import pytest

from src.sim.quests import QUESTS, Quest, generate_quest
from tests.utils.mock_llm import MockLLM

pytestmark = pytest.mark.unit


def test_generate_quest_adds_to_list() -> None:
    QUESTS.clear()
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
        quest = generate_quest("Create quest")
    assert quest == Quest(
        id=1, title="Quest", description="Do something", progress=0, status="pending"
    )
    assert QUESTS == [quest]
