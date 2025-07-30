from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.sim import quests
from src.sim.quests import Quest


@pytest.mark.asyncio
@pytest.mark.integration
async def test_generate_quest_records(monkeypatch: pytest.MonkeyPatch) -> None:
    quests.QUESTS.clear()

    async def fake_generate(prompt: str, response_model: type[Quest], **_: object) -> dict:
        return {"id": 42, "title": "Quest", "description": "A test quest"}

    monkeypatch.setattr(
        quests.llm_client,
        "async_generate_structured_output",
        AsyncMock(side_effect=fake_generate),
    )
    record_mock = MagicMock()
    fake_ledger = SimpleNamespace(record_quest=record_mock)
    monkeypatch.setattr(quests, "ledger", fake_ledger)

    quest = await quests.generate_quest("make quest")

    assert isinstance(quest, Quest)
    assert quest.id == 42
    record_mock.assert_called_once_with(42, "Quest", "A test quest", 0, "pending")
