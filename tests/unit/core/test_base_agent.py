from unittest.mock import AsyncMock

import pytest
from pytest import MonkeyPatch

from src.agents.core.base_agent import Agent

pytest.importorskip("langgraph")
pytest.importorskip("chromadb")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_select_action_intent_failsafe(monkeypatch: MonkeyPatch) -> None:
    agent = Agent(agent_id="a1")
    failsafe_output = type(
        "Failsafe",
        (),
        {"chosen_action_intent": "idle", "justification_thought": "Failsafe: error"},
    )()
    # Patch AsyncDSPyManager.get_result to return failsafe
    monkeypatch.setattr(
        agent.async_dspy_manager, "get_result", AsyncMock(return_value=failsafe_output)
    )
    result = await agent.async_select_action_intent("role", "context", "goal", ["idle"])
    assert getattr(result, "chosen_action_intent", None) == "idle"
    assert "Failsafe" in getattr(result, "justification_thought", "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_generate_l1_summary_failsafe(monkeypatch: MonkeyPatch) -> None:
    agent = Agent(agent_id="a2")
    failsafe_output = "Failsafe: No summary available due to processing error."
    monkeypatch.setattr(
        agent.async_dspy_manager, "get_result", AsyncMock(return_value=failsafe_output)
    )
    result = await agent.async_generate_l1_summary("role", "event1", "happy")
    if isinstance(result, str):
        assert "Failsafe" in result
    else:
        assert False, "Result is not a string as expected for failsafe output"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_select_action_intent_includes_trait_summary(monkeypatch: MonkeyPatch) -> None:
    agent = Agent(agent_id="a3")
    captured: dict[str, object] = {}

    async def fake_submit(_callable: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "future"

    monkeypatch.setattr(agent.async_dspy_manager, "submit", fake_submit)
    monkeypatch.setattr(
        agent.async_dspy_manager,
        "get_result",
        AsyncMock(return_value=type("R", (), {"chosen_action_intent": "idle"})()),
    )

    await agent.async_select_action_intent("role", "context", "goal", ["idle"])

    assert "traits_summary" in captured
    assert isinstance(captured["traits_summary"], str)
    assert "trait_policy_biases" in captured
    assert isinstance(captured["trait_policy_biases"], dict)
