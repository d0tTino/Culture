from unittest.mock import AsyncMock

import pytest

from src.agents.core.base_agent import Agent


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_select_action_intent_failsafe(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert result.chosen_action_intent == "idle"
    assert "Failsafe" in result.justification_thought


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_generate_l1_summary_failsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = Agent(agent_id="a2")
    failsafe_output = "Failsafe: No summary available due to processing error."
    monkeypatch.setattr(
        agent.async_dspy_manager, "get_result", AsyncMock(return_value=failsafe_output)
    )
    result = await agent.async_generate_l1_summary("role", ["event1"], "happy")
    assert "Failsafe" in result
