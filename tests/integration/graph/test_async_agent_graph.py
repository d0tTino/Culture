import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.core.base_agent import Agent
from src.agents.graphs.basic_agent_graph import basic_agent_graph_compiled
from src.utils.async_dspy_manager import AsyncDSPyManager


@pytest.fixture
def simple_agent():
    # Minimal agent with a mockable action_intent_selector_program
    agent = Agent(
        agent_id="agent_1",
        initial_state={
            "name": "Agent-1",
            "current_role": "Innovator",
            "goals": [{"description": "Test goal", "priority": "high"}],
        },
    )
    yield agent


@pytest.fixture
def async_manager():
    mgr = AsyncDSPyManager(max_workers=2, default_timeout=0.5)
    yield mgr
    mgr.shutdown()


@pytest.mark.asyncio
async def test_dspy_call_timeout_in_graph(
    simple_agent: Agent, async_manager: AsyncDSPyManager, caplog: pytest.LogCaptureFixture
) -> None:
    # Patch the agent's async_select_action_intent to simulate a slow DSPy call
    async def mock_slow_dspy_call(*args: object, **kwargs: object) -> type:
        await asyncio.sleep(2)  # Exceeds manager timeout
        return type(
            "obj",
            (object,),
            {"chosen_action_intent": "mock_success_after_sleep", "justification_thought": "..."},
        )()

    with patch.object(
        simple_agent, "async_select_action_intent", AsyncMock(side_effect=mock_slow_dspy_call)
    ):
        simple_agent.async_dspy_manager = async_manager
        # Prepare minimal graph state
        state = {
            "agent_id": simple_agent.agent_id,
            "agent_instance": simple_agent,
            "state": simple_agent.state,
            "simulation_step": 1,
            "current_role": "Innovator",
            "agent_goal": "Test goal",
            "environment_perception": {},
            "perceived_messages": [],
            "turn_sentiment_score": 0,
            "prompt_modifier": "",
            "rag_summary": "",
            "knowledge_board_content": [],
            "scenario_description": "",
        }
        # Run the graph for one step (ainvoke)
        with caplog.at_level("WARNING"):
            result = await basic_agent_graph_compiled.ainvoke(state)
        # The action_intent_selector should have timed out and used failsafe
        action = result.get("structured_output")
        assert action is not None
        assert getattr(action, "action_intent", None) == "idle"
        # NOTE: Log capture for the timeout warning may not be reliable in this integration context.


@pytest.mark.asyncio
async def test_dspy_call_exception_in_graph(
    simple_agent: Agent, async_manager: AsyncDSPyManager, caplog: pytest.LogCaptureFixture
) -> None:
    # Patch the agent's async_select_action_intent to raise an exception
    async def mock_error_dspy_call(*args: object, **kwargs: object) -> None:
        raise ValueError("Simulated DSPy Program Error during graph execution")

    with patch.object(
        simple_agent, "async_select_action_intent", AsyncMock(side_effect=mock_error_dspy_call)
    ):
        simple_agent.async_dspy_manager = async_manager
        state = {
            "agent_id": simple_agent.agent_id,
            "agent_instance": simple_agent,
            "state": simple_agent.state,
            "simulation_step": 1,
            "current_role": "Innovator",
            "agent_goal": "Test goal",
            "environment_perception": {},
            "perceived_messages": [],
            "turn_sentiment_score": 0,
            "prompt_modifier": "",
            "rag_summary": "",
            "knowledge_board_content": [],
            "scenario_description": "",
        }
        with caplog.at_level("ERROR"):
            result = await basic_agent_graph_compiled.ainvoke(state)
        action = result.get("structured_output")
        assert action is not None
        assert getattr(action, "action_intent", None) == "idle"
        # Check that an error was logged
        assert any(
            "raised exception" in r or "error" in r.lower() for r in caplog.text.splitlines()
        )
