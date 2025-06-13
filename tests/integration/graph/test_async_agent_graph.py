import asyncio
import logging
from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import pytest
from pytest import LogCaptureFixture

pytest.importorskip("langgraph")
pytest.importorskip("chromadb")

from src.agents.core.agent_state import AgentActionIntent
from src.agents.core.base_agent import Agent
from src.agents.graphs.basic_agent_graph import AgentTurnState
from src.shared.async_utils import AsyncDSPyManager

logger = logging.getLogger(__name__)


@pytest.fixture
def simple_agent() -> Generator[Agent, None, None]:
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
def async_manager() -> Generator[AsyncDSPyManager, None, None]:
    mgr = AsyncDSPyManager(max_workers=2, default_timeout=0.5)
    yield mgr
    mgr.shutdown()


@pytest.mark.asyncio
async def test_dspy_call_timeout_in_graph(
    simple_agent: Agent, async_manager: AsyncDSPyManager, caplog: LogCaptureFixture
) -> None:
    caplog.set_level(
        logging.WARNING, logger="src.shared.async_utils"
    )  # For timeout test, we expect WARNING
    caplog.set_level(logging.DEBUG)  # General debug for other logs if needed

    # Patch the agent's async_select_action_intent to simulate a slow DSPy call
    async def mock_slow_dspy_call(*args: object, **kwargs: object) -> object:
        await asyncio.sleep(2)  # Exceeds manager timeout of 0.5s
        # This part should not be reached if timeout logic in graph/manager works
        return type(
            "obj",
            (object,),
            {
                "chosen_action_intent": "mock_success_after_sleep",
                "justification_thought": "should have timed out",
            },
        )()

    simple_agent.async_dspy_manager = async_manager

    # The action_intent_selector_program is what gets called by the AsyncDSPyManager
    # We mock this program directly.
    mock_program_callable = AsyncMock(side_effect=mock_slow_dspy_call)

    logger.info(f"Simple agent dict before hasattr: {simple_agent.__dict__}")
    assert hasattr(simple_agent, "action_intent_selector_program"), (
        "Agent should have action_intent_selector_program before patch"
    )

    # Prepare a minimal AgentTurnState
    initial_turn_state = AgentTurnState(
        agent_id=simple_agent.agent_id,
        current_state=simple_agent.state.model_dump(),  # Use Pydantic model_dump for current_state
        simulation_step=1,
        previous_thought=None,
        environment_perception={},
        perceived_messages=[],
        memory_history_list=[],
        turn_sentiment_score=0,
        prompt_modifier="",
        structured_output=None,
        agent_goal=(
            simple_agent.state.goals[0]["description"] if simple_agent.state.goals else "Test Goal"
        ),
        updated_state={},  # Will be populated by graph
        vector_store_manager=None,  # Not strictly needed for this test focus
        rag_summary="(No rag summary for this test)",
        knowledge_board_content=[],
        knowledge_board=None,  # Not strictly needed
        scenario_description="Test scenario for DSPy timeout",
        current_role=simple_agent.state.role,
        influence_points=int(simple_agent.state.ip),
        steps_in_current_role=simple_agent.state.steps_in_current_role,
        data_units=int(simple_agent.state.du),
        current_project_affiliation=simple_agent.state.current_project_affiliation,
        available_projects={},
        state=simple_agent.state,  # Pass the AgentState object
        agent_instance=simple_agent,  # Pass the agent instance
        collective_ip=None,
        collective_du=None,
    )

    with patch.object(simple_agent, "async_select_action_intent", mock_program_callable):
        # Execute the graph
        assert simple_agent.graph is not None, "Agent graph should be initialized by BaseAgent"
        final_state = await simple_agent.graph.ainvoke(initial_turn_state)

        # Assertions:
        # 1. Check if the graph completed (it should, even with internal timeout)
        assert final_state is not None
        # 2. Check that the agent's action reflects a fallback due to timeout
        #    The generate_thought_and_message_node should produce a default/fallback
        #    AgentActionOutput if async_select_action_intent fails or times out.
        final_structured_output = final_state.get("structured_output")
        assert final_structured_output is not None, "Final state should have structured_output"
        # Default fallback is often 'idle'
        assert final_structured_output.action_intent == AgentActionIntent.IDLE.value, (
            f"Expected fallback action_intent to be IDLE due to timeout, got {final_structured_output.action_intent}"
        )
        assert (
            "timeout" in final_structured_output.thought.lower()
            or "fallback" in final_structured_output.thought.lower()
            or "default" in final_structured_output.thought.lower()
        ), f"Expected thought to indicate timeout/fallback, got: {final_structured_output.thought}"


@pytest.mark.asyncio
async def test_dspy_call_exception_in_graph(
    simple_agent: Agent, async_manager: AsyncDSPyManager, caplog: LogCaptureFixture
) -> None:
    caplog.set_level(
        logging.ERROR, logger="src.agents.graphs.basic_agent_graph"
    )  # For exception test, we expect ERROR
    caplog.set_level(logging.DEBUG)  # General debug

    # Patch the agent's async_select_action_intent to raise an exception
    async def mock_error_dspy_call(*args: object, **kwargs: object) -> None:
        logger.error("MOCK_ERROR_DSPY_CALL: Raising ValueError now...")
        raise ValueError("Simulated DSPy Program Error during graph execution")

    simple_agent.async_dspy_manager = async_manager

    mock_program_callable_err = AsyncMock(side_effect=mock_error_dspy_call)

    # Prepare a minimal AgentTurnState
    initial_turn_state = AgentTurnState(
        agent_id=simple_agent.agent_id,
        current_state=simple_agent.state.model_dump(),
        simulation_step=1,
        previous_thought=None,
        environment_perception={},
        perceived_messages=[],
        memory_history_list=[],
        turn_sentiment_score=0,
        prompt_modifier="",
        structured_output=None,
        agent_goal=(
            simple_agent.state.goals[0]["description"] if simple_agent.state.goals else "Test Goal"
        ),
        updated_state={},
        vector_store_manager=None,
        rag_summary="(No rag summary for this test)",
        knowledge_board_content=[],
        knowledge_board=None,
        scenario_description="Test scenario for DSPy exception",
        current_role=simple_agent.state.role,
        influence_points=int(simple_agent.state.ip),
        steps_in_current_role=simple_agent.state.steps_in_current_role,
        data_units=int(simple_agent.state.du),
        current_project_affiliation=simple_agent.state.current_project_affiliation,
        available_projects={},
        state=simple_agent.state,
        agent_instance=simple_agent,
        collective_ip=None,
        collective_du=None,
    )

    with patch.object(simple_agent, "async_select_action_intent", mock_program_callable_err):
        # Execute the graph
        assert simple_agent.graph is not None, "Agent graph should be initialized by BaseAgent"
        final_state = await simple_agent.graph.ainvoke(initial_turn_state)

        # Assertions:
        # 1. Graph completed
        assert final_state is not None
        # 2. Agent's action reflects fallback due to error
        #    The generate_thought_and_message_node should produce a default/fallback
        #    AgentActionOutput if async_select_action_intent raises an exception.
        final_structured_output_exc = final_state.get("structured_output")
        assert final_structured_output_exc is not None, (
            "Final state should have structured_output after exception"
        )
        # Default fallback is often 'idle'
        assert final_structured_output_exc.action_intent == AgentActionIntent.IDLE.value, (
            f"Expected fallback action_intent to be IDLE due to exception, got {final_structured_output_exc.action_intent}"
        )
        assert (
            "error" in final_structured_output_exc.thought.lower()
            or "fallback" in final_structured_output_exc.thought.lower()
            or "exception" in final_structured_output_exc.thought.lower()
            or "default" in final_structured_output_exc.thought.lower()
        ), (
            f"Expected thought to indicate error/fallback, got: {final_structured_output_exc.thought}"
        )

        # 3. Optional: Verify the specific error log from the mock if it appears, but don't fail if not,
        #    as primary check is the fallback action.
        # expected_error_msg_core = f"Agent {simple_agent.agent_id}: Error in async_select_action_intent: Simulated DSPy Program Error during graph execution"
        # found_log = any(
        #     expected_error_msg_core in record.getMessage() and
        #     record.levelname == "ERROR" and
        #     record.name == "src.agents.graphs.basic_agent_graph"
        #     for record in caplog.records
        # )
        # if not found_log:
        #     logger.warning(f"Expected error log for exception test not found in caplog, but testing behavior via fallback action. Expected: '{expected_error_msg_core}'")
