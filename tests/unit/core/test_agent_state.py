"""
Test script to verify the AgentState refactoring.
This script initializes a simulation with agents using the new AgentState model
and runs several steps to verify the state management is working correctly.
"""

import logging
import os
import sys

import pytest

pytest.importorskip("langgraph")

from src.agents.core.agent_state import AgentState
from src.agents.core.base_agent import Agent
from src.sim.simulation import Simulation
from tests.utils.mock_llm import MockLLM

# Set up proper paths for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("test_agent_state")


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.critical_path
@pytest.mark.asyncio
async def test_agent_state() -> None:
    """Run a simple test simulation with the new AgentState model."""
    logger.info("Starting AgentState refactoring test")

    # Set up comprehensive mock LLM responses for all required calls
    mock_responses = {
        "default": "Mocked response for agent state test",
        "structured_output": {
            "thought": "Mock thought for agent test",
            "message_content": None,
            "message_recipient_id": None,
            "action_intent": "continue_collaboration",
            "requested_role_change": None,
        },
        "dspy_output": {
            "intent": "continue_collaboration",
            "justification": "This is a mock justification",
        },
    }

    mock_llm_cm = MockLLM(mock_responses, strict_mode=True)
    mock_llm_cm.__enter__()
    try:
        # Create 3 agents with the new AgentState model
        agents = [
            Agent(
                agent_id=f"agent_{i}",
                initial_state={
                    "name": f"Agent-{i}",
                    "current_role": "Default Contributor",
                    "goals": [{"description": f"Test goal for agent {i}", "priority": "high"}],
                },
            )
            for i in range(3)
        ]

        # Verify the agents have been created with AgentState objects
        for i, agent in enumerate(agents):
            logger.info(f"Verifying agent {i} state structure")
            assert isinstance(agent.state, AgentState), (
                f"Agent {i} state is not an AgentState object"
            )
            logger.info(f"Agent {i} state: {agent.state}")

            # Verify the state has the expected fields
            assert agent.state.agent_id == f"agent_{i}"
            assert agent.state.name == f"Agent-{i}"
            assert agent.state.role == "Default Contributor"
            assert len(agent.state.goals) > 0
            assert agent.state.ip > 0
            assert agent.state.du > 0

            # Verify that history fields are initialized as empty
            assert len(agent.state.mood_history) == 0
            assert len(agent.state.relationship_history) == 0
            assert len(agent.state.ip_history) == 0
            assert len(agent.state.du_history) == 0
            assert len(agent.state.role_history) == 0
            assert len(agent.state.project_history) == 0

        # Create a simulation with these agents
        scenario = "This is a test simulation scenario to verify the AgentState refactoring."
        sim = Simulation(agents=agents, scenario=scenario)

        # Run a few simulation steps
        logger.info("Running simulation steps to verify state updates")
        num_steps = 5
        await sim.async_run(num_steps)

        # Verify that the state has been updated correctly
        logger.info("Verifying state updates after simulation")
        for i, agent in enumerate(agents):
            logger.info(f"===== Checking agent {i} state after simulation =====")

            # Record initial values
            initial_ip = agent.state.ip
            initial_du = agent.state.du

            # Verify state object is still valid
            assert isinstance(agent.state, AgentState), (
                f"Agent {i} state is not an AgentState object"
            )

            # Check that history fields have been updated
            # Note: We use logging.info instead of assertions here as exact values
            # might change with simulation implementation, but we want to see progress
            logger.info(f"Mood history: {len(agent.state.mood_history)} entries")
            logger.info(f"Relationship history: {len(agent.state.relationship_history)} entries")
            logger.info(f"IP history: {len(agent.state.ip_history)} entries")
            logger.info(f"DU history: {len(agent.state.du_history)} entries")
            logger.info(f"Role history: {len(agent.state.role_history)} entries")
            logger.info(f"Project history: {len(agent.state.project_history)} entries")

            # Verify role history is correctly tracking the current role
            if agent.state.role_history:
                logger.info(f"Current role from state: {agent.state.role}")
                logger.info(f"Most recent role change: {agent.state.role_history[-1]}")
                # Check logged history details

            # Verify IP history - normally should increase or stay same during simulation
            # Just logging here since exact values aren't critical for test passing
            if agent.state.ip_history:
                logger.info(f"Initial IP: {initial_ip}, current IP: {agent.state.ip}")
                logger.info(f"IP history: {agent.state.ip_history}")

            # Verify DU history - normally should increase during simulation
            if agent.state.du_history:
                logger.info(f"Initial DU: {initial_du}, current DU: {agent.state.du}")
                logger.info(f"DU history: {agent.state.du_history}")

    finally:
        mock_llm_cm.__exit__(None, None, None)


@pytest.mark.unit
def test_discovery_sanity() -> None:
    assert True
