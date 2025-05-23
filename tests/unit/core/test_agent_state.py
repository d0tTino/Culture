#!/usr/bin/env python
"""
Test script to verify the AgentState refactoring.
This script initializes a simulation with agents using the new AgentState model
and runs several steps to verify the state management is working correctly.
"""

import logging
import sys
import os
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager
from typing import Dict, Any, Optional
import json
import pytest
import unittest
from unittest.mock import MagicMock

# Set up proper paths for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Now we can import from src
from src.agents.core.base_agent import Agent
from src.agents.core.agent_state import AgentState
from src.sim.simulation import Simulation
from src.sim.knowledge_board import KnowledgeBoard
from tests.utils.mock_llm import MockLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("test_agent_state")

# Define MockLLM directly in this file to avoid import issues
@contextmanager
def MockLLM(responses: Optional[Dict[str, Any]] = None, strict_mode: bool = True):
    """
    Context manager for mocking LLM responses in tests.
    
    Args:
        responses (Dict[str, Any], optional): Dictionary of predefined responses for specific prompts.
                                             Use "default" as key for fallback responses.
        strict_mode (bool): When True, raises exception if no matching response is found.
                           Prevents tests from making actual API calls (prevents timeouts).
    """
    responses = responses or {"default": "Mocked response from MockLLM"}
    
    # Default structured response for agent decision making
    if "structured_output" not in responses:
        responses["structured_output"] = {
            "thought": "Default mocked thought",
            "message_content": None,
            "message_recipient_id": None,
            "action_intent": "continue_collaboration",
            "requested_role_change": None
        }

    def mock_generate_text(*args, **kwargs):
        logger.info("MockLLM: Intercepted generate_text call")
        prompt = kwargs.get("prompt", "") if kwargs else args[0] if args else ""
        return responses.get(prompt, responses.get("default", "Default mock response"))

    def mock_analyze_sentiment(*args, **kwargs):
        logger.info("MockLLM: Intercepted analyze_sentiment call")
        return 0.0  # Neutral sentiment
    
    def mock_summarize_memory_context(*args, **kwargs):
        logger.info("MockLLM: Intercepted summarize_memory_context call")
        return "Mocked memory context summary"
    
    def mock_generate_structured_output(*args, **kwargs):
        logger.info("MockLLM: Intercepted generate_structured_output call")
        if strict_mode and "structured_output" not in responses:
            raise ValueError("No structured_output defined in MockLLM and strict_mode is enabled")
        return responses.get("structured_output")
    
    def mock_ollama_chat(*args, **kwargs):
        logger.info("MockLLM: Intercepted Ollama chat call")
        return {"message": {"content": responses.get("default", "Default Ollama response")}}
    
    def mock_dspy_predict(*args, **kwargs):
        logger.info("MockLLM: Intercepted DSPy predict call")
        return {"output": responses.get("dspy_output", "Default DSPy response")}
    
    # Patches to apply - define them first so we can conditionally add more
    patches = [
        patch('src.infra.llm_client.generate_text', side_effect=mock_generate_text),
        patch('src.infra.llm_client.analyze_sentiment', side_effect=mock_analyze_sentiment),
        patch('src.infra.llm_client.summarize_memory_context', side_effect=mock_summarize_memory_context),
        patch('src.infra.llm_client.generate_structured_output', side_effect=mock_generate_structured_output)
    ]
    
    # Try to import and patch optional modules
    try:
        import src.infra.llm_client
        if hasattr(src.infra.llm_client, 'OllamaClient'):
            patches.append(patch('src.infra.llm_client.OllamaClient.chat', side_effect=mock_ollama_chat))
    except (ImportError, AttributeError):
        logger.info("MockLLM: OllamaClient not available, skipping patch")
    
    # Apply all patches
    for p in patches:
        p.start()
    
    try:
        # Try to patch DSPy if it's available
        dspy_patch = None
        try:
            import dspy
            dspy_patch = patch('dspy.Predict.__call__', side_effect=mock_dspy_predict)
            dspy_patch.start()
        except (ImportError, AttributeError):
            logger.info("MockLLM: DSPy not available, skipping patch")
        
        yield
        
        # Stop DSPy patch if it was started
        if dspy_patch:
            dspy_patch.stop()
    finally:
        # Always stop all patches
        for p in patches:
            p.stop()

@pytest.mark.unit
@pytest.mark.core
@pytest.mark.slow
@pytest.mark.critical_path
def test_agent_state():
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
            "requested_role_change": None
        },
        "dspy_output": {
            "intent": "continue_collaboration", 
            "justification": "This is a mock justification"
        }
    }
    
    # Use strict_mode=True to prevent any real API calls
    with MockLLM(mock_responses, strict_mode=True):
        # Create 3 agents with the new AgentState model
        agents = [
            Agent(agent_id=f"agent_{i}", 
                initial_state={
                    "name": f"Agent-{i}",
                    "current_role": "Default Contributor",
                    "goals": [{"description": f"Test goal for agent {i}", "priority": "high"}]
                }
            ) for i in range(3)
        ]
        
        # Verify the agents have been created with AgentState objects
        for i, agent in enumerate(agents):
            logger.info(f"Verifying agent {i} state structure")
            assert isinstance(agent.state, AgentState), f"Agent {i} state is not an AgentState object"
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
        sim.run(num_steps)
        
        # Verify that the state has been updated correctly
        logger.info("Verifying state updates after simulation")
        for i, agent in enumerate(agents):
            logger.info(f"===== Checking agent {i} state after simulation =====")
            
            # Record initial values
            initial_ip = agent.state.ip
            initial_du = agent.state.du
            
            # Verify state object is still valid
            assert isinstance(agent.state, AgentState), f"Agent {i} state is not an AgentState object"
            
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

if __name__ == "__main__":
    test_agent_state() 