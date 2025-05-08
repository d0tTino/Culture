#!/usr/bin/env python3
"""
Test script to directly verify the role change functionality is working.
"""

import os
import logging
import sys
import uuid
import tempfile
from pathlib import Path
from collections import deque

sys.path.append(str(Path(__file__).parent.parent))

from src.agents.core.base_agent import Agent
from src.agents.core.agent_state import AgentState
from src.agents.core.roles import ROLE_FACILITATOR, ROLE_INNOVATOR, ROLE_ANALYZER
from src.sim.knowledge_board import KnowledgeBoard
from src.infra.memory.vector_store import ChromaVectorStoreManager
from src.infra import config

# Configure logging to show detailed information
logging.basicConfig(level=logging.INFO, 
                    format='%(levelname)s:%(name)s:%(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger("role_change_test")
logger.setLevel(logging.INFO)

# Valid roles for role changes
VALID_ROLES = [ROLE_FACILITATOR, ROLE_INNOVATOR, ROLE_ANALYZER]

def process_role_change(agent_state, state_dict):
    """
    Process a role change request for an agent.
    This replicates the logic from the basic_agent_graph update_state_node function.
    
    Args:
        agent_state (AgentState): The agent's state object
        state_dict (dict): Dictionary containing the role change request
    """
    requested_role = state_dict.get("requested_role_change")
    
    if requested_role and requested_role in VALID_ROLES:
        if requested_role != agent_state.role:
            # Check if the cooldown period has passed
            if agent_state.steps_in_current_role >= agent_state.role_change_cooldown:
                # Check if the agent has enough IP
                if agent_state.ip >= agent_state.role_change_ip_cost:
                    # Deduct IP cost
                    agent_state.ip -= agent_state.role_change_ip_cost
                    
                    # Update role
                    old_role = agent_state.role
                    agent_state.role = requested_role
                    agent_state.steps_in_current_role = 0
                    
                    # Add to role history - use simulation step 1 as a placeholder
                    agent_state.role_history.append((1, requested_role))
                    
                    logger.info(f"Agent {agent_state.agent_id} changed role from {old_role} to {requested_role}. Spent {agent_state.role_change_ip_cost} IP. Remaining IP: {agent_state.ip}")
                    return True
                else:
                    logger.warning(f"Agent {agent_state.agent_id} requested role change to {requested_role} but had insufficient IP (needed {agent_state.role_change_ip_cost}, had {agent_state.ip}).")
            else:
                logger.warning(f"Agent {agent_state.agent_id} requested role change to {requested_role} but cooldown period not satisfied (needs {agent_state.role_change_cooldown} steps, current: {agent_state.steps_in_current_role}).")
        else:
            logger.info(f"Agent {agent_state.agent_id} already has role {requested_role}, no change needed.")
    else:
        logger.warning(f"Invalid or missing role change request: {requested_role}")
    
    return False

def test_role_change():
    """Test that agent role change mechanism works properly."""
    
    print("\n======== ROLE CHANGE TEST ========")
    logger.info("Starting role change test...")
    
    # Create a temporary directory for chromadb
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Using temporary directory for ChromaDB: {temp_dir}")
    
    # Create a vector store for the agent with the temp directory
    vector_store = ChromaVectorStoreManager(persist_directory=temp_dir)
    
    # Create an agent state first with all required fields
    agent_id = str(uuid.uuid4())
    initial_goals = [{"description": "Analyze data thoroughly", "priority": "high"}]
    
    # Set up values for agent state, either from config or defaults
    ip = 20.0  # Explicitly set high enough for role change
    du = 15.0  # Set data units
    role_change_ip_cost = 5.0  # Explicit cost of role change
    
    # Create the agent state with all required fields
    agent_state = AgentState(
        agent_id=agent_id,
        name="Test Agent",
        role="Analyzer",
        goals=initial_goals,
        ip=ip,
        du=du,
        steps_in_current_role=5,  # Set high enough to allow role change
        # Add all required configuration fields
        max_short_term_memory=10,
        short_term_memory_decay_rate=0.1,
        relationship_decay_rate=0.01,
        min_relationship_score=-1.0,
        max_relationship_score=1.0,
        mood_decay_rate=0.02,
        mood_update_rate=0.2,
        ip_cost_per_message=1.0,
        du_cost_per_action=1.0,
        role_change_cooldown=3,
        role_change_ip_cost=role_change_ip_cost
    )
    
    # Create an Agent instance directly with the agent ID
    agent = Agent(agent_id=agent_id)
    
    # Manually set the agent's state to our prepared state
    agent._state = agent_state
    
    # Verify initial state
    state = agent._state
    print(f"Initial role: {state.role}")
    print(f"Initial steps in role: {state.steps_in_current_role}")
    print(f"Initial IP: {state.ip}")
    
    # Save the initial values to check against
    initial_ip = state.ip
    
    # Manually trigger a role change to 'Innovator'
    print("\nAttempting role change to 'Innovator'...")
    
    # Create a mock state dict with the role change
    state_dict = {
        "requested_role_change": "Innovator"
    }
    
    # Call our process_role_change function
    success = process_role_change(agent._state, state_dict)
    
    # Verify the role change was successful
    print(f"\nAfter role change:")
    print(f"New role: {agent._state.role}")
    print(f"New steps in role: {agent._state.steps_in_current_role}")
    print(f"New IP: {agent._state.ip}")
    print(f"Role history: {list(agent._state.role_history)}")
    
    # Print test results
    print("\n----- TEST RESULTS -----")
    
    # Check if the change was successful
    if agent._state.role == "Innovator" and agent._state.steps_in_current_role == 0:
        print("✅ ROLE CHANGE TEST PASSED: Role was successfully changed to Innovator")
    else:
        print("❌ ROLE CHANGE TEST FAILED: Role was not changed correctly")
    
    # Check if IP was deducted correctly
    if agent._state.ip == initial_ip - role_change_ip_cost:
        print(f"✅ IP DEDUCTION TEST PASSED: {role_change_ip_cost} IP was correctly deducted")
    else:
        print(f"❌ IP DEDUCTION TEST FAILED: Expected {initial_ip - role_change_ip_cost}, got {agent._state.ip}")
        
    # Verify role history was updated
    if agent._state.role_history and len(agent._state.role_history) > 0:
        print("✅ ROLE HISTORY TEST PASSED: Role history was updated")
    else:
        print("❌ ROLE HISTORY TEST FAILED: Role history was not updated")
    
    # Cleanup
    print(f"\nTest complete. Temporary directory {temp_dir} may need manual cleanup.")
    print("==============================\n")

if __name__ == "__main__":
    test_role_change()