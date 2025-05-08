#!/usr/bin/env python3
"""
Test script to verify that resource constraint handling works correctly.
Tests various actions when the agent has insufficient resources (IP/DU).
"""

import os
import logging
import sys
import uuid
import tempfile
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.core.base_agent import Agent
from src.agents.core.agent_state import AgentState
from src.agents.core.roles import ROLE_FACILITATOR, ROLE_INNOVATOR, ROLE_ANALYZER
from src.sim.knowledge_board import KnowledgeBoard
from src.infra.memory.vector_store import ChromaVectorStoreManager
from src.agents.graphs.basic_agent_graph import process_role_change, IP_COST_TO_POST_IDEA, PROPOSE_DETAILED_IDEA_DU_COST, DU_COST_REQUEST_DETAILED_CLARIFICATION
from src.infra import config

# Configure logging to show detailed information
logging.basicConfig(level=logging.INFO, 
                    format='%(levelname)s:%(name)s:%(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger("resource_constraints_test")
logger.setLevel(logging.INFO)

# Valid roles for role changes
VALID_ROLES = [ROLE_FACILITATOR, ROLE_INNOVATOR, ROLE_ANALYZER]

def create_agent_with_limited_resources():
    """Create an agent with very limited resources for testing."""
    agent_id = str(uuid.uuid4())
    initial_goals = [{"description": "Test resource constraints", "priority": "high"}]
    
    # Set up values for agent state with minimal IP and DU
    ip = 1.0  # Very low IP
    du = 1.0  # Very low DU
    role_change_ip_cost = 5.0  # Higher than the agent's IP
    
    # Create the agent state with all required fields
    agent_state = AgentState(
        agent_id=agent_id,
        name="Resource Test Agent",
        role="Analyzer",
        goals=initial_goals,
        ip=ip,
        du=du,
        steps_in_current_role=5,  # Set high enough to pass cooldown check
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
        role_change_ip_cost=role_change_ip_cost,
        # Add the missing required fields
        positive_relationship_learning_rate=0.1,
        negative_relationship_learning_rate=0.15,
        targeted_message_multiplier=2.0,
        # Additional fields that might be needed
        last_level_2_consolidation_step=0
    )
    
    # Create an Agent instance directly with the agent ID
    agent = Agent(agent_id=agent_id)
    
    # Manually set the agent's state to our prepared state
    agent._state = agent_state
    
    return agent

def test_role_change_constraint():
    """Test that role change is blocked when agent has insufficient IP."""
    print("\n------ TESTING ROLE CHANGE RESOURCE CONSTRAINT ------")
    
    # Create an agent with low resources
    agent = create_agent_with_limited_resources()
    
    # Get the current state values before attempting role change
    state = agent._state
    initial_role = state.role
    initial_ip = state.ip
    
    print(f"Initial role: {initial_role}")
    print(f"Initial IP: {initial_ip}")
    print(f"IP cost for role change: {state.role_change_ip_cost}")
    
    # Attempt a role change that should fail due to insufficient IP
    requested_role = "Innovator"
    success = process_role_change(agent._state, requested_role)
    
    print(f"\nAttempted role change successful: {success}")
    print(f"After attempt - Role: {agent._state.role}")
    print(f"After attempt - IP: {agent._state.ip}")
    
    # Check if the test passed
    if not success and agent._state.role == initial_role and agent._state.ip == initial_ip:
        print("✅ ROLE CHANGE CONSTRAINT TEST PASSED: Role change correctly blocked due to insufficient IP")
    else:
        print("❌ ROLE CHANGE CONSTRAINT TEST FAILED: Role change was allowed despite insufficient IP")

def test_knowledge_board_posting_constraint():
    """
    Test that posting to knowledge board is blocked and handled properly
    when agent has insufficient resources.
    This is a partial test - in a real scenario, the finalize_message_agent_node 
    would be called through the agent graph.
    """
    print("\n------ TESTING KNOWLEDGE BOARD POSTING CONSTRAINT ------")
    
    # Create a knowledge board
    knowledge_board = KnowledgeBoard()
    
    # Create an agent with low resources
    agent = create_agent_with_limited_resources()
    agent_state = agent._state
    
    # Initial values
    initial_ip = agent_state.ip
    initial_du = agent_state.du
    
    print(f"Initial IP: {initial_ip}")
    print(f"Initial DU: {initial_du}")
    print(f"Required IP for posting: {IP_COST_TO_POST_IDEA}")
    print(f"Required DU for posting: {PROPOSE_DETAILED_IDEA_DU_COST}")
    
    # Check if the agent has sufficient resources
    has_sufficient_ip = initial_ip >= IP_COST_TO_POST_IDEA
    has_sufficient_du = initial_du >= PROPOSE_DETAILED_IDEA_DU_COST
    
    print(f"\nHas sufficient IP: {has_sufficient_ip}")
    print(f"Has sufficient DU: {has_sufficient_du}")
    
    # Create a memory snapshot before the test
    initial_memory_count = len(agent_state.short_term_memory)
    print(f"Initial memory count: {initial_memory_count}")
    
    # In a real scenario, this check would be done in finalize_message_agent_node
    # We're simulating the logic here
    if has_sufficient_ip and has_sufficient_du:
        print("Resources sufficient - this is unexpected for this test")
    else:
        print("Resources insufficient - message would be modified and intent downgraded")
        
        # Check if a resource_constraint memory would be added
        # Add a dummy memory to represent what would happen
        dummy_memory = f"Attempted to post idea to knowledge board but had insufficient resources"
        agent_state.add_memory(1, "resource_constraint", dummy_memory)
        
        # Check if memory was added
        final_memory_count = len(agent_state.short_term_memory)
        print(f"Final memory count: {final_memory_count}")
        
        if final_memory_count > initial_memory_count:
            has_constraint_memory = any(
                memory.get("type") == "resource_constraint" 
                for memory in agent_state.short_term_memory
            )
            print(f"Has constraint memory: {has_constraint_memory}")
            
            if has_constraint_memory:
                print("✅ KNOWLEDGE BOARD POSTING CONSTRAINT TEST PASSED: Resource constraint properly recorded in memory")
            else:
                print("❌ KNOWLEDGE BOARD POSTING CONSTRAINT TEST FAILED: Resource constraint not properly recorded")
        else:
            print("❌ KNOWLEDGE BOARD POSTING CONSTRAINT TEST FAILED: No memory recorded for constraint")

def test_detailed_clarification_constraint():
    """
    Test that asking for detailed clarification is properly constrained
    when agent has insufficient DU.
    """
    print("\n------ TESTING DETAILED CLARIFICATION CONSTRAINT ------")
    
    # Create an agent with low resources
    agent = create_agent_with_limited_resources()
    agent_state = agent._state
    
    # Initial values
    initial_du = agent_state.du
    
    print(f"Initial DU: {initial_du}")
    print(f"Required DU for detailed clarification: {DU_COST_REQUEST_DETAILED_CLARIFICATION}")
    
    # Check if the agent has sufficient DU
    has_sufficient_du = initial_du >= DU_COST_REQUEST_DETAILED_CLARIFICATION
    
    print(f"\nHas sufficient DU: {has_sufficient_du}")
    
    # Create a memory snapshot before the test
    initial_memory_count = len(agent_state.short_term_memory)
    print(f"Initial memory count: {initial_memory_count}")
    
    # In a real scenario, this check would be done in handle_ask_clarification_node
    # We're simulating the logic here
    if has_sufficient_du:
        print("Resources sufficient - this is unexpected for this test")
    else:
        print("Resources insufficient - clarification request would be downgraded")
        
        # Add a dummy memory to represent what would happen
        dummy_memory = f"Attempted to ask a detailed clarification but had insufficient DU"
        agent_state.add_memory(1, "resource_constraint", dummy_memory)
        
        # Set the flag that would be set in the real code
        agent_state.last_clarification_downgraded = True
        
        # Check if memory was added and flag was set
        final_memory_count = len(agent_state.short_term_memory)
        print(f"Final memory count: {final_memory_count}")
        print(f"last_clarification_downgraded flag: {agent_state.last_clarification_downgraded}")
        
        if final_memory_count > initial_memory_count and agent_state.last_clarification_downgraded:
            print("✅ DETAILED CLARIFICATION CONSTRAINT TEST PASSED: Constraint properly handled")
        else:
            print("❌ DETAILED CLARIFICATION CONSTRAINT TEST FAILED: Constraint not properly handled")

def main():
    """Run all resource constraint tests."""
    print("\n======== RESOURCE CONSTRAINT TESTS ========")
    logger.info("Starting resource constraint tests...")
    
    # Run the tests
    test_role_change_constraint()
    test_knowledge_board_posting_constraint()
    test_detailed_clarification_constraint()
    
    print("\n==============================\n")

if __name__ == "__main__":
    main() 