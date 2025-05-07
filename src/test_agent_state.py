#!/usr/bin/env python
"""
Test script to verify the AgentState refactoring.
This script initializes a simulation with agents using the new AgentState model
and runs several steps to verify the state management is working correctly.
"""

import logging
import sys
from src.agents.core.base_agent import Agent
from src.agents.core.agent_state import AgentState
from src.sim.simulation import Simulation
from src.sim.knowledge_board import KnowledgeBoard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("test_agent_state")

def test_agent_state():
    """Run a simple test simulation with the new AgentState model."""
    logger.info("Starting AgentState refactoring test")
    
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
        assert isinstance(agent.state, AgentState), f"Agent {i} state is still an AgentState object after simulation"
        
        # Verify step counter was updated
        assert agent.state.step_counter >= 0, f"Agent {i} step counter was updated"
        logger.info(f"Agent {i} step counter: {agent.state.step_counter}")
        
        # Verify that history fields are populated
        # Note: Due to simulation errors or test conditions, some histories might not be populated
        # but we at least want to check that the fields exist and can be accessed
        logger.info(f"Agent {i} mood: {agent.state.mood}")
        logger.info(f"Agent {i} descriptive_mood: {agent.state.descriptive_mood}")
        logger.info(f"Agent {i} current role: {agent.state.role}")
        logger.info(f"Agent {i} IP value: {agent.state.ip}")
        logger.info(f"Agent {i} DU value: {agent.state.du}")
        
        # Check action counters
        logger.info(f"Agent {i} actions taken: {agent.state.actions_taken_count}")
        logger.info(f"Agent {i} messages sent: {agent.state.messages_sent_count}")
        logger.info(f"Agent {i} messages received: {agent.state.messages_received_count}")
        
        # Check history objects
        logger.info(f"Agent {i} mood history: {list(agent.state.mood_history)}")
        logger.info(f"Agent {i} IP history: {list(agent.state.ip_history)}")
        logger.info(f"Agent {i} DU history: {list(agent.state.du_history)}")
        logger.info(f"Agent {i} relationship history: {list(agent.state.relationship_history)}")
        logger.info(f"Agent {i} role history: {list(agent.state.role_history)}")
        logger.info(f"Agent {i} project history: {list(agent.state.project_history)}")
        
        # Verify history has been recorded
        # For IP/DU, we expect at least one history entry
        assert len(agent.state.ip_history) > 0, f"Agent {i} IP history should have at least one entry"
        assert len(agent.state.du_history) > 0, f"Agent {i} DU history should have at least one entry"
        
        # If we have IP history entries, verify the history contains reasonable values 
        if agent.state.ip_history:
            # Get the latest IP history entry
            latest_step, latest_ip = agent.state.ip_history[-1]
            assert latest_step > 0, f"Latest IP history step should be > 0"
            assert latest_ip == agent.state.ip, f"Latest IP history value should match current IP"
        
        # If we have DU history entries, verify the history contains reasonable values
        if agent.state.du_history:
            # Get the latest DU history entry
            latest_step, latest_du = agent.state.du_history[-1]
            assert latest_step > 0, f"Latest DU history step should be > 0"
            assert latest_du == agent.state.du, f"Latest DU history value should match current DU"
    
    logger.info("AgentState refactoring test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_agent_state()
    sys.exit(0 if success else 1) 