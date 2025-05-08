#!/usr/bin/env python
"""
Test script to verify the passive role-based DU generation logic
and the collective IP/DU tracking and agent perception.
Creates agents with different roles and runs a simulation for multiple steps
to confirm that DU is generated passively and that collective metrics are tracked.
"""

import logging
import sys
import time
from src.app import create_base_simulation
from src.agents.core.roles import ROLE_INNOVATOR, ROLE_ANALYZER, ROLE_FACILITATOR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("collective_metrics_evaluation.log")  # Save logs to file for analysis
    ]
)

# Set more verbose logging for specific modules
logging.getLogger('src.sim.simulation').setLevel(logging.DEBUG)
logging.getLogger('src.agents.graphs.basic_agent_graph').setLevel(logging.DEBUG)
# Ensure we capture agent thoughts and decisions
logging.getLogger('src.agents.core.base_agent').setLevel(logging.DEBUG)

logger = logging.getLogger("test_collective_metrics")

def test_collective_metrics():
    """
    Tests that the collective IP/DU tracking and agent perception works properly.
    Creates agents with different roles and runs a simulation to verify:
    1. Passive role-based DU generation
    2. Collective IP/DU tracking
    3. Agent perception of collective metrics
    """
    logger.info("Starting collective metrics tracking and perception test")
    
    # Create a test scenario that will encourage agents to propose ideas
    test_scenario = """
    COLLECTIVE METRICS EVALUATION SCENARIO:
    
    This is a simulation to evaluate how agents consider collective metrics in their decision-making.
    
    The goal is to maximize both individual and collective resources through strategic collaboration.
    
    Each agent should:
    1. Consider the collective IP and DU metrics when deciding actions
    2. Propose ideas that benefit the group as a whole
    3. Perform analyses that improve others' proposals 
    4. Collaborate in ways that increase overall simulation resources
    5. Make strategic role changes if needed to benefit collective outcomes
    
    Pay special attention to how your actions affect not just your own resources but the total resources
    of the entire simulation.
    """
    
    # Create a simulation with 3 agents, each with a different role
    sim = create_base_simulation(
        num_agents=3,
        use_vector_store=True,  # This will create a ChromaVectorStoreManager with default settings
        scenario=test_scenario,
        steps=15  # Run for 15 steps to observe longer-term behavior
    )
    
    # Set different roles for each agent
    agents = sim.agents  # sim.agents is already a list
    agents[0].state.role = ROLE_INNOVATOR
    agents[1].state.role = ROLE_ANALYZER
    agents[2].state.role = ROLE_FACILITATOR
    
    # Ensure agents have enough initial IP to propose ideas
    for agent in agents:
        agent.state.ip = 10.0  # Give enough IP to propose ideas and consider role changes
    
    # Log the initial state of each agent
    logger.info("Initial agent states:")
    for agent in agents:
        logger.info(f"Agent {agent.agent_id}: Role={agent.state.role}, IP={agent.state.ip}, DU={agent.state.du}")
    
    # Log the initial collective metrics
    logger.info(f"Initial collective metrics - IP: {sim.collective_ip:.1f}, DU: {sim.collective_du:.1f}")
    
    # Run the simulation for the specified steps
    logger.info("Running simulation...")
    sim.run(sim.steps_to_run)
    
    # Log the final state of each agent
    logger.info("Final agent states:")
    for agent in agents:
        logger.info(f"Agent {agent.agent_id}: Role={agent.state.role}, IP={agent.state.ip}, DU={agent.state.du}")
    
    # Log the final collective metrics
    logger.info(f"Final collective metrics - IP: {sim.collective_ip:.1f}, DU: {sim.collective_du:.1f}")
    
    logger.info("Collective metrics tracking and perception test complete")

if __name__ == "__main__":
    test_collective_metrics() 