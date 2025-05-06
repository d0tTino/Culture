# src/app.py
"""
Main application entry point for the Culture simulation environment.
Initializes and runs the simulation.
"""

import logging
import time
import os
from src.infra import config # Import the configuration module (loads .env)
from src.agents.core.base_agent import Agent # Import the base Agent class
from src.sim.simulation import Simulation # Import the Simulation class
from src.infra.memory.vector_store import ChromaVectorStoreManager # Import the ChromaVectorStoreManager class

# Define simulation scenario
SIMULATION_SCENARIO = "The team's objective is to collaboratively design a specification for a decentralized communication protocol suitable for autonomous AI agents operating in a resource-constrained environment. Key considerations are efficiency, security, and scalability."

# Configure logging with DEBUG level for more detailed output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Set specific loggers to INFO to reduce noise from external libraries
logging.getLogger('urllib3').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.getLogger('sentence_transformers').setLevel(logging.INFO)  # Reduce noise from embedding model
logging.getLogger('chromadb').setLevel(logging.INFO)  # Reduce noise from chromadb
# But keep our own modules at DEBUG level
logging.getLogger('src').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# --- Constants ---
NUM_AGENTS = 3
SIMULATION_STEPS = 7  # Back to 7 steps for final verification
CHROMA_DB_PATH = "./chroma_db_store"  # Path for vector DB persistence

def main():
    """
    Main function to start the Culture application.
    """
    logger.info("+" * 50)
    logger.info(" Starting Culture: An AI Genesis Engine")
    logger.info("+" * 50)
    logger.info("Configuration loaded successfully.") # From config.py import

    # --- Initialize Vector Store Manager ---
    # Optionally clear old data for a fresh start
    if os.path.exists(CHROMA_DB_PATH):
        logger.info(f"Clearing existing ChromaDB data at {CHROMA_DB_PATH}")
        import shutil
        try:
            shutil.rmtree(CHROMA_DB_PATH)
            logger.info(f"Cleared ChromaDB directory: {CHROMA_DB_PATH}")
        except Exception as e:
            logger.warning(f"Failed to clear ChromaDB directory: {e}")
    
    # Initialize the vector store manager
    logger.info(f"Initializing ChromaVectorStoreManager with path: {CHROMA_DB_PATH}")
    vector_store_manager = ChromaVectorStoreManager(persist_directory=CHROMA_DB_PATH)
    logger.info("Vector store manager initialized successfully.")

    # --- Initialize Agents ---
    logger.info(f"Initializing {NUM_AGENTS} agents...")
    agents = []
    
    # Define agent-specific goals
    agent_goals = [
        "Facilitate effective collaboration and ensure all ideas are heard.",
        "Propose innovative technical solutions to problems.",
        "Analyze proposals critically and identify potential flaws or improvements."
    ]
    
    for i in range(NUM_AGENTS):
        # Give agents slightly different initial states for variety later
        initial_state = {
            'name': f'Agent_{i+1}', 
            'mood': 'neutral', 
            'step_counter': 0,
            'goal': agent_goals[i] if i < len(agent_goals) else f"Contribute positively to the group discussion (Agent {i+1})"
        }
        agent = Agent(initial_state=initial_state)
        agents.append(agent)
        time.sleep(0.1)  # Small delay for clearer logging
    logger.info("Agents initialized with unique goals.")

    # --- Initialize Simulation ---
    logger.info("Initializing simulation environment...")
    simulation = Simulation(agents=agents, vector_store_manager=vector_store_manager, scenario=SIMULATION_SCENARIO)
    logger.info("Simulation initialized with vector store manager.")

    # --- Run Simulation ---
    logger.info(f"Starting simulation run for {SIMULATION_STEPS} steps...")
    simulation.run(num_steps=SIMULATION_STEPS)

    # --- Post-Simulation: Log Final States ---
    logger.info("-" * 30)
    logger.info("Final agent states after simulation:")
    if simulation.agents: # Check if agents list is not empty
        # Retrieve the number of steps the simulation actually ran
        actual_steps_run = simulation.current_step
        logger.info(f"(Simulation expected to run for {SIMULATION_STEPS} steps, actually ran for {actual_steps_run} steps)")
        for agent in simulation.agents:
            final_state = agent.get_state()
            # Retrieve the last thought stored in the agent's main state
            last_thought = final_state.get('last_thought', '[No thought recorded]')
            last_broadcast = final_state.get('last_broadcast', '[No broadcast recorded]')
            last_proposed_idea = final_state.get('last_proposed_idea', '[No idea proposed]')
            last_clarification_question = final_state.get('last_clarification_question', '[No clarification asked]')
            agent_goal = final_state.get('goal', '[No goal set]')
            
            # Display basic agent state
            agent_id = agent.get_id()
            logger.info(f"Agent {agent_id} Final State:")
            logger.info(f"  - Goal: {agent_goal}")
            logger.info(f"  - Last Thought: {last_thought}")
            logger.info(f"  - Last Broadcast: {last_broadcast}")
            logger.info(f"  - Last Proposed Idea: {last_proposed_idea}")
            logger.info(f"  - Last Clarification Question: {last_clarification_question}")
            logger.info("-" * 20)
            
            # Check if agent's step counter matches the actual number of steps run
            agent_step_counter = final_state.get('step_counter', -1)
            if agent_step_counter != actual_steps_run:
                logger.warning(f"    Verification Warning: Agent {agent_id} final step_counter ({agent_step_counter}) != actual steps run ({actual_steps_run})")
    else:
        logger.info("  No agents were present in the simulation.")
    logger.info("-" * 30)

    logger.info("+" * 50)
    logger.info(" Culture Simulation Ended")
    logger.info("+" * 50)


if __name__ == "__main__":
    # This block executes when the script is run directly
    main() 