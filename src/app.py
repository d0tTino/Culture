# src/app.py
"""
Main application entry point for the Culture simulation environment.
Initializes and runs the simulation.
"""

import logging
import time
from src.infra import config # Import the configuration module (loads .env)
from src.agents.core.base_agent import Agent # Import the base Agent class
from src.sim.simulation import Simulation # Import the Simulation class

# Configure logging with DEBUG level for more detailed output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Set specific loggers to INFO to reduce noise from external libraries
logging.getLogger('urllib3').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.INFO)
# But keep our own modules at DEBUG level
logging.getLogger('src').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# --- Constants ---
NUM_AGENTS = 3
SIMULATION_STEPS = 7  # Back to 7 steps for final verification

def main():
    """
    Main function to start the Culture application.
    """
    logger.info("+" * 50)
    logger.info(" Starting Culture: An AI Genesis Engine")
    logger.info("+" * 50)
    logger.info("Configuration loaded successfully.") # From config.py import

    # --- Initialize Agents ---
    logger.info(f"Initializing {NUM_AGENTS} agents...")
    agents = []
    for i in range(NUM_AGENTS):
        # Give agents slightly different initial states for variety later
        initial_state = {'name': f'Agent_{i+1}', 'mood': 'neutral', 'step_counter': 0}
        agent = Agent(initial_state=initial_state)
        agents.append(agent)
        time.sleep(0.1)  # Small delay for clearer logging
    logger.info("Agents initialized.")

    # --- Initialize Simulation ---
    logger.info("Initializing simulation environment...")
    simulation = Simulation(agents=agents)
    logger.info("Simulation initialized.")

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
            logger.info(f"  - {agent.get_id()}: State={final_state}")
            logger.info(f"      Last Thought: '{last_thought}'") # Log thought separately for clarity
            logger.info(f"      Broadcast (from final step): '{last_broadcast}'") # Clarify it's the final step's broadcast
            # Verification check placeholder - actual check happens in verification step
            final_count = final_state.get('step_counter', -1) # Get final count, default to -1 if missing
            if final_count != actual_steps_run:
                logger.warning(f"    Verification Warning: Agent {agent.get_id()} final step_counter ({final_count}) != actual steps run ({actual_steps_run})")
    else:
        logger.info("  No agents were present in the simulation.")
    logger.info("-" * 30)

    logger.info("+" * 50)
    logger.info(" Culture Simulation Ended")
    logger.info("+" * 50)


if __name__ == "__main__":
    # This block executes when the script is run directly
    main() 