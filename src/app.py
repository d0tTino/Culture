# src/app.py
"""
Main application entry point for the Culture simulation environment.
Initializes and runs the simulation.
"""

import logging
import time
import os
import collections
import random
import argparse
import asyncio
from src.infra import config # Import the configuration module (loads .env)
from src.agents.core.base_agent import Agent # Import the base Agent class
from src.agents.core.roles import INITIAL_ROLES, ROLE_ANALYZER, ROLE_INNOVATOR # Import the initial roles
from src.agents.graphs.basic_agent_graph import get_mood_level, update_state_node # Import the mood level function and update_state_node function
from src.agents.graphs.basic_agent_graph import AgentActionOutput # Import the AgentActionOutput class
from src.sim.simulation import Simulation # Import the Simulation class
from src.infra.memory.vector_store import ChromaVectorStoreManager # Import the ChromaVectorStoreManager class
from src.interfaces.discord_bot import SimulationDiscordBot # Import the Discord bot class

# Define simulation scenario
SIMULATION_SCENARIO = "The team's objective is to collaboratively design a specification for a decentralized communication protocol. CRITICAL TEST DIRECTIVE: Agent_3 is currently in the 'Analyzer' role but needs to change to 'Innovator'. Agent_3 MUST use the requested_role_change='Innovator' field in the response to test the role change system."

# Configure logging with DEBUG level for more detailed output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Set specific loggers to INFO to reduce noise from external libraries
logging.getLogger('urllib3').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.getLogger('sentence_transformers').setLevel(logging.INFO)  # Reduce noise from embedding model
logging.getLogger('chromadb').setLevel(logging.INFO)  # Reduce noise from chromadb
logging.getLogger('discord').setLevel(logging.INFO)  # Reduce noise from discord.py
# But keep our own modules at DEBUG level
logging.getLogger('src').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# --- Constants ---
NUM_AGENTS = 3
SIMULATION_STEPS = 7  # Default number of steps
CHROMA_DB_PATH = "./chroma_db_store"  # Path for vector DB persistence

async def main():
    """
    Main function to start the Culture application.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Culture.ai simulation')
    parser.add_argument('--steps', type=int, default=SIMULATION_STEPS, help='Number of simulation steps to run')
    parser.add_argument('--discord', action='store_true', help='Enable Discord bot integration')
    args = parser.parse_args()
    
    logger.info("+" * 50)
    logger.info(" Starting Culture: An AI Genesis Engine")
    logger.info("+" * 50)
    logger.info("Configuration loaded successfully.") # From config.py import
    logger.info(f"Running simulation for {args.steps} steps")

    # --- Initialize Discord Bot ---
    discord_bot = None
    bot_task = None
    
    if args.discord or (config.DISCORD_BOT_TOKEN and config.DISCORD_CHANNEL_ID):
        token = config.DISCORD_BOT_TOKEN
        channel_id = config.DISCORD_CHANNEL_ID
        
        if token and channel_id:
            logger.info(f"Initializing Discord bot with channel ID: {channel_id}")
            discord_bot = SimulationDiscordBot(bot_token=token, channel_id=channel_id)
            
            # Start the Discord bot in a separate task
            bot_task = asyncio.create_task(discord_bot.run_bot())
            logger.info("Discord bot started in background task")
            
            # Wait a moment for the bot to connect
            await asyncio.sleep(2)
        else:
            logger.warning("Discord bot requested but token or channel ID not found in config")

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
        # Assign a role to each agent
        agent_name = f'Agent_{i+1}'
        agent_goal = agent_goals[i % len(agent_goals)]
        
        # Assign specific roles to each agent for testing
        if i == 0:
            agent_role = "Facilitator"  # Agent_1 is Facilitator
            logger.info(f"Setting {agent_name} as {agent_role}")
            initial_ip = 20
            steps_in_role = 5
        elif i == 1:
            agent_role = "Innovator"  # Agent_2 is Innovator
            logger.info(f"Setting {agent_name} as {agent_role}")
            initial_ip = 20
            steps_in_role = 5
        elif i == 2:
            agent_role = "Analyzer"  # Agent_3 starts as Analyzer
            logger.info(f"Setting {agent_name} as {agent_role} for role change test")
            initial_ip = 20  # Give Agent_3 extra IP for role change
            steps_in_role = 5  # Set steps in role high enough to allow change
        
        # Give agents slightly different initial states for variety later
        initial_state = {
            'name': agent_name, 
            'mood_value': 0.0,  # Initial numerical mood
            'mood': 'neutral',  
            'mood_level': get_mood_level(0.0),  # Initial descriptive mood
            'descriptive_mood': get_mood_level(0.0),  # Initial descriptive mood
            'step_counter': 0,
            'goal': agent_goal,
            'current_role': agent_role,  # Add current_role here
            'steps_in_current_role': steps_in_role,  # Track steps spent in current role
            'influence_points': initial_ip,  # Set IP based on agent
            'data_units': config.INITIAL_DATA_UNITS,  # Initialize with starting Data Units from config
            'relationships': {},
            'memory_history': collections.deque(maxlen=20),  # Use direct value instead of config.get
            'last_proposed_idea': None,
            'last_clarification_question': None,
            'current_project_affiliation': None  # Initialize with no project affiliation
        }
        agent = Agent(initial_state=initial_state)
        agents.append(agent)
        await asyncio.sleep(0.1)  # Small delay for clearer logging
    logger.info("Agents initialized with unique goals and roles.")

    # --- Initialize Simulation ---
    logger.info("Initializing simulation environment...")
    simulation = Simulation(
        agents=agents, 
        vector_store_manager=vector_store_manager, 
        scenario=SIMULATION_SCENARIO,
        discord_bot=discord_bot
    )
    logger.info("Simulation initialized with vector store manager and Discord bot.")

    # --- Run Simulation ---
    logger.info(f"Running simulation for {args.steps} steps...")
    logger.info(f"Scenario: {SIMULATION_SCENARIO}")
    
    try:
        # Run the simulation for n steps
        simulation.run(num_steps=args.steps)
        
        # Display final agent states
        logger.info("------------------------------")
        logger.info("Final agent states after simulation:")
        logger.info(f"(Simulation expected to run for {args.steps} steps, actually ran for {simulation.current_step} steps)")
        if simulation.agents: # Check if agents list is not empty
            for agent in simulation.agents:
                final_state = agent.get_state()
                # Retrieve the last thought stored in the agent's main state
                last_thought = final_state.get('last_thought', '[No thought recorded]')
                last_message = final_state.get('last_message', '[No message recorded]')
                last_proposed_idea = final_state.get('last_proposed_idea', '[No idea proposed]')
                last_clarification_question = final_state.get('last_clarification_question', '[No clarification asked]')
                agent_goal = final_state.get('goal', '[No goal set]')
                
                # Display basic agent state
                agent_id = agent.get_id()
                logger.info(f"Agent {agent_id} Final State:")
                logger.info(f"  - Goal: {agent_goal}")
                logger.info(f"  - Current Role: {final_state.get('current_role', 'N/A')}")
                logger.info(f"  - Steps in Current Role: {final_state.get('steps_in_current_role', 0)}")
                logger.info(f"  - Mood: {final_state.get('mood', 'N/A')}, Descriptive Mood: {final_state.get('descriptive_mood', 'N/A')}")
                logger.info(f"  - Influence Points: {final_state.get('influence_points', 0)}")
                logger.info(f"  - Data Units: {final_state.get('data_units', 'N/A')}")
                logger.info(f"  - Last Thought: {last_thought}")
                logger.info(f"  - Last Message: {last_message}")
                logger.info(f"  - Last Proposed Idea: {last_proposed_idea}")
                logger.info(f"  - Last Clarification Question: {last_clarification_question}")
                logger.info("-" * 20)
        else:
            logger.info("  No agents were present in the simulation.")
        logger.info("-" * 30)
    
    finally:
        # Clean up and stop the Discord bot if it was started
        if discord_bot:
            logger.info("Stopping Discord bot...")
            await discord_bot.stop_bot()
            
            if bot_task:
                try:
                    # Cancel the bot task if it's still running
                    if not bot_task.done():
                        bot_task.cancel()
                    await bot_task
                except asyncio.CancelledError:
                    logger.info("Discord bot task cancelled")
                except Exception as e:
                    logger.error(f"Error during Discord bot task cleanup: {e}")

    logger.info("+" * 50)
    logger.info(" Culture Simulation Ended")
    logger.info("+" * 50)

def test_role_change():
    """
    A simple test function for directly testing the role change functionality.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting role change test...")
    
    # Initialize an agent with Analyzer role
    from src.agents.core.base_agent import Agent
    from src.agents.core.roles import ROLE_ANALYZER, ROLE_INNOVATOR
    from src.infra.memory.vector_store import ChromaVectorStoreManager
    
    # Create a vector store manager (needed for agent initialization)
    vector_store = ChromaVectorStoreManager(persist_directory="./chroma_db_test")
    
    # Initial state with Analyzer role and sufficient steps/IP for role change
    initial_state = {
        'name': 'TestAgent', 
        'mood_value': 0.0,
        'mood': 'neutral',  
        'descriptive_mood': 'neutral',
        'step_counter': 3,
        'goal': 'Test the role change system',
        'current_role': ROLE_ANALYZER,  # Start as Analyzer
        'steps_in_current_role': 3,    # Enough steps to change role
        'influence_points': 10,        # Enough IP to change role
        'relationships': {},
        'memory_history': collections.deque(maxlen=20),
        'last_proposed_idea': None,
        'last_clarification_question': None
    }
    
    # Create the test agent
    test_agent = Agent("test-agent-id", initial_state=initial_state)
    
    # Create a simple test simulation context
    test_context = {
        'agent_id': 'test-agent-id',
        'simulation_step': 1,
        'environment_perception': {
            'other_agents_state': [],
            'perceived_messages': [],
            'knowledge_board_content': [],
            'scenario_description': "Test scenario for role change"
        },
        'structured_output': AgentActionOutput(
            thought="I think I would be more effective as an Innovator.", 
            message_content="I would like to change my role to Innovator.",
            message_recipient_id=None,
            action_intent='continue_collaboration',
            requested_role_change=ROLE_INNOVATOR  # Request change to Innovator
        ),
        'vector_store_manager': vector_store
    }
    
    # Log the agent state before role change
    logger.info(f"Agent role before change: {test_agent.state.get('current_role')}")
    logger.info(f"Agent IP before change: {test_agent.state.get('influence_points')}")
    
    # Apply the test context to the agent's state
    test_context['current_state'] = test_agent.state
    
    # Call the update_state_node function to process the role change
    updated_context = update_state_node(test_context)
    
    # Update the agent's state with the result
    test_agent.state.update(updated_context.get('updated_state', {}))
    
    # Log the agent state after role change
    logger.info(f"Agent role after change: {test_agent.state.get('current_role')}")
    logger.info(f"Agent IP after change: {test_agent.state.get('influence_points')}")
    logger.info(f"Agent steps in current role: {test_agent.state.get('steps_in_current_role')}")
    
    logger.info("Role change test completed")

if __name__ == "__main__":
    # This block executes when the script is run directly
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-role-change":
        test_role_change()
    else:
        # Use asyncio.run to run the async main function
        asyncio.run(main()) 