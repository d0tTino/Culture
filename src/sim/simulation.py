"""
Defines the Simulation class responsible for managing agents and the simulation loop.
"""

import logging
import time
from typing import List, Dict, Any, TYPE_CHECKING, Optional

# Use TYPE_CHECKING to avoid circular import issues if Agent needs Simulation later
if TYPE_CHECKING:
    from src.agents.core.base_agent import Agent
    from src.infra.memory.vector_store import ChromaVectorStoreManager

from src.sim.knowledge_board import KnowledgeBoard

# Configure the logger for this module
logger = logging.getLogger(__name__)

class Simulation:
    """
    Manages the simulation environment, agents, and time steps.
    """
    def __init__(self, agents: List['Agent'], vector_store_manager: Optional['ChromaVectorStoreManager'] = None, scenario: str = ""):
        """
        Initializes the Simulation instance.

        Args:
            agents (List[Agent]): A list of Agent instances participating
                                  in the simulation.
            vector_store_manager (Optional[ChromaVectorStoreManager]): Manager for 
                                  vector-based agent memory storage and retrieval.
            scenario (str): Description of the simulation scenario that provides context for agent interactions.
        """
        self.agents: List['Agent'] = agents
        self.current_step: int = 0
        # Add other simulation-wide state if needed (e.g., environment properties)
        # self.environment_state = {}
        
        # --- Store the simulation scenario ---
        self.scenario = scenario
        if scenario:
            logger.info(f"Simulation initialized with scenario: {scenario}")
        else:
            logger.warning("Simulation initialized without a scenario description.")
        
        # --- NEW: Initialize Knowledge Board ---
        self.knowledge_board = KnowledgeBoard()
        logger.info("Simulation initialized with Knowledge Board.")
        
        # --- Store the vector store manager ---
        self.vector_store_manager = vector_store_manager
        if vector_store_manager:
            logger.info("Simulation initialized with vector store manager for memory persistence.")
        else:
            logger.warning("Simulation initialized without vector store manager. Memory will not be persisted.")
        
        # --- Store broadcasts from the previous step ---
        self.last_step_broadcasts: List[Dict[str, Any]] = []
        logger.info("Initialized storage for last step's broadcasts.")
        # --- End NEW ---

        if not self.agents:
            logger.warning("Simulation initialized with zero agents.")
        else:
            logger.info(f"Simulation initialized with {len(self.agents)} agents:")
            for agent in self.agents:
                logger.info(f"  - {agent.get_id()}")


    def run_step(self):
        """
        Executes a single step of the simulation.
        """
        self.current_step += 1
        logger.info(f"--- Simulation Step {self.current_step} ---")

        # --- Get broadcasts from the PREVIOUS step for current perception ---
        # Make a copy to avoid modification issues if needed later
        broadcasts_to_perceive = self.last_step_broadcasts.copy()
        if broadcasts_to_perceive:
            logger.debug(f"Step {self.current_step}: Agents will perceive {len(broadcasts_to_perceive)} broadcasts from Step {self.current_step - 1}.")
        # --- End Get Broadcasts ---
        
        # --- Get current Knowledge Board state ---
        board_state = self.knowledge_board.get_state()
        logger.debug(f"Step {self.current_step}: Knowledge Board state has {len(board_state)} entries.")
        # --- End Get Board State ---

        if not self.agents:
            logger.warning("No agents to process in simulation step.")
            self.last_step_broadcasts = [] # Clear broadcasts if no agents
            return

        # List to collect broadcasts generated in THIS step
        current_step_generated_broadcasts = []

        # Simple round-robin activation for now
        for agent in self.agents:
            # --- Gather Environment Perception for the current agent ---
            # Get richer state info (id, name, mood) of other agents
            other_agents_state = []
            for other in self.agents:
                if other.get_id() != agent.get_id(): # Exclude self
                    other_state = other.get_state()
                    other_agents_state.append({
                        "id": other.get_id(),
                        "name": other_state.get('name', other.get_id()), # Use name, fallback to ID
                        "mood": other_state.get('mood', 'unknown') # Get current mood
                    })

            # Prepare the perception dictionary
            perception_data = {
                # Add other perception data here if needed in the future
                "other_agents_state": other_agents_state, # List of dicts with richer info
                # --- ADD perceived broadcasts ---
                "broadcasts": broadcasts_to_perceive, # Pass the list collected at the start
                # --- Add Knowledge Board content ---
                "knowledge_board_content": board_state, # Pass the current board state
                # --- Add simulation scenario ---
                "scenario_description": self.scenario # Pass the simulation scenario
                # --- End ADD ---
            }
            # --- End Gather Environment Perception ---

            logger.info(f"Activating agent: {agent.get_id()} for step {self.current_step}")
            logger.debug(f"  Perception for {agent.get_id()}: {perception_data}") # Log the specific perception
            
            # --- Invoke Agent's Graph Turn ---
            try:
                # Pass the perception dictionary and vector store manager to the agent's turn method
                success = agent.run_turn(
                    simulation_step=self.current_step,
                    environment_perception=perception_data, # Pass the gathered perception
                    vector_store_manager=self.vector_store_manager, # Pass the vector store manager
                    knowledge_board=self.knowledge_board # Pass the knowledge board instance
                )
                if not success:
                    logger.warning(f"Agent {agent.get_id()} turn execution reported failure for step {self.current_step}.")
                else:
                    # --- Collect broadcast generated by THIS agent in THIS turn ---
                    last_broadcast = agent.get_state().get('last_broadcast', None)
                    if last_broadcast:
                        current_step_generated_broadcasts.append({
                            "sender_id": agent.get_id(),
                            "message": last_broadcast
                        })
                    # --- End Collect Broadcast ---
            except Exception as e:
                # Catch potential errors bubbling up from run_turn if not handled internally
                logger.error(f"Unhandled exception during agent {agent.get_id()} turn activation on step {self.current_step}: {e}", exc_info=True)
            # --- End Agent's Graph Turn ---

        # --- Store broadcasts collected THIS step for the NEXT step ---
        self.last_step_broadcasts = current_step_generated_broadcasts # Overwrite with new broadcasts
        if self.last_step_broadcasts:
            logger.info(f"--- Broadcasts generated and stored in Step {self.current_step} (for perception in Step {self.current_step + 1}) ---")
            for broadcast in self.last_step_broadcasts:
                logger.info(f"  From {broadcast['sender_id']}: {broadcast['message']}")
        else:
            logger.info(f"--- No broadcasts generated in Step {self.current_step} ---")
        # --- End Store Broadcasts ---

        logger.info(f"--- End Simulation Step {self.current_step} ---")


    def run(self, num_steps: int):
        """
        Runs the simulation for a specified number of steps.

        Args:
            num_steps (int): The total number of steps to simulate.
        """
        logger.info(f"Starting simulation run for {num_steps} steps.")
        if num_steps <= 0:
            logger.warning("Number of steps must be positive.")
            return

        for step in range(num_steps):
            self.run_step()
            time.sleep(0.2)  # Small delay between steps for clearer logging
            # Add condition checks here to stop early if needed
            # (e.g., all agents inactive, goal achieved)

        logger.info(f"Simulation run finished after {self.current_step} steps.")


    # --- Optional helper methods for future use ---
    # def get_environment_view(self, agent: 'Agent'):
    #     """Provides the agent with its perception of the environment."""
    #     # To be implemented: return relevant state based on agent position, sensors etc.
    #     return {"global_time": self.current_step}

    # def execute_action(self, agent: 'Agent', action: str):
    #     """Handles the execution of an agent's chosen action."""
    #     # To be implemented: update agent state, environment state based on action
    #     logger.info(f"Agent {agent.get_id()} performs action: {action}")

    # def update_environment(self):
    #      """Updates the global environment state after agent actions."""
    #      # To be implemented
    #      pass 