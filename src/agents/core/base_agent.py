# src/agents/core/base_agent.py
"""
Defines the base class for all agents in the Culture simulation.
"""

import uuid
import logging
from typing import Dict, Any, List, Tuple, Deque
from collections import deque
from src.agents.graphs.basic_agent_graph import basic_agent_graph_compiled, AgentTurnState

logger = logging.getLogger(__name__)

class Agent:
    """
    Represents a basic agent in the simulation environment.

    Attributes:
        agent_id (str): A unique identifier for the agent.
        state (Dict[str, Any]): A dictionary holding the agent's internal state
                               (e.g., personality traits, resources, status).
        graph: The compiled LangGraph for processing agent turns.
        # Add other core attributes common to all agents if needed.
    """

    def __init__(self, agent_id: str = None, initial_state: Dict[str, Any] = None):
        """
        Initializes a new agent with a unique ID and default state.

        Args:
            agent_id (str, optional): A unique identifier for the agent.
                If None is provided, a random UUID will be generated.
            initial_state (Dict[str, Any], optional): Initial state for the agent.
                If None is provided, an empty dictionary will be used.
        """
        # Generate a unique ID if none provided
        self.agent_id = agent_id if agent_id else str(uuid.uuid4())
        
        # Initialize the state or use provided initial state
        if initial_state is None:
            initial_state = {}
        
        # Ensure basic state properties are initialized
        if 'step_counter' not in initial_state:
            initial_state['step_counter'] = 0
            
        # Set initial mood if not provided
        if 'mood' not in initial_state:
            initial_state['mood'] = 'neutral'
            
        # Copy the initial state
        self.state: Dict[str, Any] = initial_state.copy()

        # --- Initialize Memory History ---
        # Store recent events: (step, type, content)
        # type can be 'thought', 'broadcast_sent', 'broadcast_perceived'
        self.memory_history: Deque[Tuple[int, str, str]] = deque(maxlen=5) # Store last 5 events
        # Store the history directly in the agent's main state dictionary
        self.state['memory_history'] = list(self.memory_history) # Store as list in state for now
        # --- End Initialize ---

        # Initialize the agent's Lang Graph
        self.graph = basic_agent_graph_compiled
        logger.info(f"Agent {self.agent_id} initialized with basic LangGraph and memory (maxlen=5), mood: {self.state['mood']}")

    def get_id(self) -> str:
        """Returns the agent's unique ID."""
        return self.agent_id

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a copy of the agent's current internal state.
        Returning a copy prevents external modification of the internal state dict.
        """
        return self.state.copy()

    def update_state(self, key: str, value: Any):
        """
        Updates a specific key in the agent's internal state.

        Args:
            key (str): The name of the state variable to update.
            value (Any): The new value for the state variable.
        """
        self.state[key] = value
        # Potentially add logging here for state changes if needed for debugging
        # logger.debug(f"Agent {self.agent_id} state updated: {key} = {value}")
        
    def run_turn(self, simulation_step: int, environment_perception: Dict[str, Any] = None) -> bool:
        """
        Executes the agent's internal graph for one turn, passing the previous thought.

        Args:
            simulation_step (int): The current step number from the simulation.
            environment_perception (Dict[str, Any], optional): Perception data from the environment.
                Defaults to an empty dict if not provided.

        Returns:
            bool: True if the turn executed successfully, False otherwise.
        """
        logger.debug(f"Agent {self.agent_id} starting graph turn for simulation_step: {simulation_step}.")
        
        # Ensure environment_perception is a dictionary even if None is passed
        if environment_perception is None:
            environment_perception = {}
            
        # Log received perception data
        if environment_perception:
            logger.debug(f"  Received environment perception: {environment_perception}")

        # --- Retrieve previous thought from agent's main state ---
        previous_thought = self.state.get('last_thought', None)
        if previous_thought:
            logger.debug(f"  Retrieved last_thought from state: '{previous_thought}'")
        # --- End Retrieve ---
        
        # --- Retrieve Memory History ---
        # Get the history list from the agent's main state
        current_memory_list = self.state.get('memory_history', [])
        logger.debug(f"  Retrieved memory history (length {len(current_memory_list)}): {current_memory_list}")
        # --- End Retrieve ---

        # Prepare the input state for this turn's graph execution
        initial_turn_state: AgentTurnState = {
            "agent_id": self.agent_id,
            "current_state": self.state.copy(), # Pass a copy of current state
            "simulation_step": simulation_step,
            "previous_thought": previous_thought, # Pass previous thought into graph state
            "environment_perception": environment_perception, # Pass environment perception data
            "memory_history_list": current_memory_list, # Pass history list
            "turn_sentiment_score": 0, # Initialize sentiment score for this turn
            "llm_thought": None, # Initialize as None for this turn
            "broadcast_message": None, # Initialize broadcast for this turn
            "updated_state": {} # Initialize empty
        }

        try:
            # Invoke the graph with the initial state for the turn
            final_turn_state = self.graph.invoke(initial_turn_state)

            # Update the agent's main state with the result from the graph
            # (This includes the 'last_thought' generated THIS turn, ready for the NEXT turn)
            if "updated_state" in final_turn_state and final_turn_state["updated_state"]:
                self.state = final_turn_state["updated_state"]
                logger.debug(f"Agent {self.agent_id} main state updated from graph result.")
                return True
            else:
                logger.warning(f"Agent {self.agent_id} graph turn finished but 'updated_state' was missing or empty.")
                return False

        except Exception as e:
            logger.error(f"Exception during agent {self.agent_id} graph turn: {e}", exc_info=True)
            return False

    def __str__(self) -> str:
        """Returns a string representation of the agent."""
        return f"Agent(id={self.agent_id}, state={self.state})"

    def __repr__(self) -> str:
        """Returns a detailed string representation for debugging."""
        return f"Agent(agent_id='{self.agent_id}', initial_state={self.state})"

    # --- Placeholder Methods for Future Functionality ---

    # def perceive(self, environment_state: Dict[str, Any]):
    #     """Placeholder for how the agent perceives the environment."""
    #     logger.warning(f"Agent {self.agent_id}: perceive() method not implemented.")
    #     pass

    # def decide(self) -> str:
    #     """Placeholder for the agent's decision-making process."""
    #     logger.warning(f"Agent {self.agent_id}: decide() method not implemented.")
    #     return "idle" # Default action

    # def act(self, action: str, environment: Any):
    #     """Placeholder for executing the chosen action."""
    #     logger.warning(f"Agent {self.agent_id}: act() method not implemented for action '{action}'.")
    #     pass 