# src/agents/core/base_agent.py
"""
Defines the base class for all agents in the Culture simulation.
"""

import uuid
import logging
from typing import Dict, Any, List, Tuple, Deque, Optional, TYPE_CHECKING
from collections import deque
from src.agents.graphs.basic_agent_graph import basic_agent_graph_compiled, AgentTurnState

# Use TYPE_CHECKING to avoid circular import issues
if TYPE_CHECKING:
    from src.infra.memory.vector_store import ChromaVectorStoreManager
    from src.sim.knowledge_board import KnowledgeBoard

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

        # --- Initialize Relationships Structure ---
        if 'relationships' not in self.state:
            self.state['relationships'] = {}  # Maps agent_id -> score (e.g., float)
        # Ensure memory_history is also initialized correctly
        if 'memory_history' not in self.state:
            self.state['memory_history'] = []  # Initialize if missing from initial_state
        # --- End Initialize Relationships ---

        # --- Initialize Memory History ---
        # Store recent events: (step, type, content)
        # type can be 'thought', 'broadcast_sent', 'broadcast_perceived'
        self.memory_history: Deque[Tuple[int, str, str]] = deque(maxlen=5) # Store last 5 events
        # Store the history directly in the agent's main state dictionary
        self.state['memory_history'] = list(self.memory_history) # Store as list in state for now
        # --- End Initialize ---

        # Initialize the agent's Lang Graph
        self.graph = basic_agent_graph_compiled
        logger.info(f"Agent {self.agent_id} initialized with basic LangGraph and memory (maxlen=5), mood: {self.state['mood']}, including relationships dict.")

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
        
    def run_turn(self, simulation_step: int, environment_perception: Dict[str, Any] = None, 
                vector_store_manager: Optional[Any] = None, knowledge_board: Optional['KnowledgeBoard'] = None) -> bool:
        """
        Executes the agent's internal graph for one turn, passing the previous thought.

        Args:
            simulation_step (int): The current step number from the simulation.
            environment_perception (Dict[str, Any], optional): Perception data from the environment.
                Defaults to an empty dict if not provided.
            vector_store_manager (Optional[Any], optional): Manager for vector-based memory
                storage and retrieval. Used to persist memory events.
            knowledge_board (Optional[KnowledgeBoard], optional): Knowledge board instance
                that agents can read from and write to.

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
        
        # --- Retrieve agent goal ---
        agent_goal = self.state.get('goal', "Contribute to the simulation as effectively as possible.")
        logger.debug(f"  Retrieved agent goal: '{agent_goal}'")
        # --- End Retrieve ---
        
        # --- Extract Knowledge Board Content ---
        knowledge_board_content = environment_perception.get('knowledge_board_content', [])
        logger.debug(f"  Retrieved knowledge board content (entries: {len(knowledge_board_content)})")
        # --- End Extract Knowledge Board ---

        # --- Extract Simulation Scenario ---
        scenario_description = environment_perception.get('scenario_description', "")
        logger.debug(f"  Retrieved simulation scenario: '{scenario_description}'")
        # --- End Extract Simulation Scenario ---

        # --- Extract Perceived Messages ---
        perceived_messages = environment_perception.get('perceived_messages', [])
        logger.debug(f"  Retrieved {len(perceived_messages)} perceived messages")
        # --- End Extract Perceived Messages ---

        # Prepare the input state for this turn's graph execution
        initial_turn_state: AgentTurnState = {
            "agent_id": self.agent_id,
            "current_state": self.state.copy(), # Pass a copy of current state
            "simulation_step": simulation_step,
            "previous_thought": previous_thought, # Pass previous thought into graph state
            "environment_perception": environment_perception, # Pass environment perception data
            "perceived_messages": perceived_messages, # Pass perceived messages list
            "memory_history_list": current_memory_list, # Pass history list
            "turn_sentiment_score": 0, # Initialize sentiment score for this turn
            "prompt_modifier": "", # Initialize prompt modifier
            "structured_output": None, # Initialize as None for this turn
            "agent_goal": agent_goal, # Pass the agent's goal
            "updated_state": {}, # Initialize empty
            "vector_store_manager": vector_store_manager, # Pass the vector store manager
            "rag_summary": "(No memory summary available yet)", # Initialize with default summary
            "knowledge_board_content": knowledge_board_content, # Pass the knowledge board content
            "knowledge_board": knowledge_board, # Pass the knowledge board instance
            "scenario_description": scenario_description # Pass the simulation scenario
        }

        try:
            # Invoke the graph with the initial state for the turn
            final_result_state = self.graph.invoke(initial_turn_state)
            
            # Add debug logging to inspect the graph.invoke result
            logger.debug(f"RUN_TURN_GRAPH_RESULT :: Agent {self.agent_id}: Full result from graph.invoke: {final_result_state}")
            
            # --- Process Graph Output ---
            # Extract the final agent state from the graph result
            # Ensure we have a valid final state dictionary
            if final_result_state is None:
                logger.error(f"Agent {self.agent_id} graph execution returned None")
                return {'message_content': None, 'message_recipient_id': None, 'action_intent': 'idle'}
            
            # Log dictionary keys to help debug
            logger.debug(f"RUN_TURN_KEYS :: Agent {self.agent_id}: Available keys in graph result: {list(final_result_state.keys())}")    
            
            # Try the updated_agent_state key first
            updated_state_dict = final_result_state.get('updated_agent_state')
            
            # If not found, try the updated_state key
            if not updated_state_dict and 'updated_state' in final_result_state:
                updated_state_dict = final_result_state.get('updated_state')
                logger.debug(f"RUN_TURN_FALLBACK :: Agent {self.agent_id}: Using 'updated_state' key instead of 'updated_agent_state'")
            
            logger.debug(f"RUN_TURN_PRE_UPDATE :: Agent {self.agent_id}: Attempting to update self.state with: {updated_state_dict}")
            
            if updated_state_dict:
                # Update the agent's current state
                self.state.update(updated_state_dict)
                logger.debug(f"RUN_TURN_POST_UPDATE :: Agent {self.agent_id}: self.state updated.")
                
                # Return turn output dict (messages, etc.)
                turn_output = {
                    'message_content': final_result_state.get('message_content'),
                    'message_recipient_id': final_result_state.get('message_recipient_id'),
                    'action_intent': final_result_state.get('action_intent', 'idle')
                }
                
                return turn_output
            else:
                logger.warning(f"RUN_TURN_UPDATE_FAIL :: Agent {self.agent_id}: No updated state found in graph result (checked both 'updated_agent_state' and 'updated_state' keys). self.state NOT updated.")
                return {'message_content': None, 'message_recipient_id': None, 'action_intent': 'idle'}

        except Exception as e:
            logger.error(f"Error running turn for agent {self.agent_id} at step {simulation_step}: {e}", exc_info=True)
            return {'message_content': None, 'message_recipient_id': None, 'action_intent': 'idle'}

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