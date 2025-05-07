# src/agents/core/base_agent.py
"""
Defines the base class for all agents in the Culture simulation.
"""

import uuid
import logging
from typing import Dict, Any, List, Tuple, Deque, Optional, TYPE_CHECKING
from collections import deque
from src.agents.graphs.basic_agent_graph import basic_agent_graph_compiled, AgentTurnState
from .agent_state import AgentState
from src.infra import config

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
        _state (AgentState): Pydantic model holding the agent's internal state.
        graph: The compiled LangGraph for processing agent turns.
    """

    def __init__(self, agent_id: str = None, initial_state: Dict[str, Any] = None):
        """
        Initializes a new agent with a unique ID and default state.

        Args:
            agent_id (str, optional): A unique identifier for the agent.
                If None is provided, a random UUID will be generated.
            initial_state (Dict[str, Any], optional): Initial state for the agent.
                If None is provided, default values will be used.
        """
        # Generate a unique ID if none provided
        self.agent_id = agent_id if agent_id else str(uuid.uuid4())
        
        # Initialize as empty if not provided
        if initial_state is None:
            initial_state = {}
        
        # Get necessary values from initial_state or use defaults
        name = initial_state.get('name', f"Agent-{self.agent_id[:8]}")
        role = initial_state.get('current_role', "Default Contributor")
        steps_in_role = initial_state.get('steps_in_current_role', 0)
        mood = initial_state.get('mood', 'neutral')
        
        # Get values from config or use defaults
        ip = initial_state.get('influence_points', config.INITIAL_INFLUENCE_POINTS if hasattr(config, 'INITIAL_INFLUENCE_POINTS') else 10.0)
        du = initial_state.get('data_units', config.INITIAL_DATA_UNITS if hasattr(config, 'INITIAL_DATA_UNITS') else 20.0)
        max_short_term_memory = config.MAX_SHORT_TERM_MEMORY if hasattr(config, 'MAX_SHORT_TERM_MEMORY') else 10
        short_term_memory_decay_rate = config.SHORT_TERM_MEMORY_DECAY_RATE if hasattr(config, 'SHORT_TERM_MEMORY_DECAY_RATE') else 0.1
        relationship_decay_rate = config.RELATIONSHIP_DECAY_RATE if hasattr(config, 'RELATIONSHIP_DECAY_RATE') else 0.01
        min_relationship_score = config.MIN_RELATIONSHIP_SCORE if hasattr(config, 'MIN_RELATIONSHIP_SCORE') else -1.0
        max_relationship_score = config.MAX_RELATIONSHIP_SCORE if hasattr(config, 'MAX_RELATIONSHIP_SCORE') else 1.0
        mood_decay_rate = config.MOOD_DECAY_RATE if hasattr(config, 'MOOD_DECAY_RATE') else 0.02
        mood_update_rate = config.MOOD_UPDATE_RATE if hasattr(config, 'MOOD_UPDATE_RATE') else 0.2
        ip_cost_per_message = config.IP_COST_PER_MESSAGE if hasattr(config, 'IP_COST_PER_MESSAGE') else 1.0
        du_cost_per_action = config.DU_COST_PER_ACTION if hasattr(config, 'DU_COST_PER_ACTION') else 1.0
        role_change_cooldown = config.ROLE_CHANGE_COOLDOWN if hasattr(config, 'ROLE_CHANGE_COOLDOWN') else 3
        role_change_ip_cost = config.ROLE_CHANGE_IP_COST if hasattr(config, 'ROLE_CHANGE_IP_COST') else 5.0
        
        # Get project id and ensure current_project_id and current_project_affiliation are in sync
        current_project = initial_state.get('current_project_id', None)
        
        # Create the AgentState object
        self._state = AgentState(
            agent_id=self.agent_id,
            name=name,
            role=role,
            steps_in_current_role=steps_in_role,
            mood=mood,
            descriptive_mood=initial_state.get('descriptive_mood', 'neutral'),
            ip=ip,
            du=du,
            # Initialize deques and dictionaries with default empty values (using default_factory in model)
            # Import specific values from initial_state if provided
            relationships=initial_state.get('relationships', {}),
            short_term_memory=deque(initial_state.get('memory_history', []), maxlen=max_short_term_memory),
            goals=initial_state.get('goals', []),
            current_project_id=current_project,
            current_project_affiliation=current_project,  # Keep these two fields in sync
            messages_sent_count=initial_state.get('messages_sent_count', 0),
            messages_received_count=initial_state.get('messages_received_count', 0),
            actions_taken_count=initial_state.get('actions_taken_count', 0),
            available_action_intents=initial_state.get('available_action_intents', []),
            step_counter=initial_state.get('step_counter', 0),
            # Configuration fields
            max_short_term_memory=max_short_term_memory,
            short_term_memory_decay_rate=short_term_memory_decay_rate,
            relationship_decay_rate=relationship_decay_rate,
            min_relationship_score=min_relationship_score,
            max_relationship_score=max_relationship_score,
            mood_decay_rate=mood_decay_rate,
            mood_update_rate=mood_update_rate,
            ip_cost_per_message=ip_cost_per_message,
            du_cost_per_action=du_cost_per_action,
            role_change_cooldown=role_change_cooldown,
            role_change_ip_cost=role_change_ip_cost
        )

        # Initialize the agent's Lang Graph
        self.graph = basic_agent_graph_compiled
        logger.info(f"Agent {self.agent_id} initialized with basic LangGraph, role: {self._state.role}, mood: {self._state.mood}")

    def get_id(self) -> str:
        """Returns the agent's unique ID."""
        return self.agent_id

    @property
    def state(self) -> AgentState:
        """
        Returns the agent's current internal state.
        """
        return self._state

    def update_state(self, updated_state: AgentState):
        """
        Updates the agent's internal state with a new AgentState object.

        Args:
            updated_state (AgentState): The new state for the agent.
        """
        self._state = updated_state
        logger.debug(f"Agent {self.agent_id} state updated")
        
    def add_memory(self, step: int, memory_type: str, content: str):
        """
        Adds a memory to the agent's short-term memory.
        
        Args:
            step (int): The simulation step in which the memory occurred
            memory_type (str): Type of memory (e.g., 'thought', 'broadcast_sent', 'broadcast_received')
            content (str): The content of the memory
        """
        memory_entry = {"step": step, "type": memory_type, "content": content}
        self._state.short_term_memory.append(memory_entry)
        logger.debug(f"Added {memory_type} memory for agent {self.agent_id}")
        
    def update_relationship(self, other_agent_id: str, delta: float):
        """
        Updates the relationship score with another agent.
        
        Args:
            other_agent_id (str): ID of the other agent
            delta (float): Change in relationship score (positive or negative)
        """
        current_score = self._state.relationships.get(other_agent_id, 0.0)
        new_score = max(self._state.min_relationship_score, min(self._state.max_relationship_score, current_score + delta))
        self._state.relationships[other_agent_id] = new_score
        logger.debug(f"Updated relationship for agent {self.agent_id} with {other_agent_id}: {current_score} -> {new_score}")
        
    def update_mood(self, sentiment_score: float):
        """
        Updates the agent's mood based on a sentiment score.
        
        Args:
            sentiment_score (float): The sentiment score to apply to the mood
        """
        from src.agents.graphs.basic_agent_graph import get_mood_level, get_descriptive_mood
        
        # Calculate new mood value using configured decay and update rates
        current_mood_value = float(self._state.mood_history[-1][1]) if self._state.mood_history else 0.0
        new_mood_value = current_mood_value * (1 - self._state.mood_decay_rate) + sentiment_score * self._state.mood_update_rate
        new_mood_value = max(-1.0, min(1.0, new_mood_value))  # Keep within [-1, 1]
        
        # Update mood and descriptive mood
        new_mood = get_mood_level(new_mood_value)
        new_descriptive_mood = get_descriptive_mood(new_mood_value)
        
        # Track if mood changed
        mood_changed = self._state.mood != new_mood
        
        # Update state
        self._state.mood = new_mood
        self._state.mood_history.append((self._state.last_action_step, new_mood))
        
        if mood_changed:
            logger.info(f"Agent {self.agent_id} mood changed to {new_mood} ({new_descriptive_mood})")
        
    def run_turn(self, simulation_step: int, environment_perception: Dict[str, Any] = None, 
                vector_store_manager: Optional[Any] = None, knowledge_board: Optional['KnowledgeBoard'] = None) -> Dict[str, Any]:
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
            Dict[str, Any]: Contains the message content, recipient ID, and action intent
        """
        logger.debug(f"Agent {self.agent_id} starting graph turn for simulation_step: {simulation_step}.")
        
        # Ensure environment_perception is a dictionary even if None is passed
        if environment_perception is None:
            environment_perception = {}
            
        # Log received perception data
        if environment_perception:
            logger.debug(f"  Received environment perception: {environment_perception}")

        # --- Retrieve previous thought ---
        previous_thought = None
        for memory in self._state.short_term_memory:
            if memory["type"] == "thought":
                previous_thought = memory["content"]
                break
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
        
        # Convert the state to dictionary for compatibility with the existing graph
        state_dict = self._state.model_dump()

        # Prepare the input state for this turn's graph execution
        initial_turn_state: AgentTurnState = {
            "agent_id": self.agent_id,
            "current_state": state_dict, # Pass the state as a dictionary for now
            "simulation_step": simulation_step,
            "previous_thought": previous_thought, # Pass previous thought into graph state
            "environment_perception": environment_perception, # Pass environment perception data
            "perceived_messages": perceived_messages, # Pass perceived messages list
            "memory_history_list": list(self._state.short_term_memory), # Pass history list as a list
            "turn_sentiment_score": 0, # Initialize sentiment score for this turn
            "prompt_modifier": "", # Initialize prompt modifier
            "structured_output": None, # Initialize as None for this turn
            "agent_goal": state_dict.get("goals", ["Contribute to the simulation"])[0], # Pass the agent's goal
            "updated_state": {}, # Initialize empty
            "vector_store_manager": vector_store_manager, # Pass the vector store manager
            "rag_summary": "(No memory summary available yet)", # Initialize with default summary
            "knowledge_board_content": knowledge_board_content, # Pass the knowledge board content
            "knowledge_board": knowledge_board, # Pass the knowledge board instance
            "scenario_description": scenario_description, # Pass the simulation scenario
            "current_role": self._state.role, # Pass the agent's current role
            "influence_points": self._state.ip, # Pass the agent's influence points
            "steps_in_current_role": self._state.steps_in_current_role, # Pass the agent's steps in current role
            "data_units": self._state.du, # Pass the agent's data units
            "state": self._state  # Pass the AgentState object directly
        }

        try:
            # Invoke the graph with the initial state for the turn
            final_result_state = self.graph.invoke(initial_turn_state)
            
            # Add debug logging to inspect the graph.invoke result
            logger.debug(f"RUN_TURN_GRAPH_RESULT :: Agent {self.agent_id}: Full result from graph.invoke: {final_result_state}")
            
            # --- Process Graph Output ---
            if final_result_state is None:
                logger.error(f"Agent {self.agent_id} graph execution returned None")
                return {'message_content': None, 'message_recipient_id': None, 'action_intent': 'idle'}
            
            # Log dictionary keys to help debug
            logger.debug(f"RUN_TURN_KEYS :: Agent {self.agent_id}: Available keys in graph result: {list(final_result_state.keys())}")
            
            # Extract the updated state from the AgentState object
            if 'state' in final_result_state:
                updated_state = final_result_state.get('state')
                self._state = updated_state
                logger.debug(f"RUN_TURN_POST_UPDATE :: Agent {self.agent_id}: self._state updated with new AgentState object.")
                
                # Return turn output dict (messages, etc.)
                turn_output = {
                    'message_content': final_result_state.get('message_content'),
                    'message_recipient_id': final_result_state.get('message_recipient_id'),
                    'action_intent': final_result_state.get('action_intent', 'idle')
                }
                
                return turn_output
            else:
                logger.warning(f"RUN_TURN_UPDATE_FAIL :: Agent {self.agent_id}: No 'state' found in graph result. self._state NOT updated.")
                return {'message_content': None, 'message_recipient_id': None, 'action_intent': 'idle'}

        except Exception as e:
            logger.error(f"Error running turn for agent {self.agent_id} at step {simulation_step}: {e}", exc_info=True)
            return {'message_content': None, 'message_recipient_id': None, 'action_intent': 'idle'}

    def __str__(self) -> str:
        """Returns a string representation of the agent."""
        return f"Agent(id={self.agent_id}, role={self._state.role})"

    def __repr__(self) -> str:
        """Returns a detailed string representation for debugging."""
        return f"Agent(agent_id='{self.agent_id}', role='{self._state.role}', ip={self._state.ip}, du={self._state.du})"

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