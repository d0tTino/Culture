# src/agents/core/base_agent.py
"""
Defines the base class for all agents in the Culture simulation.
"""

import asyncio
import copy
import logging
import uuid
from collections.abc import Awaitable
from math import sqrt

# LangGraph imports
# from langgraph.graph import StateGraph, END # No longer needed here
# Import node functions and router from basic_agent_graph
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

from pydantic import BaseModel
from typing_extensions import Self

if TYPE_CHECKING:
    from src.agents.memory.vector_store import ChromaVectorStoreManager
else:  # pragma: no cover - optional dependency
    try:
        from src.agents.memory.vector_store import ChromaVectorStoreManager
    except Exception:
        ChromaVectorStoreManager = None
from src.agents.memory.weaviate_vector_store_manager import WeaviateVectorStoreManager
from src.infra import config
from src.infra.async_dspy_manager import AsyncDSPyManager
from src.infra.config import get_config
from src.infra.llm_client import get_ollama_client

from .embedding_utils import compute_embedding
from .roles import ensure_profile

if TYPE_CHECKING:
    from src.interfaces.dashboard_backend import AgentMessage as DashboardAgentMessage
    from src.interfaces.dashboard_backend import message_sse_queue
else:  # pragma: no cover - optional runtime dependency
    try:
        from src.interfaces.dashboard_backend import AgentMessage as DashboardAgentMessage
        from src.interfaces.dashboard_backend import (
            message_sse_queue,
        )
    except Exception:

        class DashboardAgentMessage(BaseModel):  # minimal stub for tests
            agent_id: str
            content: str
            step: int
            recipient_id: str | None = None
            action_intent: str | None = None
            timestamp: float | None = None
            extra: dict[str, Any] | None = None

        message_sse_queue: asyncio.Queue[DashboardAgentMessage] = asyncio.Queue()

from src.shared.memory_store import MemoryStore
from src.shared.typing import SimulationMessage

from .agent_controller import AgentController

# Import graph types from the new file
from .agent_graph_types import AgentActionOutput, AgentTurnState  # RESTORE THIS IMPORT

# Import AgentState and AgentActionIntent first (no circular dependency)
from .agent_state import _PYDANTIC_V2, AgentActionIntent, AgentState

# Use TYPE_CHECKING to avoid circular import issues
if TYPE_CHECKING:
    from src.agents.memory.vector_store import ChromaVectorStoreManager
    from src.sim.knowledge_board import KnowledgeBoard

# --- REMOVE MOVED Graph State Definitions ---
# class AgentActionOutput(BaseModel):
#     ...
# class AgentTurnState(TypedDict):
#     ...

# Corrected imports for DSPy program getters
from src.agents.dspy_programs.action_intent_selector import get_optimized_action_selector
from src.agents.dspy_programs.relationship_updater import get_relationship_updater
from src.agents.dspy_programs.role_thought_generator import get_role_thought_generator

AgentMessage = DashboardAgentMessage

logger = logging.getLogger(__name__)


class Agent:
    """
    Represents a basic agent in the simulation environment.

    Attributes:
        agent_id (str): A unique identifier for the agent.
        _state (AgentState): Pydantic model holding the agent's internal state.
        graph: The compiled LangGraph for processing agent turns.
    """

    def __init__(
        self: Self,
        agent_id: str | None = None,
        initial_state: dict[str, Any] | None = None,
        name: str | None = None,
        vector_store_manager: Optional[MemoryStore] = None,
        async_dspy_manager: Optional[AsyncDSPyManager] = None,
    ):
        """
        Initializes a new agent with a unique ID and default state.

        Args:
            agent_id (str, optional): A unique identifier for the agent.
                If None is provided, a random UUID will be generated.
            initial_state (Dict[str, Any], optional): Initial state for the agent.
                If None is provided, default values will be used.
            name (str, optional): A name for the agent. If None is provided,
                a default name based on agent_id will be used.
            vector_store_manager (Optional[MemoryStore], optional): Manager for vector-based memory
                storage and retrieval. Used to persist memory events.
            async_dspy_manager (Optional[AsyncDSPyManager], optional): Manager for DSPy program execution.
        """
        self.agent_id = agent_id if agent_id else str(uuid.uuid4())

        # Explicitly declare vector_store_manager for Mypy
        self.vector_store_manager: MemoryStore | None = vector_store_manager

        # Initialize as empty if not provided
        if initial_state is None:
            initial_state = {}

        # Get necessary values from initial_state or use defaults
        name = name or initial_state.get("name", f"Agent-{self.agent_id[:8]}")
        role_value = initial_state.get("current_role", "Default Contributor")
        role_name: str
        role_embedding = initial_state.get("role_embedding")
        if isinstance(role_value, dict):
            role_name = str(role_value.get("name", "Default Contributor"))
            role_embedding = role_embedding or role_value.get("embedding")
        elif hasattr(role_value, "name"):
            role_name = str(getattr(role_value, "name"))
            role_embedding = role_embedding or getattr(role_value, "embedding", None)
        else:
            role_name = str(role_value)
        if role_embedding is None:
            role_embedding = compute_embedding(role_name)
        steps_in_role = initial_state.get("steps_in_current_role", 0)
        mood = initial_state.get("mood", "neutral")
        agent_goal = initial_state.get(
            "agent_goal", "Contribute to the simulation as effectively as possible."
        )
        reputation = initial_state.get("reputation", {})

        # LLM Client Initialization - MOVED EARLIER
        self.llm_client = get_ollama_client()

        # Get project id and ensure current_project_id and current_project_affiliation are in sync

        # Check for goal in initial_state - handle the case where it's a string instead of dict
        # list
        goals = []
        if "goals" in initial_state:
            goals = initial_state["goals"]
        elif "goal" in initial_state:
            # Convert single goal string to the proper dictionary format expected by AgentState
            goal_text = initial_state["goal"]
            goals = [{"description": goal_text, "priority": "high"}]

        # This overrides any default_factory behavior in AgentState for these specific fields.
        constructor_args_from_config = {
            "positive_relationship_learning_rate": get_config(
                "POSITIVE_RELATIONSHIP_LEARNING_RATE"
            ),
            "negative_relationship_learning_rate": get_config(
                "NEGATIVE_RELATIONSHIP_LEARNING_RATE"
            ),
            "neutral_relationship_learning_rate": get_config("NEUTRAL_RELATIONSHIP_LEARNING_RATE"),
            "targeted_message_multiplier": get_config("TARGETED_MESSAGE_MULTIPLIER"),
            "relationship_decay_rate": get_config("RELATIONSHIP_DECAY_FACTOR"),
            "mood_decay_rate": get_config("MOOD_DECAY_FACTOR"),
            "mood_update_rate": get_config("MOOD_UPDATE_RATE"),
            "max_short_term_memory": get_config("MAX_SHORT_TERM_MEMORY"),
            "short_term_memory_decay_rate": get_config("SHORT_TERM_MEMORY_DECAY_RATE"),
            "min_relationship_score": get_config("MIN_RELATIONSHIP_SCORE"),
            "max_relationship_score": get_config("MAX_RELATIONSHIP_SCORE"),
            "ip_cost_per_message": get_config("IP_COST_SEND_DIRECT_MESSAGE"),
            "du_cost_per_action": get_config("DU_COST_PER_ACTION"),
            "role_change_cooldown": get_config("ROLE_CHANGE_COOLDOWN"),
            "role_change_ip_cost": get_config("ROLE_CHANGE_IP_COST"),
            "ip": get_config("INITIAL_INFLUENCE_POINTS"),  # Use 'ip' as AgentState field name
            "du": get_config("INITIAL_DATA_UNITS"),  # Use 'du' as AgentState field name
        }

        # Start with config-derived values
        agent_state_kwargs = constructor_args_from_config.copy()

        # Layer explicit initial_state values over them.
        # Pydantic will ignore extra keys in initial_state if not defined in AgentState.
        # initial_state can override values from constructor_args_from_config.
        if initial_state:  # Ensure initial_state is not None
            # Handle specific remapping from initial_state keys to AgentState field names
            if "influence_points" in initial_state:
                agent_state_kwargs["ip"] = initial_state.pop("influence_points")  # Use and remove
            if "data_units" in initial_state:
                agent_state_kwargs["du"] = initial_state.pop("data_units")  # Use and remove

            # Update with the rest of initial_state; Pydantic handles what it needs.
            agent_state_kwargs.update(initial_state)

        # Add/override mandatory/always-set fields AFTER initial_state processing
        agent_state_kwargs["agent_id"] = self.agent_id
        agent_state_kwargs["name"] = name  # name is derived above
        agent_state_kwargs["current_role"] = ensure_profile(role_value)
        if role_embedding is not None:
            agent_state_kwargs["current_role"].embedding = role_embedding
        agent_state_kwargs["role_embedding"] = list(agent_state_kwargs["current_role"].embedding)
        agent_state_kwargs["reputation_score"] = agent_state_kwargs["current_role"].reputation
        agent_state_kwargs["steps_in_current_role"] = steps_in_role  # derived above
        agent_state_kwargs["mood"] = mood  # derived above
        agent_state_kwargs["agent_goal"] = agent_goal  # derived above
        agent_state_kwargs["reputation"] = reputation
        agent_state_kwargs["llm_client"] = self.llm_client  # NOW self.llm_client is initialized

        # Handle 'goals' separately if it was a single string in initial_state (already handled when deriving 'goals' var)
        # The 'goals' variable prepared earlier (either from initial_state['goals'] or converted from initial_state['goal'])
        # should be used here if it's not already covered by agent_state_kwargs.update(initial_state)
        # If initial_state had 'goals' or 'goal', it's already in agent_state_kwargs.
        # If not, AgentState's default for 'goals' (e.g., empty list) should apply.
        # We can ensure 'goals' is set if it was prepared:
        if goals:  # 'goals' variable from earlier logic
            agent_state_kwargs["goals"] = goals
        elif "goals" not in agent_state_kwargs:  # Ensure it has a default if not provided at all
            agent_state_kwargs["goals"] = []

        # Remove any None values for keys that AgentState expects a default for if not provided,
        # to let Pydantic's field defaults take effect.
        # This is only necessary if get_config() could return None for keys in constructor_args_from_config
        # AND AgentState fields for those keys are Optional without a default_factory or direct default.
        # Given current get_config logic (falls back to DEFAULT_CONFIG), Nones should be rare for these.
        # agent_state_kwargs = {k: v for k, v in agent_state_kwargs.items() if v is not None} # Risky if None is valid for some fields

        self._state = AgentState(**agent_state_kwargs)

        # Record initial resources in the ledger
        try:
            from src.infra.ledger import ledger

            ledger.log_change(
                self.agent_id,
                float(self._state.ip),
                float(self._state.du),
                "init",
            )
        except Exception:  # pragma: no cover - ledger failures shouldn't break init
            logger.debug("Ledger logging failed during agent init", exc_info=True)

        # Initialize LangGraph graph by calling the compiler function.
        # Some tests monkeypatch bag.compile_agent_graph without undoing it,
        # so capture and restore the original function to avoid side effects.
        import importlib
        import sys

        bag_mod = importlib.import_module("src.agents.graphs.basic_agent_graph")
        self.graph = bag_mod.compile_agent_graph()
        # Restore the original module in case tests monkeypatched it
        if getattr(bag_mod, "__file__", None):
            importlib.reload(bag_mod)
        else:
            sys.modules.pop("src.agents.graphs.basic_agent_graph", None)
            importlib.import_module("src.agents.graphs.basic_agent_graph")

        # Vector Store Manager Initialization
        if vector_store_manager:
            self.vector_store_manager = vector_store_manager
        else:
            backend = (
                config.VECTOR_STORE_BACKEND
                if hasattr(config, "VECTOR_STORE_BACKEND")
                else "chroma"
            )
            if backend == "weaviate":
                self.vector_store_manager = WeaviateVectorStoreManager(
                    url=getattr(config, "WEAVIATE_URL", "http://localhost:8080"),
                    collection_name="AgentMemory",
                    embedding_function=None,
                    # Should be set to the SentenceTransformer instance if needed
                )
            else:
                if ChromaVectorStoreManager is None:
                    raise ImportError("chromadb is required for ChromaVectorStoreManager")
                self.vector_store_manager = ChromaVectorStoreManager(
                    persist_directory=getattr(config, "VECTOR_STORE_DIR", "./chroma_db")
                )

        # Async DSPy Manager Initialization
        if async_dspy_manager:
            self.async_dspy_manager = async_dspy_manager
        else:
            # Configure the global DSPy LM instance first
            # configure_dspy_with_ollama() # This is called at a higher level, or should be ensured it's called once globally
            # For now, let's assume it's configured globally elsewhere (e.g. main application setup or conftest for tests)
            # If not, it needs to be called here, but carefully to avoid reconfiguring on every Agent init.

            # Initialize AsyncDSPyManager without the lm argument
            self.async_dspy_manager = AsyncDSPyManager()

        self.action_intent_selector_program = get_optimized_action_selector()
        self.role_thought_generator_program = get_role_thought_generator()
        self.relationship_updater_program = get_relationship_updater()

        # Track retrieved memories across turns for debugging/analysis
        self._memory_history: list[dict[str, Any]] = []

        logger.info(
            f"Agent {self.agent_id} __init__: self.action_intent_selector_program is {type(self.action_intent_selector_program)}"
        )
        logger.info(
            f"Agent {self.agent_id} __init__: __dict__ after dspy programs assignment: {self.__dict__}"
        )

    def get_id(self: Self) -> str:
        """Returns the agent's unique ID."""
        return self.agent_id

    @property
    def state(self: Self) -> AgentState:
        """
        Returns the agent's current internal state.
        """
        return self._state

    def update_state(self: Self, updated_state: AgentState) -> None:
        """
        Updates the agent's internal state with a new AgentState object.

        Args:
            updated_state (AgentState): The new state for the agent.
        """
        self._state = updated_state
        logger.debug(f"Agent {self.agent_id} state updated")

    def add_memory(self: Self, step: int, memory_type: str, content: str) -> None:
        """
        Adds a memory to the agent's short-term memory.

        Args:
            step (int): The simulation step in which the memory occurred
            memory_type (str): Type of memory (e.g., 'thought',
                'broadcast_sent',
                'broadcast_received')
            content (str): The content of the memory
        """
        memory_entry = {"step": step, "type": memory_type, "content": content}
        self._state.short_term_memory.append(memory_entry)
        logger.debug(f"Added {memory_type} memory for agent {self.agent_id}")

    def update_relationship(
        self: Self, other_agent_id: str, delta: float, is_targeted: bool = False
    ) -> None:
        """Delegate to :func:`agent_actions.update_relationship`."""
        from .agent_actions import update_relationship as _update

        _update(self._state, other_agent_id, delta, is_targeted=is_targeted)

    def update_mood(self: Self, sentiment_score: float) -> None:
        """
        Updates the agent's mood based on a sentiment score.

        Args:
            sentiment_score (float): The sentiment score to apply to the mood
        """
        AgentController(self._state).update_mood(sentiment_score)

    async def run_turn(
        self: Self,
        simulation_step: int,
        environment_perception: dict[str, Any] | None = None,
        vector_store_manager: (
            MemoryStore | None
        ) = None,  # Accepts any vector store manager implementation
        knowledge_board: Optional["KnowledgeBoard"] = None,
    ) -> dict[str, Any]:
        """
        Executes the agent's internal graph for one turn, passing the previous thought.

        Args:
            simulation_step (int): The current step number from the simulation.
            environment_perception (Dict[str, Any], optional): Perception data from the
                environment.
            vector_store_manager (Optional[MemoryStore], optional): Manager for vector-based memory
                storage and retrieval. Used to persist memory events.
            knowledge_board (Optional[KnowledgeBoard], optional): Knowledge board instance
                that agents can read from and write to.

        Returns:
            Dict[str, Any]: Contains the message content, recipient ID, and action intent
        """
        logger.debug(
            f"Agent {self.agent_id} starting graph turn for simulation_step: {simulation_step}."
        )

        # Ensure environment_perception is a dictionary even if None is passed
        if environment_perception is None:
            environment_perception = {}

        # Log received perception data
        if environment_perception:
            logger.debug(f"  Received environment perception: {environment_perception}")

        # --- Retrieve previous thought ---
        for memory in self._state.short_term_memory:
            if memory["type"] == "thought":
                environment_perception["previous_thought"] = memory["content"]
                break
        # --- End Retrieve ---

        # --- Extract Knowledge Board Content ---
        knowledge_board_content = environment_perception.get("knowledge_board_content", [])
        logger.debug(
            f"  Retrieved knowledge board content (entries: {len(knowledge_board_content)})"
        )
        # --- End Extract Knowledge Board ---

        # --- Extract Simulation Scenario ---
        scenario_description = environment_perception.get("scenario_description", "")
        logger.debug(f"  Retrieved simulation scenario: '{scenario_description}'")
        # --- End Extract Simulation Scenario ---

        # --- Extract Perceived Messages ---
        perceived_messages = environment_perception.get("perceived_messages", [])
        logger.debug(f"  Retrieved {len(perceived_messages)} perceived messages")
        # --- End Extract Perceived Messages ---

        # --- Retrieve Memory History ---
        # Start with any memories retrieved in previous turns
        memory_history_list: list[dict[str, Any]] = list(self._memory_history)

        active_store = vector_store_manager or getattr(self, "vector_store_manager", None)
        if active_store is not None and hasattr(active_store, "aretrieve_relevant_memories"):
            try:
                retrieved_memories = await cast(Any, active_store).aretrieve_relevant_memories(
                    self.agent_id,
                    query="",
                    k=5,
                )
                # Persist and accumulate retrieved memories for this agent
                self._memory_history.extend(retrieved_memories)
                memory_history_list = list(self._memory_history)

            except Exception as e:  # pragma: no cover - defensive
                logger.error(
                    f"Agent {self.agent_id}: failed to retrieve memories: {e}",
                    exc_info=True,
                )

        # Convert the state to dictionary for compatibility with the existing graph
        if _PYDANTIC_V2:
            state_dict = cast(BaseModel, self._state).model_dump()
        else:
            state_dict = cast(BaseModel, self._state).dict()

        # Extract agent goal - handle goals which may be in different formats:
        # 1. From the AgentState goals list (which might be empty)
        # 2. From a flat 'goal' in the state_dict (legacy format)
        # 3. Default to "Contribute to the simulation" if neither is found
        agent_goal = "Contribute to the simulation"

        # First check goals list - if it has entries, use the first one's description
        if state_dict.get("goals") and len(state_dict["goals"]) > 0:
            agent_goal = state_dict["goals"][0].get("description", agent_goal)
        # Otherwise, check if there's a single 'goal' string in the state
        elif state_dict.get("goal"):
            agent_goal = state_dict["goal"]

        logger.debug(f"  Using agent goal: '{agent_goal}'")

        # Prepare the input state for this turn's graph execution
        initial_turn_state: AgentTurnState = {
            "agent_id": self.agent_id,
            "current_state": (
                cast(BaseModel, self._state).model_dump(exclude_none=True)
                if _PYDANTIC_V2
                else cast(BaseModel, self._state).dict(exclude_none=True)
            ),  # Current full state
            "simulation_step": simulation_step,
            "previous_thought": self._state.last_thought,
            "environment_perception": environment_perception,
            "perceived_messages": copy.deepcopy(
                environment_perception.get("perceived_messages", [])
            ),
            "memory_history_list": memory_history_list,
            "turn_sentiment_score": 0,  # Placeholder
            "individual_message_sentiments": [],  # Initialize empty list for per-message sentiments
            "prompt_modifier": "",  # Initialize prompt modifier
            "structured_output": None,  # Initialize as None for this turn
            "agent_goal": agent_goal,  # Use the extracted agent goal
            "updated_state": {},  # Initialize empty
            "vector_store_manager": vector_store_manager,  # Pass the vector store manager
            "rag_summary": "(No memory summary available yet)",  # Initialize with default summary
            "knowledge_board_content": knowledge_board_content,  # Pass the knowledge board content
            "knowledge_board": knowledge_board,  # Pass the knowledge board instance
            "scenario_description": scenario_description,  # Pass the simulation scenario
            "current_role": self._state.current_role.name,  # Pass the agent's current role
            "influence_points": int(self._state.ip),  # Cast to int for TypedDict compliance
            "steps_in_current_role": self._state.steps_in_current_role,
            # Pass the agent's steps in current role
            "data_units": int(self._state.du),  # Cast to int for TypedDict compliance
            "state": self._state,  # Pass the AgentState object directly
            "agent_instance": self,  # Pass the agent instance itself for DSPy calls
            "current_project_affiliation": getattr(
                self._state, "current_project_affiliation", None
            ),
            "available_projects": getattr(self._state, "available_projects", {}),
            "collective_ip": getattr(self._state, "collective_ip", None),
            "collective_du": getattr(self._state, "collective_du", None),
        }
        from src.agents.graphs.basic_agent_graph import compute_trace_hash

        trace_hash = compute_trace_hash(initial_turn_state)
        initial_turn_state["trace_hash"] = trace_hash

        if self.graph is None:
            logger.error(f"Agent {self.agent_id} has no graph assigned.")
            # Ensure the return type matches run_turn's annotation: dict[str, Any]
            # Also, ensure AgentActionOutput is available for creating structured_output if needed by other logic.
            # For now, returning a simpler dict consistent with other error returns.
            return {
                "message_content": None,
                "message_recipient_id": None,
                "action_intent": "idle",
                "structured_output": None,
            }

        # The cast below was to object, which is too generic.
        # It should be cast to a Callable that accepts AgentTurnState and returns a state-like dict or AgentTurnState.
        # Langchain/Langgraph graphs usually return the full state or a dict to update it.
        # graph_callable = cast(Callable[[AgentTurnState], Union[AgentTurnState, dict[str, Any]]], self.graph.ainvoke)
        # Simpler: LangGraph's ainvoke on a CompiledStateGraph should return the full state (AgentTurnState).
        graph_ainvoke_callable = cast(
            Callable[[AgentTurnState], Awaitable[AgentTurnState]], self.graph.ainvoke
        )

        try:
            # Invoke the graph asynchronously for the turn
            logger.debug(
                f"Agent {self.agent_id} invoking graph. Type of graph: {type(self.graph)}"
            )
            logger.debug(
                f"Agent {self.agent_id} initial_turn_state before invoke: {initial_turn_state.keys()}"
            )
            final_result_state: AgentTurnState = await graph_ainvoke_callable(initial_turn_state)

            # Add debug logging to inspect the graph.ainvoke result
            logger.debug(
                f"RUN_TURN_GRAPH_RESULT :: Agent {self.agent_id}: Full result from graph.ainvoke: "
                f"{final_result_state}"
            )

            # --- Process Graph Output ---
            if final_result_state is None:
                logger.error(f"Agent {self.agent_id} graph execution returned None")
                return {
                    "message_content": None,
                    "message_recipient_id": None,
                    "action_intent": "idle",
                }

            # Log dictionary keys to help debug
            logger.debug(
                f"RUN_TURN_KEYS :: Agent {self.agent_id}: Available keys in graph result: "
                f"{list(final_result_state.keys())}"
            )

            # Extract the updated state from the AgentState object
            if "state" in final_result_state:
                updated_agent_state_obj = final_result_state.get("state")
                if isinstance(updated_agent_state_obj, AgentState):
                    if self.agent_id == "agent_b_analyzer_conflict":  # Check for Agent B
                        logger.debug(
                            f"DEBUG_GRAPH_OUTPUT (Agent B): BEFORE self._state update. "
                            f"final_graph_output['state'].targeted_message_multiplier type: {type(updated_agent_state_obj.targeted_message_multiplier)}, "
                            f"value: {updated_agent_state_obj.targeted_message_multiplier}"
                        )
                    self._state = updated_agent_state_obj
                    if self.agent_id == "agent_b_analyzer_conflict":  # Check for Agent B
                        logger.debug(
                            f"DEBUG_GRAPH_OUTPUT (Agent B): AFTER self._state update. "
                            f"self._state.targeted_message_multiplier type: {type(self._state.targeted_message_multiplier)}, "
                            f"value: {self._state.targeted_message_multiplier}"
                        )
                    logger.debug(  # Keep this log for state update confirmation
                        f"RUN_TURN_POST_UPDATE :: Agent {self.agent_id}: self._state updated with new "
                        f"AgentState object (Role: {self._state.current_role.name}, Mood: {self._state.mood_value})."
                    )
                else:
                    logger.warning(
                        f"RUN_TURN_POST_UPDATE :: Agent {self.agent_id}: 'state' in graph result is not "
                        f"an AgentState object (type: {type(updated_agent_state_obj)}). self._state NOT updated."
                    )

                # Extract message and action details from the 'structured_output' field
                structured_output_from_graph = final_result_state.get("structured_output")
                message_content = None
                message_recipient_id = None
                action_intent = AgentActionIntent.IDLE.value  # Default

                if isinstance(structured_output_from_graph, AgentActionOutput):
                    message_content = structured_output_from_graph.message_content
                    message_recipient_id = structured_output_from_graph.message_recipient_id
                    action_intent = structured_output_from_graph.action_intent
                    logger.debug(
                        f"Extracted from AgentActionOutput: msg='{message_content}', "
                        f"recip='{message_recipient_id}', intent='{action_intent}'"
                    )
                elif isinstance(
                    structured_output_from_graph, dict
                ):  # If it was already a dict from the graph
                    message_content = structured_output_from_graph.get("message_content")
                    message_recipient_id = structured_output_from_graph.get("message_recipient_id")
                    action_intent = structured_output_from_graph.get(
                        "action_intent", AgentActionIntent.IDLE.value
                    )
                    logger.debug(
                        f"Extracted from dict: msg='{message_content}', "
                        f"recip='{message_recipient_id}', intent='{action_intent}'"
                    )
                else:
                    logger.warning(
                        f"Agent {self.agent_id}: 'structured_output' in graph result is not "
                        f"AgentActionOutput or dict: {type(structured_output_from_graph)}. "
                        f"Content: {structured_output_from_graph}"
                    )
                    # Ensure action_intent has a valid default
                    if not isinstance(action_intent, str) or action_intent not in [
                        item.value for item in AgentActionIntent
                    ]:
                        action_intent = AgentActionIntent.IDLE.value

                turn_output = {
                    "message_content": message_content,
                    "message_recipient_id": message_recipient_id,
                    "action_intent": action_intent,
                    "trace_hash": trace_hash,
                }
                logger.debug(f"Agent {self.agent_id} run_turn returning: {turn_output}")

                # Update last_action_intent on the agent's state
                if (
                    turn_output
                    and isinstance(turn_output, dict)
                    and "action_intent" in turn_output
                ):
                    self._state.last_action_intent = AgentActionIntent(
                        turn_output["action_intent"]
                    )
                    logger.debug(
                        f"Agent {self.agent_id} updated last_action_intent to: {self._state.last_action_intent}"
                    )
                elif turn_output and hasattr(
                    turn_output, "action_intent"
                ):  # If it's an AgentActionOutput object
                    self._state.last_action_intent = turn_output.action_intent
                    logger.debug(
                        f"Agent {self.agent_id} updated last_action_intent to: {self._state.last_action_intent}"
                    )

                return turn_output
            else:
                logger.warning(
                    f"RUN_TURN_UPDATE_FAIL :: Agent {self.agent_id}: No 'state' found in graph "
                    f"result. "
                    f"self._state NOT updated."
                )
                return {
                    "message_content": None,
                    "message_recipient_id": None,
                    "action_intent": "idle",
                }

        except Exception as e:
            logger.error(
                f"Error running turn for agent {self.agent_id} at step {simulation_step}: {e}",
                exc_info=True,
            )
            return {"message_content": None, "message_recipient_id": None, "action_intent": "idle"}

    def __str__(self: Self) -> str:
        """Returns a string representation of the agent."""
        role = (
            self._state.current_role.name
            if hasattr(self._state.current_role, "name")
            else self._state.current_role
        )
        return f"Agent(id={self.agent_id}, role={role})"

    def __repr__(self: Self) -> str:
        """Returns a detailed string representation for debugging."""
        role = (
            self._state.current_role.name
            if hasattr(self._state.current_role, "name")
            else self._state.current_role
        )
        return (
            f"Agent(agent_id='{self.agent_id}', role='{role}', "
            f"ip={self._state.ip}, du={self._state.du})"
        )

    def receive_gossip(self: Self, gossip: dict[str, float]) -> None:
        """Update reputation scores based on gossip from other agents."""
        for agent_id, rep in gossip.items():
            current = self._state.reputation.get(agent_id, 0.0)
            self._state.reputation[agent_id] = (current + rep) / 2

    def role_similarity(self: Self, other_agent: "Agent") -> float:
        """Return similarity between this agent's role and another's."""
        emb1 = self._state.role_embedding
        emb2 = other_agent.state.role_embedding
        if not emb1 or not emb2:
            return 0.0
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sqrt(sum(a * a for a in emb1))
        norm2 = sqrt(sum(b * b for b in emb2))
        base_sim = dot / (norm1 * norm2) if norm1 and norm2 else 0.0
        reputation_boost = self._state.reputation.get(other_agent.agent_id, 0.0)
        return base_sim * (1 + reputation_boost)

    # --- Async DSPy Methods ---
    async def async_generate_role_prefixed_thought(
        self: Self, agent_role: str | None, current_situation: str
    ) -> object:  # DSPy async output is dynamic
        """
        Asynchronously generate a role-prefixed thought using a DSPy program.
        Uses AsyncDSPyManager for non-blocking execution. On timeout or error,
        returns a failsafe output and logs the issue.
        Must be awaited.
        """
        from src.agents.dspy_programs.role_thought_generator import (
            generate_role_prefixed_thought,
        )
        from src.agents.dspy_programs.role_thought_generator import (
            get_failsafe_output as role_thought_failsafe,
        )

        role_prompt = agent_role or self._state.role_prompt
        future = await self.async_dspy_manager.submit(
            cast(Callable[..., object], generate_role_prefixed_thought),
            agent_role=role_prompt,
            current_situation=current_situation,
        )
        result = await self.async_dspy_manager.get_result(
            future,
            default_value=role_thought_failsafe(role_prompt, current_situation),
            dspy_callable=cast(Callable[..., object], generate_role_prefixed_thought),
        )
        return result

    async def async_select_action_intent(
        self: Self,
        agent_role: str | None,
        current_situation: str,
        agent_goal: str,
        available_actions: str,
    ) -> object:  # DSPy async output is dynamic
        """
        Asynchronously select an action intent using a DSPy program.
        Uses AsyncDSPyManager for non-blocking execution. On timeout or error,
        returns a failsafe output and logs the issue.
        Must be awaited.
        """
        from src.agents.dspy_programs import action_intent_selector

        selector = action_intent_selector.get_optimized_action_selector()
        selector_callable = cast(Callable[..., object], selector)
        role_prompt = agent_role or self._state.role_prompt
        future = await self.async_dspy_manager.submit(
            selector_callable,
            agent_role=role_prompt,
            current_situation=current_situation,
            agent_goal=agent_goal,
            available_actions=available_actions,
        )
        default_value = action_intent_selector.get_failsafe_output(
            role_prompt, current_situation, agent_goal, available_actions
        )
        result = await self.async_dspy_manager.get_result(
            future,
            default_value=default_value,
            dspy_callable=selector_callable,
        )
        return result

    async def async_generate_l1_summary(
        self: Self,
        agent_role: str | None,
        recent_events: str,
        current_mood: str | None = None,
    ) -> object:  # DSPy async output is dynamic
        """
        Asynchronously generate an L1 summary using a DSPy program.
        Uses AsyncDSPyManager for non-blocking execution. On timeout or error,
        returns a failsafe output and logs the issue.
        Must be awaited.
        """
        from src.agents.dspy_programs.l1_summary_generator import L1SummaryGenerator

        l1_gen = L1SummaryGenerator()
        l1_callable = cast(Callable[..., object], l1_gen.generate_summary)
        role_prompt = agent_role or self._state.role_prompt
        future = await self.async_dspy_manager.submit(
            l1_callable, role_prompt, recent_events, current_mood
        )
        result = await self.async_dspy_manager.get_result(
            future,
            default_value=L1SummaryGenerator.get_failsafe_output(
                role_prompt, recent_events, current_mood
            ),
            dspy_callable=l1_callable,
        )
        return result

    async def async_generate_l2_summary(
        self: Self,
        agent_role: str | None,
        l1_summaries_context: str,
        overall_mood_trend: str | None = None,
        agent_goals: str | None = None,
    ) -> object:  # DSPy async output is dynamic
        """
        Asynchronously generate an L2 summary using a DSPy program.
        Uses AsyncDSPyManager for non-blocking execution. On timeout or error,
        returns a failsafe output and logs the issue.
        Must be awaited.
        """
        from src.agents.dspy_programs.l2_summary_generator import L2SummaryGenerator

        l2_gen = L2SummaryGenerator()
        l2_callable = cast(Callable[..., object], l2_gen.generate_summary)
        role_prompt = agent_role or self._state.role_prompt
        future = await self.async_dspy_manager.submit(
            l2_callable,
            role_prompt,
            l1_summaries_context,
            overall_mood_trend,
            agent_goals,
        )
        result = await self.async_dspy_manager.get_result(
            future,
            default_value=L2SummaryGenerator.get_failsafe_output(
                role_prompt,
                l1_summaries_context,
                overall_mood_trend,
                agent_goals,
            ),
            dspy_callable=l2_callable,
        )
        return result

    async def async_update_relationship(
        self: Self,
        current_relationship_score: float,
        interaction_summary: str,
        agent1_persona: str,
        agent2_persona: str,
        interaction_sentiment: float,
    ) -> object:  # DSPy async output is dynamic
        """
        Asynchronously update a relationship score using a DSPy program.
        Uses AsyncDSPyManager for non-blocking execution. On timeout or error,
        returns a failsafe output and logs the issue.
        Must be awaited.
        """
        from src.agents.dspy_programs.relationship_updater import (
            get_failsafe_output as relationship_failsafe,
        )
        from src.agents.dspy_programs.relationship_updater import (
            get_relationship_updater,
        )

        updater = get_relationship_updater()
        # Ensure updater is callable, else fallback to failsafe
        if not callable(updater):
            updater = relationship_failsafe
        updater_callable = cast(Callable[..., object], updater)
        persona1 = agent1_persona or self._state.role_prompt
        persona2 = agent2_persona or self._state.role_prompt
        future = await self.async_dspy_manager.submit(
            updater_callable,
            current_relationship_score,
            interaction_summary,
            persona1,
            persona2,
            interaction_sentiment,
        )
        result = await self.async_dspy_manager.get_result(
            future,
            default_value=relationship_failsafe(current_relationship_score),
            dspy_callable=updater_callable,
        )
        return result

    async def _broadcast_message(
        self: Self,
        content: str,
        step: int,
        recipient_id: str | None = None,
        action_intent: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Broadcasts an AgentMessage to the dashboard SSE queue.
        Args:
            content (str): The message content.
            step (int): The simulation step.
            recipient_id (str, optional): The recipient agent ID.
            action_intent (str, optional): The action intent.
            extra (dict, optional): Any extra info.
        """
        try:
            msg = AgentMessage(
                agent_id=self.agent_id,
                content=content,
                step=step,
                recipient_id=recipient_id,
                action_intent=action_intent,
                extra=extra,
            )
            await message_sse_queue.put(msg)
        except Exception as e:
            logging.error(f"Failed to broadcast message to dashboard SSE: {e}")

    def apply_action_costs_and_awards(
        self: Self,
        action_intent: str | None,
        message_content: str | None,
        recipient_id: str | None,
        state: AgentState,
    ) -> None:
        """
        Applies IP/DU costs and awards based on the action intent and message.
        This method directly modifies the passed 'state' object.
        """
        if not action_intent:
            return

        # --- IP Costs/Awards ---
        ip_change = 0.0
        if message_content:
            if recipient_id is None:  # Broadcast message
                ip_change -= state.ip_cost_per_message
                logger.debug(
                    f"Agent {self.agent_id}: Applied IP cost for broadcast: {state.ip_cost_per_message}"
                )
            else:  # Targeted message
                ip_change -= getattr(config, "IP_COST_TARGETED_MESSAGE", state.ip_cost_per_message)
                logger.debug(
                    f"Agent {self.agent_id}: Applied IP cost for targeted message: {getattr(config, 'IP_COST_TARGETED_MESSAGE', state.ip_cost_per_message)}"
                )

        if action_intent == AgentActionIntent.CREATE_PROJECT.value:
            ip_change -= getattr(config, "IP_COST_CREATE_PROJECT", 10.0)
        # Other intent-specific IP awards/costs can be added here.
        # Example: Proposing an idea might have an award, but cost to post is in graph.
        # if action_intent == AgentActionIntent.PROPOSE_IDEA.value:
        #     ip_change += getattr(config, "IP_AWARD_FOR_PROPOSAL", 0.0) # Typically cost is in graph

        start_ip = state.ip
        state.ip += ip_change
        if ip_change != 0:
            logger.info(
                f"Agent {self.agent_id}: IP changed by {ip_change:.2f} due to action/message. New IP: {state.ip:.2f}"
            )

        # --- DU Costs/Awards ---
        du_change = 0.0
        # Apply a generic DU cost for taking any action, if configured (can be 0)
        # Many specific DU costs (like PROPOSE_DETAILED_IDEA_DU_COST) are handled in graph nodes.
        # This cost here would be *additional* or for actions not covered by specific graph node costs.
        if state.du_cost_per_action > 0 and action_intent != AgentActionIntent.IDLE.value:
            du_change -= state.du_cost_per_action

        # Intent-specific DU generation/costs from ACTION_DU_GENERATION_V2
        action_du_generation_cfg = config.get_config("ACTION_DU_GENERATION_V2")
        action_du_config = {}
        if isinstance(action_du_generation_cfg, dict):
            action_du_config = action_du_generation_cfg.get(action_intent, {})
        base_du_generation = action_du_config.get("base_amount", 0.0)
        role_bonus_factor = action_du_config.get("role_bonus_factor", 0.0)

        # Get the role-specific generation dictionary (e.g., {"base": 1.0, "bonus_factor": 0.2})
        role_specific_generation_dict = config.ROLE_DU_GENERATION.get(state.current_role.name, {})
        # Extract the 'base' DU generation for the role, to be multiplied by the action's role_bonus_factor
        role_base_generation_for_action_bonus = role_specific_generation_dict.get("base", 0.0)

        # --- Start Debug Logging ---
        logger.debug(
            f"DU Calc: action_intent='{action_intent}', agent_role='{state.current_role.name}'"
        )
        logger.debug(f"DU Calc: action_du_config='{action_du_config}'")
        logger.debug(
            f"DU Calc: base_du_generation (type: {type(base_du_generation)})='{base_du_generation}'"
        )
        logger.debug(f"DU Calc: role_specific_generation_dict='{role_specific_generation_dict}'")
        logger.debug(
            f"DU Calc: role_base_generation_for_action_bonus (type: {type(role_base_generation_for_action_bonus)})='{role_base_generation_for_action_bonus}'"
        )
        logger.debug(
            f"DU Calc: role_bonus_factor (type: {type(role_bonus_factor)})='{role_bonus_factor}'"
        )
        # --- End Debug Logging ---

        # Calculate DU generated from action's base amount and role-modified bonus
        generated_du_for_action = base_du_generation + (
            role_base_generation_for_action_bonus * role_bonus_factor
        )

        du_change += generated_du_for_action

        # Apply fixed DU costs for specific intents if not handled by the generative config or if additive
        if action_intent == AgentActionIntent.PERFORM_DEEP_ANALYSIS.value:
            du_change -= getattr(config, "DU_COST_DEEP_ANALYSIS", 10.0)
        # Example: Cost for ASK_CLARIFICATION is usually in the graph node.
        # elif action_intent == AgentActionIntent.ASK_CLARIFICATION.value:
        #     du_change -= getattr(config, "DU_COST_REQUEST_CLARIFICATION", 1.0)

        start_du = state.du
        state.du += du_change
        if du_change != 0:
            logger.info(
                f"Agent {self.agent_id}: DU changed by {du_change:.2f} due to action. New DU: {state.du:.2f}"
            )

        # Clamp IP and DU to non-negative values
        state.ip = max(0, state.ip)
        state.du = max(0, state.du)

        # Record final deltas to ledger. Log even when no economic change
        # occurred so the action itself is traceable in the ledger.
        final_ip_change = state.ip - start_ip
        final_du_change = state.du - start_du
        try:
            from src.infra.ledger import ledger

            ledger.log_change(
                self.agent_id,
                final_ip_change,
                final_du_change,
                f"action:{action_intent}",
            )
        except Exception:  # pragma: no cover - ledger errors should not block
            logger.debug("Ledger logging failed", exc_info=True)

    def perceive_messages(self: Self, messages: list[SimulationMessage]) -> None:
        """Allows the agent to perceive messages from other agents or the environment."""
        if not messages:
            return

        logger.debug(f"Agent {self.agent_id} perceiving {len(messages)} messages.")

        enriched_messages_for_state_update = []
        for msg_data in messages:
            enriched_msg = msg_data.copy()
            if "sentiment_score" not in enriched_msg:  # If sentiment not already present
                content = str(enriched_msg.get("content", ""))
                content_lower = content.lower()
                mock_sentiment = 0.0  # Default
                if (
                    "disagree" in content_lower
                    or "problematic" in content_lower
                    or "concern" in content_lower
                ):
                    mock_sentiment = -0.7
                elif (
                    "agree" in content_lower
                    or "great idea" in content_lower
                    or "support" in content_lower
                ):
                    mock_sentiment = 0.6
                elif "question" in content_lower or "clarify" in content_lower:
                    mock_sentiment = 0.0
                else:
                    mock_sentiment = 0.05  # Slight positive for generic
                enriched_msg["sentiment_score"] = mock_sentiment
                logger.debug(
                    f"Agent {self.agent_id} (perceive_messages fallback): Applied mock_sentiment {mock_sentiment} to message from {enriched_msg.get('sender_id')}"
                )
            enriched_messages_for_state_update.append(enriched_msg)

        # Option 1: Directly update state (if no complex graph logic needed for perception)
        # This is the most direct way if the graph isn't invoked for perception-only updates.
        logger.debug(
            f"Agent {self.agent_id} (perceive_messages): About to call AgentController.process_perceived_messages with: {enriched_messages_for_state_update}"
        )
        logger.debug(
            f"BaseAgent ({self.agent_id}): id(self.state) before process_perceived_messages: {id(self.state)}"
        )
        AgentController(self.state).process_perceived_messages(
            cast(list[dict[str, Any]], enriched_messages_for_state_update)
        )
        logger.info(
            f"Agent {self.agent_id} processed {len(enriched_messages_for_state_update)} messages directly in perceive_messages, updating mood/relationships."
        )
        logger.debug(
            f"BaseAgent ({self.agent_id}): id(self.state) after process_perceived_messages: {id(self.state)}, relationships: {self.state.relationships}"
        )

        # Option 2: Invoke a specific part of the graph (e.g., update_state_node)
        # This requires careful setup of initial_graph_state_for_perception
        # and ensuring the graph node correctly handles this limited-scope invocation.
        # if self.graph and hasattr(self.graph, "nodes") and "update_state_node" in self.graph.nodes:
        #     try:
        #         initial_graph_state_for_perception = {
        #             "agent_id": self.agent_id,
        #             "state": self.state,
        #             "perceived_messages": enriched_messages_for_state_update, # Use enriched messages
        #             "simulation_step": self.state.last_action_step if self.state.last_action_step is not None else 0,
        #             "knowledge_board": None, # Pass if needed by update_state_node
        #             "vector_store_manager": self.vector_store_manager,
        #             "structured_output": None,
        #             "current_role": self.state.role,
        #             "steps_in_current_role": self.state.steps_in_current_role,
        #             "turn_sentiment_score": sum(m.get("sentiment_score", 0) for m in enriched_messages_for_state_update) / len(enriched_messages_for_state_update) if enriched_messages_for_state_update else 0.0
        #         }
        #         logger.debug(f"Agent {self.agent_id} (perceive_messages): Invoking update_state_node for perception.")
        #         updated_graph_state = self.graph.nodes["update_state_node"].invoke(initial_graph_state_for_perception)
        #         if "state" in updated_graph_state and isinstance(updated_graph_state["state"], AgentState):
        #             self._state = updated_graph_state["state"]
        #             logger.info(f"Agent {self.agent_id} (perceive_messages): State updated via graph node after perception.")
        #         else:
        #             logger.warning(f"Agent {self.agent_id} (perceive_messages): Graph node did not return expected state output.")
        #     except Exception as e:
        #         logger.error(f"Agent {self.agent_id} (perceive_messages): Error invoking graph node for perception: {e}")
        #         # Fallback to direct state update if graph invocation fails
        #         self.state.process_perceived_messages(enriched_messages_for_state_update, self.vector_store_manager)
        # else:
        #     # Fallback if graph or node not available
        #     self.state.process_perceived_messages(enriched_messages_for_state_update, self.vector_store_manager)

        # Consolidate memories after processing perceptions, if significant new info gained
