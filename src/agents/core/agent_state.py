# mypy: ignore-errors
import logging
import random
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

# Local imports (ensure these are correct and not causing cycles if possible)
from src.agents.core.mood_utils import get_descriptive_mood, get_mood_level
from src.infra.config import get_config  # Import get_config function

logger = logging.getLogger(__name__)


# Helper function for the default_factory to keep the lambda clean
def _get_default_role() -> str:
    ROLE_DU_GENERATION_val = get_config("ROLE_DU_GENERATION")
    logger.debug(
        f"AGENT_STATE_DEBUG (in _get_default_role): ROLE_DU_GENERATION type: {type(ROLE_DU_GENERATION_val)}, value: {ROLE_DU_GENERATION_val}"
    )
    if not isinstance(ROLE_DU_GENERATION_val, dict) or not ROLE_DU_GENERATION_val.keys():
        logger.error(
            f"AGENT_STATE_DEBUG: ROLE_DU_GENERATION is not a valid dict or is empty! Value: {ROLE_DU_GENERATION_val}. Cannot choose a default role."
        )
        return "Innovator"  # Fallback
    keys_list = list(ROLE_DU_GENERATION_val.keys())
    logger.debug(
        f"AGENT_STATE_DEBUG (in _get_default_role): list(ROLE_DU_GENERATION.keys()) type: {type(keys_list)}, value: {keys_list}"
    )
    if not keys_list:
        logger.error(
            "AGENT_STATE_DEBUG: ROLE_DU_GENERATION.keys() is empty after list conversion! Cannot choose a default role."
        )
        return "Innovator"
    chosen_role = str(random.choice(keys_list)) # Cast to str
    logger.debug(
        f"AGENT_STATE_DEBUG (in _get_default_role): Chosen role by random.choice: {chosen_role}"
    )
    return chosen_role


class AgentActionIntent(str, Enum):
    IDLE = "idle"
    CONTINUE_COLLABORATION = "continue_collaboration"
    PROPOSE_IDEA = "propose_idea"
    ASK_CLARIFICATION = "ask_clarification"
    PERFORM_DEEP_ANALYSIS = "perform_deep_analysis"
    CREATE_PROJECT = "create_project"
    JOIN_PROJECT = "join_project"
    LEAVE_PROJECT = "leave_project"
    REQUEST_ROLE_CHANGE = "request_role_change"
    SEND_DIRECT_MESSAGE = "send_direct_message"


DEFAULT_AVAILABLE_ACTIONS: list[AgentActionIntent] = [
    AgentActionIntent.IDLE,
    AgentActionIntent.CONTINUE_COLLABORATION,
    AgentActionIntent.PROPOSE_IDEA,
    AgentActionIntent.ASK_CLARIFICATION,
    AgentActionIntent.PERFORM_DEEP_ANALYSIS,
    AgentActionIntent.CREATE_PROJECT,
    AgentActionIntent.JOIN_PROJECT,
    AgentActionIntent.LEAVE_PROJECT,
    AgentActionIntent.SEND_DIRECT_MESSAGE,
]

# Forward reference for Agent (used in RelationshipHistoryEntry)
if TYPE_CHECKING:
    from src.infra.llm_client import LLMClient, LLMClientConfig  # type: ignore


class AgentStateData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    agent_id: str
    name: str
    role_history: list[tuple[int, str]] = Field(default_factory=list)

    mood_level: float = Field(default=0.0)  # Numeric mood level from -1.0 to 1.0
    mood_history: list[tuple[int, float]] = Field(
        default_factory=list
    )  # Stores (step, numeric_mood_level)

    ip: float = Field(
        default_factory=lambda: float(str(get_config("INITIAL_INFLUENCE_POINTS") or "0"))
    )
    du: float = Field(
        default_factory=lambda: float(str(get_config("INITIAL_DATA_UNITS") or "0"))
    )
    relationships: dict[str, float] = Field(default_factory=dict)
    relationship_history: dict[str, list[tuple[int, float]]] = Field(default_factory=dict)  # Stores snapshots
    short_term_memory: deque[dict[str, Any]] = Field(default_factory=deque)
    goals: list[dict[str, Any]] = Field(default_factory=list)
    projects: dict[str, dict[str, Any]] = Field(default_factory=dict)  # project_id: {details}
    current_project_id: Optional[str] = None
    llm_client_config: Optional[Any] = None  # Optional[LLMClientConfig] - Using Any temporarily
    llm_client: Optional[Any] = None
    memory_store_manager: Optional[Any] = None  # Optional[VectorStoreManager]
    mock_llm_client: Optional[Any] = None
    last_thought: Optional[str] = None
    last_clarification_question: Optional[str] = None
    last_clarification_downgraded: bool = False
    last_action_intent: Optional[AgentActionIntent] = None
    last_message_step: Optional[int] = None
    last_action_step: Optional[int] = None
    available_action_intents: list[AgentActionIntent] = Field(
        default_factory=lambda: list(DEFAULT_AVAILABLE_ACTIONS)
    )
    step_counter: int = 0  # Tracks how many steps this agent has processed
    messages_sent_count: int = 0
    messages_received_count: int = 0
    actions_taken_count: int = 0
    # Memory consolidation tracking
    last_level_2_consolidation_step: int = 0
    current_role: str = Field(default_factory=_get_default_role)
    steps_in_current_role: int = 0
    conversation_history: deque[str] = Field(default_factory=deque)  # Added for process_perceived_messages

    # Configuration parameters (will be initialized from global config)
    _max_short_term_memory: int = PrivateAttr()
    _short_term_memory_decay_rate: float = PrivateAttr()
    _relationship_decay_rate: float = PrivateAttr()
    _min_relationship_score: float = PrivateAttr()
    _max_relationship_score: float = PrivateAttr()
    _mood_decay_rate: float = PrivateAttr()
    _mood_update_rate: float = PrivateAttr()
    _ip_cost_per_message: float = PrivateAttr()
    _du_cost_per_action: float = PrivateAttr()
    _role_change_cooldown: int = PrivateAttr()
    _role_change_ip_cost: float = PrivateAttr()
    _positive_relationship_learning_rate: float = PrivateAttr()
    _negative_relationship_learning_rate: float = PrivateAttr()
    _neutral_relationship_learning_rate: float = PrivateAttr()
    _targeted_message_multiplier: float = PrivateAttr()

    @field_validator("mood_level", mode="before")  # Validate before Pydantic tries to coerce
    @classmethod
    def check_mood_level_type_before(cls, v: Any, info: ValidationInfo) -> Any:
        if not isinstance(v, (float, int)):
            logger.warning(
                f"AGENT_STATE_VALIDATOR_DEBUG ({info.data.get('agent_id', 'Unknown')}): mood_level input is not float/int before coercion. Type: {type(v)}, Value: {v}"
            )
            if isinstance(v, str) and v.lower() == "neutral":
                logger.warning(
                    f"AGENT_STATE_VALIDATOR_DEBUG ({info.data.get('agent_id', 'Unknown')}): mood_level input was 'neutral', coercing to 0.0"
                )
                return 0.0  # Attempt to coerce common problematic string to float
            # If it cannot be coerced, Pydantic will raise a validation error later if not a float
        return v

    @field_validator("mood_level", mode="after")
    @classmethod
    def check_mood_level_type_after(cls, v: float, info: ValidationInfo) -> float:
        if not isinstance(v, float):
            # This should ideally not happen if Pydantic's coercion to float worked or failed earlier
            logger.error(
                f"AGENT_STATE_VALIDATOR_ERROR ({info.data.get('agent_id', 'Unknown')}): mood_level is not float AFTER Pydantic processing. Type: {type(v)}, Value: {v}. This is unexpected."
            )
        return v

    @property
    def mood_category(self) -> str:  # Returns the string like "neutral", "positive"
        return get_mood_level(self.mood_level)  # Uses mood_utils.get_mood_level

    def model_post_init(self, __context: Any) -> None:
        # Initialize private attributes from global config
        self._max_short_term_memory = int(str(get_config("MAX_SHORT_TERM_MEMORY") or "100"))
        self._short_term_memory_decay_rate = float(
            str(get_config("SHORT_TERM_MEMORY_DECAY_RATE") or "0.1")
        )
        self._relationship_decay_rate = float(
            str(get_config("RELATIONSHIP_DECAY_FACTOR") or "0.01")
        )
        self._min_relationship_score = float(
            str(get_config("MIN_RELATIONSHIP_SCORE") or "-1.0")
        )
        self._max_relationship_score = float(
            str(get_config("MAX_RELATIONSHIP_SCORE") or "1.0")
        )
        self._mood_decay_rate = float(str(get_config("MOOD_DECAY_FACTOR") or "0.01"))
        self._mood_update_rate = float(str(get_config("MOOD_UPDATE_RATE") or "0.1"))
        self._ip_cost_per_message = float(
            str(get_config("IP_COST_SEND_DIRECT_MESSAGE") or "1.0")
        )
        self._du_cost_per_action = float(str(get_config("DU_COST_PER_ACTION") or "1.0"))
        self._role_change_cooldown = int(str(get_config("ROLE_CHANGE_COOLDOWN") or "10"))
        self._role_change_ip_cost = float(str(get_config("ROLE_CHANGE_IP_COST") or "5.0"))
        self._positive_relationship_learning_rate = float(
            str(get_config("POSITIVE_RELATIONSHIP_LEARNING_RATE") or "0.1")
        )
        self._negative_relationship_learning_rate = float(
            str(get_config("NEGATIVE_RELATIONSHIP_LEARNING_RATE") or "0.1")
        )
        self._neutral_relationship_learning_rate = float(
            str(get_config("NEUTRAL_RELATIONSHIP_LEARNING_RATE") or "0.05")
        )
        self._targeted_message_multiplier = float(
            str(get_config("TARGETED_MESSAGE_MULTIPLIER") or "1.5")
        )

        if not self.role_history:
            self.role_history.append((self.step_counter, self.current_role))
        if not self.mood_history:  # If mood_history is empty, add initial mood level
            self.mood_history.append((self.step_counter, self.mood_level))
        # Ensure initial relationship scores exist for all other agents if not provided
        # This might be better handled at the Simulation level when all agents are known

    # ... (rest of AgentStateData and AgentState classes, ensuring they use get_config() for these values)
    # For example, in update_mood:
    # change = sentiment_score * self._mood_update_rate
    # And in can_change_role:
    # if self.ip < self._role_change_ip_cost:


class AgentState(AgentStateData):  # Keep AgentState for now if BaseAgent uses it
    # ... (methods like update_mood, update_relationship, change_role, can_change_role, etc.)
    # Ensure these methods use the private attributes (e.g., self._mood_decay_rate)
    # which were initialized using get_config() in model_post_init.

    def update_mood(self, sentiment_score: float) -> None:
        current_numeric_mood_level = self.mood_level  # Access the numeric mood_level
        decayed_mood = current_numeric_mood_level * (
            1.0 - self._mood_decay_rate
        )  # _mood_decay_rate from config

        # Ensure sentiment_score is float
        sentiment_score_float = float(sentiment_score) if sentiment_score is not None else 0.0
        change = sentiment_score_float * self._mood_update_rate  # _mood_update_rate from config

        new_mood_level_numeric = decayed_mood + change
        self.mood_level = max(
            -1.0, min(1.0, new_mood_level_numeric)
        )  # Update public mood_level field

        # Log using the descriptive_mood property (which itself uses self.mood_level) and numeric mood_level
        logger.debug(
            f"AGENT_STATE MOOD_DEBUG ({self.agent_id}): Initial: {current_numeric_mood_level:.2f}, Decayed: {decayed_mood:.2f}, Sentiment: {sentiment_score_float:.2f}, Change: {change:.2f}, Final Level: {self.mood_level:.2f}, Descriptive: {self.descriptive_mood}"
        )
        self.mood_history.append(
            (self.step_counter, self.mood_level)
        )  # Store numeric mood level in history

    def update_relationship(
        self, other_agent_id: str, sentiment_score: float, is_targeted: bool = False
    ) -> None:
        logger.debug(
            f"AGENT_STATE_ENTRY_DEBUG ({self.agent_id}): Entering update_relationship. targeted_multiplier type: {type(self._targeted_message_multiplier)}, value: {self._targeted_message_multiplier}, relationship_decay_rate type: {type(self._relationship_decay_rate)}, value: {self._relationship_decay_rate}"
        )

        current_score = self.relationships.get(other_agent_id, 0.0)

        # Ensure sentiment_score is float
        sentiment_score = float(sentiment_score) if sentiment_score is not None else 0.0

        effective_sentiment = (
            sentiment_score * self._targeted_message_multiplier if is_targeted else sentiment_score
        )

        if effective_sentiment > 0:
            learning_rate = self._positive_relationship_learning_rate
        elif effective_sentiment < 0:
            learning_rate = self._negative_relationship_learning_rate
        else:
            learning_rate = self._neutral_relationship_learning_rate

        change = effective_sentiment * learning_rate
        new_score = current_score + change
        new_score = max(self._min_relationship_score, min(self._max_relationship_score, new_score))

        logger.debug(
            f"AGENT_STATE_REL_DEBUG ({self.agent_id} -> {other_agent_id}): "
            f"is_targeted={is_targeted}, sentiment_score={sentiment_score} (type: {type(sentiment_score)}), "
            f"targeted_multiplier={self._targeted_message_multiplier} (type: {type(self._targeted_message_multiplier)}), "
            f"positive_lr={self._positive_relationship_learning_rate} (type: {type(self._positive_relationship_learning_rate)}), "
            f"negative_lr={self._negative_relationship_learning_rate} (type: {type(self._negative_relationship_learning_rate)}), "
            f"chosen_lr={learning_rate} (type: {type(learning_rate)}), "
            f"relationship_decay_rate={self._relationship_decay_rate} (type: {type(self._relationship_decay_rate)})"
        )

        logger.debug(
            f"AGENT_STATE ({self.agent_id}, id: {id(self)}, relationships_dict_id: {id(self.relationships)}): "
            f"UpdateRelationship with {other_agent_id}. "
            f"Sentiment: {sentiment_score:.2f}, Targeted: {is_targeted}, "
            f"EffectiveSent: {effective_sentiment:.2f}, LR: {learning_rate:.2f}, Change: {change:.4f}. "
            f"Old score: {current_score:.2f}, Assigned new_score: {new_score:.2f}, "
            f"Retrieved after assignment: {self.relationships.get(other_agent_id, 'KEY_ERROR_AFTER_ASSIGN')}"
        )
        self.relationships[other_agent_id] = new_score
        # Verify after assignment
        final_retrieved_score = self.relationships.get(other_agent_id, "VERIFY_KEY_ERROR")
        logger.debug(
            f"FINAL CHECK in update_relationship for {self.agent_id} towards {other_agent_id}: value is {final_retrieved_score}"
        )

        # Log relationship history snapshot if it changed significantly
        if abs(new_score - current_score) > 0.01:
            if other_agent_id not in self.relationship_history:
                self.relationship_history[other_agent_id] = []
            self.relationship_history[other_agent_id].append((self.step_counter, new_score))

    def change_role(self, new_role: str, current_step: int) -> bool:
        if self.can_change_role(new_role, current_step):
            self.ip -= self._role_change_ip_cost
            logger.info(
                f"AGENT_STATE ({self.agent_id}): Role changed from {self.current_role} to {new_role}. IP cost: {self._role_change_ip_cost}. New IP: {self.ip}"
            )
            self.current_role = new_role
            self.steps_in_current_role = 0
            self.role_history.append((current_step, new_role))
            return True
        return False

    def can_change_role(self, new_role: str, current_step: int) -> bool:
        if new_role == self.current_role:
            logger.debug(
                f"AGENT_STATE ({self.agent_id}): Role change to {new_role} denied (already current role)."
            )
            return False
        if self.ip < self._role_change_ip_cost:
            logger.debug(
                f"AGENT_STATE ({self.agent_id}): Role change to {new_role} denied (insufficient IP: {self.ip} < {self._role_change_ip_cost})."
            )
            return False

        last_change_step = -1
        if self.role_history:
            # Find the step of the most recent role that is different from the current (pending) one
            # to correctly assess cooldown from the *last actual change*
            # If only one entry [ (0, initial_role) ], then last_change_step remains -1 (or 0 if preferred).
            # Let's find the step of the *previous* role change.
            # The role_history stores (step, role_name_at_that_step_start)
            # if role_history is [(0, R1), (5, R2)], last change was at step 5. Cooldown applies from step 5.
            if len(self.role_history) > 1:  # More than just the initial role
                # last_actual_role_change_step = self.role_history[-1][0] # this is the step the current role *started*
                # Find the step of the *previous* role change.
                # The role_history stores (step, role_name_at_that_step_start)
                # if role_history is [(0, R1), (5, R2)], last change was at step 5. Cooldown applies from step 5.
                last_change_step = self.role_history[-1][0] if self.role_history else 0

        if (
            (current_step - last_change_step) < self._role_change_cooldown
            and last_change_step != -1
        ):  # Make sure last_change_step is valid
            logger.debug(
                f"AGENT_STATE ({self.agent_id}): Role change to {new_role} denied (cooldown period). Current step: {current_step}, Last change: {last_change_step}, Cooldown: {self._role_change_cooldown}"
            )
            return False

        # Validate new_role against configured ROLE_DU_GENERATION if available
        role_du_generation = get_config("ROLE_DU_GENERATION")
        if isinstance(role_du_generation, dict) and new_role not in role_du_generation:
            logger.warning(
                f"AGENT_STATE ({self.agent_id}): Attempted role change to unrecognized role '{new_role}'. Denying."
            )
            return False

        return True

    @property
    def descriptive_mood(self) -> str:
        return get_descriptive_mood(self.mood_level)

    @property
    def mood_value(self) -> float:  # This property now correctly accesses self.mood_level
        return self.mood_level

    @property
    def agent_goal(self) -> str:
        # Simple goal selection: returns the first goal's description or a default.
        if self.goals and isinstance(self.goals, list) and len(self.goals) > 0:
            first_goal = self.goals[0]
            if isinstance(first_goal, dict) and "description" in first_goal:
                return str(first_goal["description"])
        return "Contribute to the simulation as effectively as possible."

    @property
    def ip_cost_per_message(self) -> float:
        return self._ip_cost_per_message

    @property
    def du_cost_per_action(self) -> float:
        return self._du_cost_per_action

    @property
    def role_change_ip_cost(self) -> float:
        return self._role_change_ip_cost

    @property
    def role_change_cooldown(self) -> int:
        return self._role_change_cooldown

    def get_collective_metrics_summary(self) -> dict[str, Any]:
        """Returns a summary of metrics that might be aggregated across all agents."""
        return {
            "agent_id": self.agent_id,
            "ip": self.ip,
            "du": self.du,
            "role": self.current_role,
            # Add other relevant metrics here if needed for collective tracking
        }

    def update_collective_metrics(self, collective_ip: float, collective_du: float) -> None:
        """Updates the agent's view of collective metrics. For now, this is a placeholder."""
        # In a more complex system, an agent might adjust its strategy based on these.
        # For now, we'll just log that it received them.
        logger.debug(f"AGENT_STATE_COLLECTIVE_METRICS for {self.agent_id} (state ID: {id(self)}):")
        logger.debug(f"  - Pre-relationships ID: {id(self.relationships)}")
        logger.debug(f"  - Pre-relationships content: {self.relationships}")
        logger.debug(
            f"  - Pre-collective_ip: {getattr(self, 'collective_ip', 'N/A')}, collective_du: {getattr(self, 'collective_du', 'N/A')}"
        )

        self.collective_ip = collective_ip  # Storing them on the state
        self.collective_du = collective_du

        logger.debug(
            f"  - Incoming collective_ip: {collective_ip}, collective_du: {collective_du}"
        )
        logger.debug(f"  - Post-relationships ID: {id(self.relationships)}")  # Should be same
        logger.debug(f"  - Post-relationships content: {self.relationships}")  # Should be same
        logger.debug(
            f"  - Post-collective_ip: {self.collective_ip}, collective_du: {self.collective_du}"
        )

    def __hash__(self) -> int:
        # Pydantic models are not hashable by default if they have mutable fields like lists/dicts.
        # For use in sets or as dict keys, if needed, a specific hash can be implemented.
        # Often, agent_id is sufficient if it's guaranteed unique.
        return hash(self.agent_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AgentState):
            return NotImplemented
        return self.agent_id == other.agent_id

    def get_current_mood(self) -> float:
        """Returns the current mood of the agent."""
        if not self.mood_history:
            return 0.0
        return self.mood_history[-1][1]

    def get_current_relationship_score(self, agent_name: str) -> float:
        """Returns the current relationship score with the specified agent."""
        if agent_name not in self.relationship_history:
            # Return a neutral score if no history exists
            return 0.0
        return self.relationship_history[-1][agent_name]

    @field_validator("memory_store_manager", mode="before")
    @classmethod
    def _validate_memory_store_manager(cls, value: Any) -> Any:
        if hasattr(value, "get_retriever"):  # Check for a specific method
            return value
        raise ValueError("Invalid memory_store_manager provided")

    @model_validator(mode="after")
    def _validate_model_after(self) -> "AgentState":
        if self.llm_client_config and not self.llm_client:
            from src.infra.llm_client import LLMClient  # type: ignore # Local import

            if self.mock_llm_client:
                self.llm_client = self.mock_llm_client
            else:
                # Ensure llm_client_config is a dict if it's from Pydantic model
                config_data = self.llm_client_config
                if hasattr(config_data, 'model_dump'): # Check if it's a Pydantic model
                    config_data = cast(dict, config_data.model_dump())
                elif not isinstance(config_data, dict):
                    raise ValueError("llm_client_config must be a Pydantic model or a dict")

                # Temporarily using LLMClientConfig directly if it's an instance
                if isinstance(self.llm_client_config, BaseModel):
                     self.llm_client = LLMClient(config=self.llm_client_config) # type: ignore
                else: # Assumes it's a dict
                     self.llm_client = LLMClient(config=LLMClientConfig(**config_data))

        return self

    def get_llm_client(self) -> "LLMClient":
        if not self.llm_client:
            raise ValueError("LLMClient not initialized")
        return cast("LLMClient", self.llm_client)

    def get_memory_retriever(self) -> Any: #VectorStoreRetriever:
        """Returns the memory retriever for the agent."""
        if not self.memory_store_manager:
            raise ValueError("MemoryStoreManager not initialized")
        return self.memory_store_manager.get_retriever() # type: ignore

    def add_memory(self, memory_text: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Adds a memory to the agent's memory store."""
        if not self.memory_store_manager:
            raise ValueError("MemoryStoreManager not initialized to add memory.")
        self.memory_store_manager.add_memory(memory_text, metadata=metadata) # type: ignore

    async def aretrieve_relevant_memories(
        self, query: str, top_k: int = 5
    ) -> list[tuple[Any, float]]: # list[tuple[Document, float]]
        """Retrieves relevant memories for the agent."""
        if not self.memory_store_manager:
            raise ValueError("MemoryStoreManager not initialized to retrieve memories.")
        return await self.memory_store_manager.aretrieve_relevant_memories(query, top_k=top_k) # type: ignore

    def reset_state(self) -> None:
        """Resets the agent's state to its initial configuration."""
        # Clear conversation history
        self.relationship_history.clear()
        # Reset mood to initial value (assuming 0.0 or a configurable initial mood)
        self.mood_history = [(0, 0.0)]  # Reset to neutral mood at turn 0
        # Reset relationship history (assuming neutral relationships initially)
        self.relationship_history = {name: [(0, 0.0)] for name in self.relationship_history.keys()}

        # Re-initialize models if they were loaded
        # This part depends on how models are loaded and if they need to be reset or reloaded.
        # For now, we'll assume they are stateless or reloaded as needed.
        # If models have internal state that needs resetting, add that logic here.
        print(f"Agent {self.name} state has been reset.")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the agent state to a dictionary."""
        # Using Pydantic's json_encoders for robust serialization including custom types
        # Exclude fields that are not easily serializable or not needed for persistence
        # or rehydration
        return self.model_dump(
            exclude={"llm_client", "memory_store_manager", "action_intent_model", "thought_model", "l1_summary_model", "mock_llm_client"}
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], llm_client_config_override: Optional[Any] = None, memory_store_manager_override: Optional[Any] = None) -> "AgentState": # memory_store_manager: Optional["VectorStoreManager"]
        """Deserializes an agent state from a dictionary."""
        # Use overrides if provided, otherwise use from data or let Pydantic handle defaults
        if llm_client_config_override:
            data['llm_client_config'] = llm_client_config_override
        if memory_store_manager_override:
            data['memory_store_manager'] = memory_store_manager_override

        # Ensure history fields are in correct format
        if "mood_history" in data and data["mood_history"] is not None:
            data["mood_history"] = [(int(turn), float(m_val)) for turn, m_val in data["mood_history"]]

        if "relationship_history" in data and data["relationship_history"] is not None:
            processed_rh = {}
            for agent_name, history_list in data["relationship_history"].items():
                processed_rh[agent_name] = [(int(turn), float(r_val)) for turn, r_val in history_list]
            data["relationship_history"] = processed_rh

        # Ensure relationships (current scores) are float
        if "relationships" in data and data["relationships"] is not None:
            data["relationships"] = {k: float(v) for k, v in data["relationships"].items()}

        return cls(**data)

    # Add methods to update dynamic config values
    def update_targeted_message_multiplier(self, value: float) -> None:
        self._targeted_message_multiplier = value

    def update_positive_relationship_learning_rate(self, value: float) -> None:
        self._positive_relationship_learning_rate = value

    def update_negative_relationship_learning_rate(self, value: float) -> None:
        self._negative_relationship_learning_rate = value

    def update_mood_decay_rate(self, value: float) -> None:
        self._mood_decay_rate = value

    def update_mood_update_rate(self, value: float) -> None:
        self._mood_update_rate = value

    def update_min_relationship_score(self, value: float) -> None:
        self._min_relationship_score = value

    def update_max_relationship_score(self, value: float) -> None:
        self._max_relationship_score = value

    def update_dynamic_config(self, key: str, value: Any) -> None:
        """Updates a dynamic configuration attribute if it exists and is private."""
        private_attr_name = f"_{key}"
        if hasattr(self, private_attr_name):
            try:
                # Attempt to cast to the type of the existing attribute
                current_val = getattr(self, private_attr_name)
                casted_value = type(current_val)(value)
                setattr(self, private_attr_name, casted_value)
                logger.info(f"AGENT_STATE ({self.agent_id}): Updated dynamic config '{key}' to '{casted_value}'.")
            except (ValueError, TypeError) as e:
                logger.error(f"AGENT_STATE ({self.agent_id}): Failed to update dynamic config '{key}' with value '{value}'. Error: {e}")
        else:
            logger.warning(f"AGENT_STATE ({self.agent_id}): Attempted to update non-existent or non-private dynamic config '{key}'.")

    def process_perceived_messages(self, messages: list[dict[str, Any]]) -> None:
        """Processes perceived messages and updates agent state (placeholder)."""
        for msg in messages:
            sender = msg.get("sender_name", "Unknown")
            content = msg.get("content", "")
            self.conversation_history.append(f"{sender}: {content}") # Add to conversation history

            # Example: Update relationship based on a simplistic sentiment from message
            # This is highly naive and should be replaced by actual sentiment analysis
            try:
                # Attempt to parse a numeric value from content as sentiment
                # This is just for demonstrating relationship update, not a real approach
                sentiment_value = float(content.split()[0]) # Naive: assumes first word is a number
                self.update_relationship(sender, sentiment_value, is_targeted=True)
            except (ValueError, IndexError):
                # If content doesn't start with a number, use neutral sentiment
                self.update_relationship(sender, 0.0, is_targeted=True)

        logger.debug(f"Agent {self.name} processed {len(messages)} messages and updated conversation history/relationships.")


# Example usage
if __name__ == "__main__":
    from src.infra.llm_client import LLMClientConfig  # Late import for example

    llm_config_instance = LLMClientConfig(model_name="test-model", api_key="test-key") # type: ignore

    agent_data_example = {
        "agent_id": "agent1",
        "name": "TestAgent",
        "current_role": "Innovator", # Matches default factory if not provided
        "mood_level": 0.5,
        "mood_history": [(0, 0.5), (1, 0.6)],
        "relationships": {"agent2": 0.8, "agent3": -0.2},
        "relationship_history": {
            "agent2": [(0, 0.7), (1, 0.8)],
            "agent3": [(0, -0.1), (1, -0.2)]
        },
        "ip": 100.0,
        "du": 50.0,
        "llm_client_config": llm_config_instance.model_dump(), # Pass as dict
        # other fields can use defaults
    }

    agent_state = AgentState(**agent_data_example)
    print(f"AgentState created: {agent_state.name}, Role: {agent_state.current_role}, Mood: {agent_state.mood_level:.2f}")
    print(f"LLM Client from state: {agent_state.get_llm_client()}")

    serialized = agent_state.to_dict()
    print(f"Serialized state: {serialized}")

    # Test deserialization with overrides
    new_llm_config = LLMClientConfig(model_name="override-model", api_key="override-key") # type: ignore
    deserialized_state = AgentState.from_dict(serialized, llm_client_config_override=new_llm_config.model_dump())
    print(f"Deserialized state: {deserialized_state.name}, LLM Model: {deserialized_state.get_llm_client().config.model_name if deserialized_state.get_llm_client() else 'None'}") # type: ignore

    agent_state.update_mood(sentiment_score=0.5)
    print(f"Mood after update: {agent_state.mood_level:.2f}, History: {agent_state.mood_history}")

    agent_state.update_relationship("agent2", sentiment_score=-0.3)
    print(f"Relationship with agent2: {agent_state.relationships.get('agent2'):.2f}, History: {agent_state.relationship_history.get('agent2')}")

    agent_state.reset_state()
    print(f"Mood after reset: {agent_state.mood_level}, Mood History: {agent_state.mood_history}")
    print(f"Relationships after reset: {agent_state.relationships}, Relationship History: {agent_state.relationship_history}")

    agent_state.update_dynamic_config("mood_decay_rate", 0.05) # Example dynamic update
    # This should reflect in subsequent mood updates if logic uses self._mood_decay_rate

    agent_state.process_perceived_messages([
        {"sender_name": "agent2", "content": "0.9 Great idea!"},
        {"sender_name": "agent3", "content": "-0.5 I disagree."}
    ])
    print(f"Conversation history: {agent_state.conversation_history}")
    print(f"Relationships with agent2 after msg: {agent_state.relationships.get('agent2'):.2f}")
    print(f"Relationships with agent3 after msg: {agent_state.relationships.get('agent3'):.2f}")
