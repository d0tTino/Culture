# ruff: noqa: ANN101, ANN102
import logging
import random
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, cast

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from typing_extensions import Self

try:  # Support pydantic >= 2 if installed
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - fallback for old pydantic
    # pydantic<2 provides ConfigDict as a plain dict
    ConfigDict = dict  # type: ignore[misc, assignment]

# Local imports (ensure these are correct and not causing cycles if possible)
from src.agents.core.mood_utils import get_descriptive_mood, get_mood_level
from src.infra.config import get_config  # Import get_config function
from src.infra.llm_client import LLMClient, LLMClientConfig

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
    chosen_role = str(random.choice(keys_list))  # Cast to str
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
def _dummy_llm_client() -> Any:
    """Fallback function returned when ``llm_client`` is unavailable."""
    return None


if TYPE_CHECKING:
    try:
        from src.infra.llm_client import (
            OllamaClientProtocol,
            get_default_llm_client,
        )
    except Exception:  # pragma: no cover - fallback when llm_client is missing
        OllamaClientProtocol = Any  # type: ignore[misc, assignment]
        get_default_llm_client = _dummy_llm_client
else:  # pragma: no cover - used only for typing
    OllamaClientProtocol = Any  # type: ignore[misc, assignment]
    get_default_llm_client = _dummy_llm_client


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
    du: float = Field(default_factory=lambda: float(str(get_config("INITIAL_DATA_UNITS") or "0")))
    relationships: dict[str, float] = Field(default_factory=dict)
    relationship_history: dict[str, list[tuple[int, float]]] = Field(
        default_factory=dict
    )  # Stores snapshots
    short_term_memory: deque[dict[str, Any]] = Field(default_factory=deque)
    goals: list[dict[str, Any]] = Field(default_factory=list)
    projects: dict[str, dict[str, Any]] = Field(default_factory=dict)  # project_id: {details}
    current_project_id: Optional[str] = None
    llm_client_config: Optional[Any] = None  # Configuration data for LLM client
    llm_client: Optional[OllamaClientProtocol] = None
    memory_store_manager: Optional[Any] = None  # Optional[VectorStoreManager]
    mock_llm_client: Optional[Any] = None

    def __init__(self, **data: Any) -> None:
        """Initialize and conditionally call ``model_post_init`` for Pydantic v1."""
        super().__init__(**data)
        if hasattr(self, "model_post_init"):
            self.model_post_init(None)

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
    collective_ip: float = 0.0
    collective_du: float = 0.0
    current_role: str = Field(default_factory=_get_default_role)
    steps_in_current_role: int = 0
    conversation_history: deque[str] = Field(
        default_factory=deque
    )  # Added for process_perceived_messages

    # Configuration parameters (will be initialized from global config)
    _max_short_term_memory: int = PrivateAttr(
        default_factory=lambda: int(str(get_config("MAX_SHORT_TERM_MEMORY") or "100"))
    )
    _short_term_memory_decay_rate: float = PrivateAttr(
        default_factory=lambda: float(str(get_config("SHORT_TERM_MEMORY_DECAY_RATE") or "0.1"))
    )
    _relationship_decay_rate: float = PrivateAttr(
        default_factory=lambda: float(str(get_config("RELATIONSHIP_DECAY_FACTOR") or "0.01"))
    )
    _min_relationship_score: float = PrivateAttr(
        default_factory=lambda: float(str(get_config("MIN_RELATIONSHIP_SCORE") or "-1.0"))
    )
    _max_relationship_score: float = PrivateAttr(
        default_factory=lambda: float(str(get_config("MAX_RELATIONSHIP_SCORE") or "1.0"))
    )
    _mood_decay_rate: float = PrivateAttr(
        default_factory=lambda: float(str(get_config("MOOD_DECAY_FACTOR") or "0.01"))
    )
    _mood_update_rate: float = PrivateAttr(
        default_factory=lambda: float(str(get_config("MOOD_UPDATE_RATE") or "0.1"))
    )
    _ip_cost_per_message: float = PrivateAttr(
        default_factory=lambda: float(str(get_config("IP_COST_SEND_DIRECT_MESSAGE") or "1.0"))
    )
    _du_cost_per_action: float = PrivateAttr(
        default_factory=lambda: float(str(get_config("DU_COST_PER_ACTION") or "1.0"))
    )
    _role_change_cooldown: int = PrivateAttr(
        default_factory=lambda: int(str(get_config("ROLE_CHANGE_COOLDOWN") or "10"))
    )
    _role_change_ip_cost: float = PrivateAttr(
        default_factory=lambda: float(str(get_config("ROLE_CHANGE_IP_COST") or "5.0"))
    )
    _positive_relationship_learning_rate: float = PrivateAttr(
        default_factory=lambda: float(
            str(get_config("POSITIVE_RELATIONSHIP_LEARNING_RATE") or "0.1")
        )
    )
    _negative_relationship_learning_rate: float = PrivateAttr(
        default_factory=lambda: float(
            str(get_config("NEGATIVE_RELATIONSHIP_LEARNING_RATE") or "0.1")
        )
    )
    _neutral_relationship_learning_rate: float = PrivateAttr(
        default_factory=lambda: float(
            str(get_config("NEUTRAL_RELATIONSHIP_LEARNING_RATE") or "0.05")
        )
    )
    _targeted_message_multiplier: float = PrivateAttr(
        default_factory=lambda: float(str(get_config("TARGETED_MESSAGE_MULTIPLIER") or "1.5"))
    )

    @property
    def positive_relationship_learning_rate(self) -> float:
        return self._positive_relationship_learning_rate

    @property
    def negative_relationship_learning_rate(self) -> float:
        return self._negative_relationship_learning_rate

    @property
    def neutral_relationship_learning_rate(self) -> float:
        return self._neutral_relationship_learning_rate

    @property
    def targeted_message_multiplier(self) -> float:
        """Multiplier applied when sending targeted messages."""
        return self._targeted_message_multiplier

    @property
    def min_relationship_score(self) -> float:
        return self._min_relationship_score

    @property
    def max_relationship_score(self) -> float:
        return self._max_relationship_score

    @field_validator("mood_level", mode="before")
    @classmethod
    def check_mood_level_type_before(cls, v: Any) -> Any:
        if not isinstance(v, (float, int)):
            logger.warning(
                "AGENT_STATE_VALIDATOR_DEBUG: mood_level input is not float/int before coercion. "
                f"Type: {type(v)}, Value: {v}"
            )
            if isinstance(v, str) and v.lower() == "neutral":
                logger.warning(
                    "AGENT_STATE_VALIDATOR_DEBUG: mood_level input was 'neutral', coercing to 0.0"
                )
                return 0.0  # Attempt to coerce common problematic string to float
            # If it cannot be coerced, Pydantic will raise a validation error later if not a float
        return v

    @field_validator("mood_level")
    @classmethod
    def check_mood_level_type_after(cls, v: float) -> float:
        if not isinstance(v, float):
            logger.error(
                "AGENT_STATE_VALIDATOR_ERROR: mood_level is not float AFTER Pydantic processing. "
                f"Type: {type(v)}, Value: {v}. This is unexpected."
            )
        return v

    @property
    def mood_category(self) -> str:  # Returns the string like "neutral", "positive"
        return get_mood_level(self.mood_level)  # Uses mood_utils.get_mood_level

    # ... (rest of AgentStateData and AgentState classes, ensuring they use get_config() for these values)
    # For example, in update_mood:
    # change = sentiment_score * self._mood_update_rate
    # And in can_change_role:
    # if self.ip < self._role_change_ip_cost:


class AgentState(AgentStateData):  # Keep AgentState for now if BaseAgent uses it
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
        history = self.relationship_history.get(agent_name, [])
        if not history:
            return 0.0
        return history[-1][1]

    @field_validator("memory_store_manager", mode="before")
    @classmethod
    def _validate_memory_store_manager(cls, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "get_retriever"):  # Check for a specific method
            return value
        raise ValueError("Invalid memory_store_manager provided")

    @model_validator(mode="after")  # type: ignore[arg-type]
    def _validate_model_after(cls, model: "AgentState") -> "AgentState":
        llm_client_config = model.llm_client_config
        llm_client = model.llm_client
        mock_llm_client = model.mock_llm_client
        if llm_client_config and not llm_client:
            if mock_llm_client:
                model.llm_client = mock_llm_client
            else:
                config_data = llm_client_config
                if hasattr(config_data, "model_dump"):
                    config_data = cast(dict[str, Any], config_data.model_dump())
                elif not isinstance(config_data, dict):
                    raise ValueError("llm_client_config must be a Pydantic model or a dict")

                if isinstance(llm_client_config, BaseModel):
                    model.llm_client = LLMClient(config=cast(LLMClientConfig, llm_client_config))
                else:
                    model.llm_client = LLMClient(config=LLMClientConfig(**config_data))

        if not model.role_history:
            model.role_history = [(model.step_counter, model.current_role)]
        if not model.mood_history:
            model.mood_history = [(model.step_counter, model.mood_level)]
        return model

    def get_llm_client(self) -> OllamaClientProtocol:
        if not self.llm_client:
            raise ValueError("LLM client not initialized")
        return self.llm_client

    def get_memory_retriever(self) -> Any:  # VectorStoreRetriever:
        """Returns the memory retriever for the agent."""
        if not self.memory_store_manager:
            raise ValueError("MemoryStoreManager not initialized")
        return self.memory_store_manager.get_retriever()

    # ------------------------------------------------------------------
    # Serialization helpers for tests
    # ------------------------------------------------------------------
    def to_dict(self: Self) -> dict[str, Any]:
        """Return a dictionary representation of the agent state."""
        return self.model_dump()

    @classmethod
    def from_dict(cls: type[Self], data: dict[str, Any]) -> "AgentState":
        """Create an ``AgentState`` instance from a serialized dictionary."""
        clean_data = data.copy()
        if clean_data.get("memory_store_manager") is None:
            clean_data.pop("memory_store_manager")
        return cls(**clean_data)
