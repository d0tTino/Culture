# ruff: noqa: ANN101, ANN102
import logging
import random
from collections import deque
from enum import Enum
from typing import Any, Callable, Optional, Protocol, cast, TYPE_CHECKING

from pydantic import BaseModel, Extra, Field, PrivateAttr
from typing_extensions import Self

_Validator = Callable[..., Any]

if TYPE_CHECKING:
    from pydantic import ConfigDict
else:
    try:  # pragma: no cover - pydantic>=2 preferred at runtime
        from pydantic import ConfigDict
    except ImportError:  # pragma: no cover - fallback for old pydantic
        class ConfigDict(dict[str, Any]):
            """Fallback ``ConfigDict`` for pydantic < 2."""
            pass

# Local imports (ensure these are correct and not causing cycles if possible)
try:  # pragma: no cover - pydantic>=2 preferred
    from pydantic import field_validator as pyd_field_validator, model_validator as pyd_model_validator

    _field_validator: _Validator = pyd_field_validator
    _model_validator: _Validator = pyd_model_validator
    _PYDANTIC_V2 = True
except Exception:  # pragma: no cover - fallback to pydantic<2
    from pydantic import root_validator, validator

    _model_validator = cast(_Validator, root_validator)
    _field_validator = cast(_Validator, validator)
    _PYDANTIC_V2 = False

from src.agents.core.mood_utils import get_descriptive_mood, get_mood_level
from src.agents.core.roles import (
    ROLE_EMBEDDINGS,
    RoleProfile,
    create_role_profile,
    ensure_profile,
)
from .embedding_utils import compute_embedding
from src.infra.config import get_config  # Import get_config function
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.infra.llm_client import LLMClient, LLMClientConfig

logger = logging.getLogger(__name__)


def field_validator(*args: Any, **kwargs: Any) -> Any:
    """Compatibility wrapper for Pydantic field validators."""
    if not _PYDANTIC_V2:
        mode = kwargs.pop("mode", None)
        if mode == "before":
            kwargs["pre"] = True
    return _field_validator(*args, **kwargs)


def model_validator(*args: Any, **kwargs: Any) -> Any:
    """Compatibility wrapper for Pydantic model validators."""
    if not _PYDANTIC_V2:
        mode = kwargs.pop("mode", None)
        if mode == "before":
            kwargs["pre"] = True
    return _model_validator(*args, **kwargs)


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


def _get_default_role_profile() -> RoleProfile:
    name = _get_default_role()
    return create_role_profile(name)


def _generate_default_genes() -> dict[str, float]:
    """Return a set of random genes for a new agent."""
    return {f"g{i}": random.random() for i in range(3)}


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
    MOVE = "move"
    GATHER = "gather"
    BUILD = "build"


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
    AgentActionIntent.MOVE,
    AgentActionIntent.GATHER,
    AgentActionIntent.BUILD,
]


# Forward reference for Agent (used in RelationshipHistoryEntry)
if TYPE_CHECKING:
    from src.infra.llm_client import (
        OllamaClientProtocol,
        get_default_llm_client,
    )
else:
    try:
        from src.infra.llm_client import (
            OllamaClientProtocol,
            get_default_llm_client,
        )
    except Exception:  # pragma: no cover - fallback when llm_client is missing
        class OllamaClientProtocol(Protocol):
            """Fallback protocol used when the real client is unavailable."""
            ...

        def get_default_llm_client() -> OllamaClientProtocol | None:
            return None


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
    age: int = 0
    is_alive: bool = True
    inheritance: float = 0.0
    genes: dict[str, float] = Field(default_factory=_generate_default_genes)
    parent_id: Optional[str] = None
    # Memory consolidation tracking
    last_level_2_consolidation_step: int = 0
    collective_ip: float = 0.0
    collective_du: float = 0.0
    current_role: RoleProfile = Field(default_factory=_get_default_role_profile)
    steps_in_current_role: int = 0
    reputation: dict[str, float] = Field(default_factory=dict)
    role_reputation: dict[str, float] = Field(default_factory=dict)
    learned_roles: dict[str, list[float]] = Field(default_factory=dict)
    conversation_history: deque[str] = Field(
        default_factory=deque
    )  # Added for process_perceived_messages
    role_embedding: list[float] = Field(default_factory=list)
    reputation_score: float = 0.0

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

    # ------------------------------------------------------------------
    # Role embedding utilities
    # ------------------------------------------------------------------
    @property
    def role_prompt(self) -> str:
        """Return a prompt snippet derived from the embedding and reputation."""
        avg_rep = sum(self.reputation.values()) / len(self.reputation) if self.reputation else 0.0
        role_rep = self.reputation_score
        emb_str = " ".join(f"{v:.2f}" for v in self.role_embedding)
        return f"Embedding: {emb_str}; reputation: {avg_rep:.2f}; role_rep: {role_rep:.2f}"

    # ------------------------------------------------------------------
    # Compatibility properties
    # ------------------------------------------------------------------
    @property
    def role(self) -> str:
        return self.current_role.name

    @role.setter
    def role(self, value: str) -> None:
        self.current_role = create_role_profile(value)
        self.role_embedding = list(self.current_role.embedding)
        self.reputation_score = self.current_role.reputation

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

    @role_change_cooldown.setter
    def role_change_cooldown(self, value: int) -> None:
        """Set the cooldown period before another role change can occur."""
        self._role_change_cooldown = int(value)

    def get_collective_metrics_summary(self) -> dict[str, Any]:
        """Returns a summary of metrics that might be aggregated across all agents."""
        return {
            "agent_id": self.agent_id,
            "ip": self.ip,
            "du": self.du,
            "role": self.current_role.name,
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

    def apply_gossip(self, other_embedding: list[float], interaction_score: float) -> None:
        """Update internal role data based on gossip."""
        from .role_embeddings import ROLE_EMBEDDINGS

        if not self.current_role.embedding or not other_embedding:
            return
        lr = 0.1
        self.role_embedding = [
            a + lr * interaction_score * (b - a)
            for a, b in zip(self.role_embedding, other_embedding)
        ]
        self.current_role.embedding = list(self.role_embedding)
        role_name, sim = ROLE_EMBEDDINGS.nearest_role_from_embedding(other_embedding)
        if role_name:
            cur = self.role_reputation.get(role_name, 0.0)
            self.role_reputation[role_name] = (cur + sim * interaction_score) / 2
            self.learned_roles[role_name] = other_embedding
            if role_name == self.current_role.name:
                self.reputation_score = self.role_reputation[role_name]
            ROLE_EMBEDDINGS.update_role_vector(role_name, other_embedding)
            ROLE_EMBEDDINGS.update_reputation(role_name, sim * interaction_score)

    @field_validator("current_role", mode="before")
    @classmethod
    def _ensure_role_profile(cls, value: Any) -> RoleProfile:
        return ensure_profile(value)

    @field_validator("memory_store_manager", mode="before")
    @classmethod
    def _validate_memory_store_manager(cls, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "get_retriever"):  # Check for a specific method
            return value
        raise ValueError("Invalid memory_store_manager provided")

    @model_validator(mode="after")
    def _validate_model_after(cls, model: Any) -> Any:
        """Post-validation to ensure LLM client initialization works on all Pydantic versions."""
        if isinstance(model, dict):
            llm_client_config = model.get("llm_client_config")
            llm_client = model.get("llm_client")
            mock_llm_client = model.get("mock_llm_client")
        else:
            llm_client_config = model.llm_client_config
            llm_client = model.llm_client
            mock_llm_client = model.mock_llm_client

        if not llm_client:
            if mock_llm_client:
                if isinstance(model, dict):
                    model["llm_client"] = mock_llm_client
                else:
                    model.llm_client = mock_llm_client
            elif llm_client_config:
                from src.infra.llm_client import LLMClient, LLMClientConfig

                client = None
                if isinstance(llm_client_config, BaseModel):
                    client = LLMClient(config=cast(LLMClientConfig, llm_client_config))
                else:
                    client = LLMClient(config=LLMClientConfig(**llm_client_config))
                if isinstance(model, dict):
                    model["llm_client"] = client
                else:
                    model.llm_client = client
            else:
                default_client = get_default_llm_client()
                if isinstance(model, dict):
                    model["llm_client"] = default_client
                else:
                    model.llm_client = default_client

        if isinstance(model, dict):
            if not model.get("role_history"):
                cur = model.get("current_role")
                role_name = cur.name if isinstance(cur, RoleProfile) else cur
                model["role_history"] = [(model.get("step_counter", 0), role_name)]
            if not model.get("mood_history"):
                model["mood_history"] = [
                    (model.get("step_counter", 0), model.get("mood_level", 0.0))
                ]
            if not model.get("role_reputation"):
                model["role_reputation"] = {}
            if not model.get("learned_roles"):
                model["learned_roles"] = {}
            if not model.get("role_embedding"):
                cur = model.get("current_role")
                if isinstance(cur, RoleProfile):
                    model["role_embedding"] = list(cur.embedding)
                else:
                    model["role_embedding"] = compute_embedding(str(cur))
            if not model.get("reputation_score"):
                cur_role = model.get("current_role")
                if isinstance(cur_role, RoleProfile):
                    model["reputation_score"] = cur_role.reputation
                else:
                    model["reputation_score"] = 0.0
            return model
        else:
            if not model.role_history:
                role_name = model.current_role.name if isinstance(model.current_role, RoleProfile) else model.current_role
                model.role_history = [(model.step_counter, role_name)]
            if not model.mood_history:
                model.mood_history = [(model.step_counter, model.mood_level)]
            if not model.role_reputation:
                model.role_reputation = {}
            if not model.learned_roles:
                model.learned_roles = {}
            if not model.role_embedding:
                model.role_embedding = list(model.current_role.embedding)
            if not model.reputation_score:
                model.reputation_score = model.current_role.reputation
            return model

    def get_llm_client(self) -> Any:
        if not self.llm_client:
            raise ValueError("LLM client not initialized")
        return self.llm_client

    def get_memory_retriever(self) -> Any:  # VectorStoreRetriever:
        """Returns the memory retriever for the agent."""
        if not self.memory_store_manager:
            raise ValueError("MemoryStoreManager not initialized")
        return self.memory_store_manager.get_retriever()

    def mutate_genes(self: Self, mutation_rate: float) -> None:
        """Randomly mutate genes in-place."""
        mutated: dict[str, float] = {}
        for gene, value in self.genes.items():
            if random.random() < mutation_rate:
                value = min(max(value + random.uniform(-0.1, 0.1), 0.0), 1.0)
            mutated[gene] = value
        self.genes = mutated

    # ------------------------------------------------------------------
    # Serialization helpers for tests
    # ------------------------------------------------------------------
    def to_dict(self: Self, *, exclude_none: bool = False) -> dict[str, Any]:
        """Return a dictionary representation of the agent state."""
        base_model = cast(BaseModel, self)
        if _PYDANTIC_V2 and hasattr(base_model, "model_dump"):
            dump_fn = getattr(base_model, "model_dump")
            return cast(
                dict[str, Any],
                dump_fn(
                    exclude={
                        "llm_client",
                        "mock_llm_client",
                        "memory_store_manager",
                    },
                ),
            )
        return base_model.dict(exclude={"llm_client", "mock_llm_client", "memory_store_manager"})

    @classmethod
    def from_dict(cls: type[Self], data: dict[str, Any]) -> "AgentState":
        """Create an ``AgentState`` instance from a serialized dictionary."""
        clean_data = data.copy()
        if clean_data.get("memory_store_manager") is None:
            clean_data.pop("memory_store_manager", None)
        cur = clean_data.get("current_role")
        if isinstance(cur, str):
            clean_data["current_role"] = create_role_profile(cur)
        obj = cls(**clean_data)
        if not obj.llm_client:
            obj.llm_client = get_default_llm_client()
        if not obj.role_history:
            obj.role_history = [(obj.step_counter, obj.current_role.name)]
        if not obj.mood_history:
            obj.mood_history = [(obj.step_counter, obj.mood_level)]
        if not obj.role_reputation:
            obj.role_reputation = {}
        if not obj.learned_roles:
            obj.learned_roles = {}
        if not obj.role_embedding:
            obj.role_embedding = list(obj.current_role.embedding)
        if not obj.reputation_score:
            obj.reputation_score = obj.current_role.reputation
        return obj
