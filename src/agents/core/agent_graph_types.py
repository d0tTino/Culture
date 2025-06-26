"""
Defines common types used in the agent's LangGraph state.
"""

from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Extra, Field
from typing_extensions import NotRequired

from src.shared.typing import SimulationMessage

# These imports are crucial for get_type_hints to resolve forward references
# in AgentTurnState when StateGraph is initialized.
from .agent_state import AgentState

if TYPE_CHECKING:
    from .base_agent import Agent
else:  # pragma: no cover - runtime stub to satisfy get_type_hints

    class Agent:  # type: ignore[too-few-type-args]
        pass


class AgentActionOutput(BaseModel):
    """Defines the expected structured output from the LLM."""

    model_config = ConfigDict(extra=Extra.forbid)
    thought: str = Field(
        ...,
        json_schema_extra={
            "description": "The agent's internal thought or reasoning for the turn."
        },
    )
    message_content: str | None = Field(
        None,
        json_schema_extra={
            "description": (
                "The message to send to other agents, or None if choosing not to send a message."
            )
        },
    )
    message_recipient_id: str | None = Field(
        None,
        json_schema_extra={
            "description": (
                "The ID of the agent this message is directed to. None means broadcast to all "
                "agents."
            )
        },
    )
    action_intent: Literal[
        "idle",
        "continue_collaboration",
        "propose_idea",
        "ask_clarification",
        "perform_deep_analysis",
        "create_project",
        "join_project",
        "leave_project",
        "send_direct_message",
    ] = Field(
        default="idle",  # Default intent
        json_schema_extra={"description": "The agent's primary intent for this turn."},
    )
    requested_role_change: str | None = Field(
        None,
        json_schema_extra={
            "description": (
                "Optional: If you wish to request a change to a different role, specify the role "
                "name here (e.g., 'Innovator', 'Analyzer', 'Facilitator'). Otherwise, leave as "
                "null."
            )
        },
    )
    project_name_to_create: str | None = Field(
        None,
        json_schema_extra={
            "description": (
                "Optional: If you want to create a new project, specify the name here. This is "
                "used with the 'create_project' intent."
            )
        },
    )
    project_description_for_creation: str | None = Field(
        None,
        json_schema_extra={
            "description": (
                "Optional: If you want to create a new project, specify the description here. "
                "This is used with the 'create_project' intent."
            )
        },
    )
    project_id_to_join_or_leave: str | None = Field(
        None,
        json_schema_extra={
            "description": (
                "Optional: If you want to join or leave a project, specify the project ID here. "
                "This is used with the 'join_project' and 'leave_project' intents."
            )
        },
    )
    justification: str | None = Field(
        None,
        json_schema_extra={
            "description": (
                "Optional: Justification for the chosen action intent, provided by the LLM."
            )
        },
    )


class AgentTurnState(TypedDict):
    """Represents the state passed into and modified by the agent's graph turn."""

    agent_id: str
    current_state: dict[str, object]  # The agent\'s full state dictionary
    simulation_step: int  # The current step number from the simulation
    previous_thought: str | None  # The thought from the *last* turn
    environment_perception: dict[str, object]  # Perception data from the environment
    perceived_messages: list[SimulationMessage]  # Messages perceived from last step
    memory_history_list: list[dict[str, Any]]  # Field for memory history list
    turn_sentiment_score: float  # Field for aggregated sentiment score.
    individual_message_sentiments: list[dict[str, Any]]  # For per-message sentiment scores
    prompt_modifier: str  # Field for relationship-based prompt adjustments
    structured_output: AgentActionOutput | None  # Holds the parsed LLM output object
    agent_goal: str  # The agent\'s goal for the simulation
    updated_state: dict[str, object]  # Output field: The updated state after the turn
    vector_store_manager: object | None  # For persisting memories to vector store
    rag_summary: str  # Summarized memories from vector store
    knowledge_board_content: list[str]  # Current entries on the knowledge board
    knowledge_board: object | None  # The knowledge board instance for posting entries
    scenario_description: str  # Description of the simulation scenario
    current_role: str  # The agent\'s current role in the simulation
    influence_points: float  # The agent\'s current Influence Points
    steps_in_current_role: int  # Steps taken in the current role
    data_units: float  # The agent\'s current Data Units
    current_project_affiliation: str | None  # The agent\'s current project ID (if any)
    available_projects: dict[str, object]  # Dictionary of available projects
    state: "AgentState"  # Forward reference to AgentState
    agent_instance: "Agent"  # Forward reference to Agent
    collective_ip: float | None  # Total IP across all agents in the simulation
    collective_du: float | None  # Total DU across all agents in the simulation
    trace_hash: NotRequired[str]
