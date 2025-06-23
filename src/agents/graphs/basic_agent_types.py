"""Type definitions for the basic agent graph."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Extra, Field
from typing_extensions import NotRequired

from src.agents.core.agent_state import AgentState


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
        default="idle",
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


class AgentTurnState(TypedDict):
    """Represents the state passed into and modified by the agent's graph turn."""

    agent_id: str
    current_state: dict[str, object]
    simulation_step: int
    previous_thought: str | None
    environment_perception: dict[str, object]
    perceived_messages: list[dict[str, object]]
    memory_history_list: list[dict[str, Any]]
    turn_sentiment_score: int
    prompt_modifier: str
    structured_output: AgentActionOutput | None
    agent_goal: str
    updated_state: dict[str, object]
    vector_store_manager: object | None
    rag_summary: str
    knowledge_board_content: list[str]
    knowledge_board: object | None
    scenario_description: str
    current_role: str
    influence_points: int
    steps_in_current_role: int
    data_units: int
    current_project_affiliation: str | None
    available_projects: dict[str, object]
    state: AgentState
    collective_ip: float | None
    collective_du: float | None
    trace_hash: NotRequired[str]
