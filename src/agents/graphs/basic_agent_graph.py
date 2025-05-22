# src/agents/graphs/basic_agent_graph.py
"""
Defines the basic LangGraph structure for an agent's turn.
"""

import logging
import os
import random  # Add import for random to test negative sentiment
import re  # Add import for regular expressions
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
)

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from src.agents.core.agent_state import AgentState
from src.agents.core.roles import (
    ROLE_ANALYZER,
    ROLE_DESCRIPTIONS,
    ROLE_FACILITATOR,
    ROLE_INNOVATOR,
)

# Import L1SummaryGenerator for DSPy-based L1 summary generation
from src.agents.dspy_programs.l1_summary_generator import L1SummaryGenerator

# Import L2SummaryGenerator for DSPy-based L2 summary generation
from src.agents.dspy_programs.l2_summary_generator import L2SummaryGenerator
from src.infra import config  # Import config for role change parameters
from src.infra.llm_client import (  # Updated import structure to match function name
    analyze_sentiment,
    generate_structured_output,
    generate_text,
)

# At the top of the file, after other DSPy imports
try:
    import os

    from dspy import Predict

    from src.agents.dspy_programs.relationship_updater import RelationshipUpdater

    _REL_UPDATER_PATH = os.path.join(
        os.path.dirname(__file__), "../dspy_programs/compiled/optimized_relationship_updater.json"
    )
    if os.path.exists(_REL_UPDATER_PATH):
        _optimized_relationship_updater = Predict.load(RelationshipUpdater, _REL_UPDATER_PATH)
    else:
        _optimized_relationship_updater = None
except Exception as e:
    _optimized_relationship_updater = None
    logger.error(f"Failed to load OptimizedRelationshipUpdater: {e}")

# Add dspy imports with detailed error handling
try:
    logging.info("Attempting to import DSPy modules...")
    sys.path.insert(0, os.path.abspath("."))  # Ensure the root directory is in the path
    import dspy

    from src.agents.dspy_programs.action_intent_selector import get_optimized_action_selector

    # Only set DSPY_AVAILABLE if LM is configured
    DSPY_AVAILABLE = hasattr(dspy.settings, "lm") and dspy.settings.lm is not None
    logging.info(
        f"[DSPy] DSPY_AVAILABLE set to {DSPY_AVAILABLE} (LM: {getattr(dspy.settings, 'lm', None)})"
    )
    DSPY_ACTION_SELECTOR_AVAILABLE = False  # Will be set to True if loading succeeds
    logging.info("DSPy modules imported successfully")

    # Initialize the action selector (loads the compiled program)
    # Try to load it once at module level to avoid repeated loads
    try:
        optimized_action_selector = get_optimized_action_selector()
        DSPY_ACTION_SELECTOR_AVAILABLE = True
        logging.info("Successfully loaded DSPy optimized_action_selector.")
    except Exception as e:
        logging.error(
            f"Failed to load DSPy optimized_action_selector: {e}. Action intent selection will use fallback.",
            exc_info=True,
        )
        optimized_action_selector = None
except ImportError as e:
    logging.error(f"Failed to import DSPy modules: {e}")
    # Get detailed traceback info
    import traceback

    logging.error(f"Import traceback: {traceback.format_exc()}")
    DSPY_AVAILABLE = False
    DSPY_ACTION_SELECTOR_AVAILABLE = False
    logging.warning("DSPy modules not available - will use standard LLM prompt")
    optimized_action_selector = None

# Use TYPE_CHECKING to avoid circular import issues
if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Decay factors for mood and relationships (loaded from config)
MOOD_DECAY_FACTOR = config.MOOD_DECAY_FACTOR
RELATIONSHIP_DECAY_FACTOR = config.RELATIONSHIP_DECAY_FACTOR

# IP award constants (loaded from config)
IP_AWARD_FOR_PROPOSAL = config.IP_AWARD_FOR_PROPOSAL
IP_COST_TO_POST_IDEA = config.IP_COST_TO_POST_IDEA

# Role change constants (loaded from config)
ROLE_CHANGE_IP_COST = config.ROLE_CHANGE_IP_COST
ROLE_CHANGE_COOLDOWN = config.ROLE_CHANGE_COOLDOWN

# Data Units constants (loaded from config)
INITIAL_DATA_UNITS = config.INITIAL_DATA_UNITS
ROLE_DU_GENERATION = config.ROLE_DU_GENERATION
PROPOSE_DETAILED_IDEA_DU_COST = config.PROPOSE_DETAILED_IDEA_DU_COST
DU_AWARD_IDEA_ACKNOWLEDGED = config.DU_AWARD_IDEA_ACKNOWLEDGED
DU_AWARD_SUCCESSFUL_ANALYSIS = config.DU_AWARD_SUCCESSFUL_ANALYSIS
DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE = config.DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE
DU_COST_DEEP_ANALYSIS = config.DU_COST_DEEP_ANALYSIS
DU_COST_REQUEST_DETAILED_CLARIFICATION = config.DU_COST_REQUEST_DETAILED_CLARIFICATION

# List of valid roles
VALID_ROLES = [ROLE_FACILITATOR, ROLE_INNOVATOR, ROLE_ANALYZER]


# Define the Pydantic model for structured LLM output
class AgentActionOutput(BaseModel):
    """Defines the expected structured output from the LLM."""

    model_config = ConfigDict(extra="forbid")
    thought: str = Field(
        ...,
        json_schema_extra={
            "description": "The agent's internal thought or reasoning for the turn."
        },
    )
    message_content: str | None = Field(
        None,
        json_schema_extra={
            "description": "The message to send to other agents, or None if choosing not to send a message."
        },
    )
    message_recipient_id: str | None = Field(
        None,
        json_schema_extra={
            "description": "The ID of the agent this message is directed to. None means broadcast to all agents."
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
            "description": "Optional: If you wish to request a change to a different role, specify the role name here (e.g., 'Innovator', 'Analyzer', 'Facilitator'). Otherwise, leave as null."
        },
    )
    project_name_to_create: str | None = Field(
        None,
        json_schema_extra={
            "description": "Optional: If you want to create a new project, specify the name here. This is used with the 'create_project' intent."
        },
    )
    project_description_for_creation: str | None = Field(
        None,
        json_schema_extra={
            "description": "Optional: If you want to create a new project, specify the description here. This is used with the 'create_project' intent."
        },
    )
    project_id_to_join_or_leave: str | None = Field(
        None,
        json_schema_extra={
            "description": "Optional: If you want to join or leave a project, specify the project ID here. This is used with the 'join_project' and 'leave_project' intents."
        },
    )


# Define the state the graph will operate on during a single agent turn
class AgentTurnState(TypedDict):
    """Represents the state passed into and modified by the agent's graph turn."""

    agent_id: str
    current_state: Dict[str, Any]  # The agent's full state dictionary (for backward compatibility)
    simulation_step: int  # The current step number from the simulation
    previous_thought: str | None  # The thought from the *last* turn
    environment_perception: Dict[str, Any]  # Perception data from the environment
    perceived_messages: List[
        Dict[str, Any]
    ]  # Messages perceived from last step (broadcasts and targeted)
    memory_history_list: List[Dict[str, Any]]  # Field for memory history list
    turn_sentiment_score: int  # Field for aggregated sentiment score
    prompt_modifier: str  # Field for relationship-based prompt adjustments
    structured_output: Optional[AgentActionOutput]  # Holds the parsed LLM output object
    agent_goal: str  # The agent's goal for the simulation
    updated_state: Dict[
        str, Any
    ]  # Output field: The updated state after the turn (for backward compatibility)
    vector_store_manager: Optional[Any]  # For persisting memories to vector store
    rag_summary: str  # Summarized memories from vector store
    knowledge_board_content: List[str]  # Current entries on the knowledge board
    knowledge_board: Optional[Any]  # The knowledge board instance for posting entries
    scenario_description: str  # Description of the simulation scenario
    current_role: str  # The agent's current role in the simulation
    influence_points: int  # The agent's current Influence Points
    steps_in_current_role: int  # Steps taken in the current role
    data_units: int  # The agent's current Data Units
    current_project_affiliation: Optional[str]  # The agent's current project ID (if any)
    available_projects: Dict[str, Any]  # Dictionary of available projects
    state: AgentState  # The agent's structured state object (new Pydantic model)
    collective_ip: Optional[float]  # Total IP across all agents in the simulation
    collective_du: Optional[float]  # Total DU across all agents in the simulation


# --- Node Functions ---


def analyze_perception_sentiment_node(state: AgentTurnState) -> dict[str, Any]:
    """
    Analyzes the sentiment of perceived messages from the previous step.
    Calculates an aggregated sentiment score for the turn.
    """
    agent_id = state["agent_id"]
    sim_step = state["simulation_step"]
    perceived_messages = state.get("perceived_messages", [])
    logger.debug(
        f"Node 'analyze_perception_sentiment_node' executing for agent {agent_id} at step {sim_step}"
    )

    total_sentiment_score = 0
    analyzed_count = 0

    if not perceived_messages:
        logger.debug("  No messages perceived, sentiment score remains 0.")
        return {"turn_sentiment_score": 0}

    for msg in perceived_messages:
        sender_id = msg.get("sender_id", "unknown")
        message_content = msg.get("content", None)
        msg.get("recipient_id", None)

        # Optional: Add special handling for targeted messages
        # For example, private messages might have more emotional impact

        # Skip analyzing own messages if they are included in perception
        if sender_id == agent_id:
            continue

        if message_content:
            sentiment = analyze_sentiment(message_content)  # Use the utility function
            if sentiment == "positive":
                total_sentiment_score += 1
                analyzed_count += 1
            elif sentiment == "negative":
                total_sentiment_score -= 1
                analyzed_count += 1
            elif sentiment == "neutral":
                analyzed_count += 1  # Count neutral messages but don't change score
            # else: sentiment is None (error occurred), do nothing

    logger.info(
        f"Agent {agent_id}: Aggregated sentiment score from {analyzed_count} perceived messages: {total_sentiment_score}"
    )
    # Return the calculated score to be added to the graph state
    return {"turn_sentiment_score": total_sentiment_score}


def prepare_relationship_prompt_node(state: AgentTurnState) -> dict[str, str]:
    """
    Prepares a detailed relationship prompt modifier with specific guidance on how
    relationships should influence the agent's decisions, message tone, target selection,
    and action intent.
    """
    from src.infra.config import get_relationship_label

    state["agent_id"]
    agent_state = state["state"]
    relationships = agent_state.relationships

    # Default neutral modifier
    modifier = "You have neutral relationships with all agents. Maintain a balanced approach in your interactions."

    # Only generate relationship-based prompts if we have relationships
    if relationships:
        # Get descriptive labels for all relationships
        relationship_descriptions = []
        strong_positive_relations = []
        strong_negative_relations = []

        for other_id, score in relationships.items():
            label = get_relationship_label(score)
            relationship_descriptions.append(
                f"- Relationship with Agent_{other_id}: {label} (Score: {score:.2f})"
            )

            # Track strongly positive/negative relationships for targeted guidance
            if score > 0.5:
                strong_positive_relations.append(other_id)
            elif score < -0.5:
                strong_negative_relations.append(other_id)

        # Generate primary relationship summary
        relationship_summary = "RELATIONSHIPS:\n" + "\n".join(relationship_descriptions)

        # Generate explicit guidance for the LLM
        guidance_parts = ["RELATIONSHIP INFLUENCE GUIDANCE:"]

        # Target selection guidance
        target_guidance = "When choosing a target for messages or actions:"
        if strong_positive_relations:
            agents_str = ", ".join([f"Agent_{aid}" for aid in strong_positive_relations[:3]])
            target_guidance += f" Prefer to target {agents_str} for collaboration since you have positive relationships with them."
        if strong_negative_relations:
            agents_str = ", ".join([f"Agent_{aid}" for aid in strong_negative_relations[:3]])
            if strong_positive_relations:
                target_guidance += f" Be more cautious when engaging with {agents_str} due to your negative relationships."
            else:
                target_guidance += f" Be cautious when engaging with {agents_str} due to your negative relationships."
        guidance_parts.append(target_guidance)

        # Message tone guidance
        tone_guidance = "Adjust your communication tone based on relationships:"
        tone_guidance += " Use warm, friendly language with allies; neutral, professional tone with neutral relationships; cautious, formal tone with negative relationships."
        guidance_parts.append(tone_guidance)

        # Action intent guidance
        intent_guidance = "Let relationships influence your action intentions:"
        intent_guidance += (
            " Consider 'continue_collaboration' or 'propose_idea' with positive relationships;"
        )
        intent_guidance += " 'ask_clarification' for neutral relationships;"
        intent_guidance += " 'idle' or careful 'ask_clarification' with negative relationships."
        guidance_parts.append(intent_guidance)

        # Decision making influence
        decision_guidance = "When making decisions:"
        decision_guidance += (
            " Give more weight to input from agents you have positive relationships with;"
        )
        decision_guidance += " Critically evaluate suggestions from agents with whom you have negative relationships;"
        decision_guidance += " When opinions differ, prioritize input from your closest allies."
        guidance_parts.append(decision_guidance)

        # Combine all parts for the final modifier
        new_line = "\n"
        modifier = f"{relationship_summary}{new_line}{new_line}{new_line.join(guidance_parts)}"

    return {"prompt_modifier": modifier}


async def retrieve_and_summarize_memories_node(state: AgentTurnState) -> dict[str, str]:
    """
    Asynchronously retrieve and summarize memories using the agent's async DSPy method.
    Uses AsyncDSPyManager for non-blocking DSPy calls. On timeout or error, returns a failsafe output and logs the issue.
    Must be awaited in the graph.
    """
    agent = state.get("agent_instance")
    agent_id = state["agent_id"]
    agent_goal = state.get("agent_goal", "Contribute to the simulation effectively.")
    vector_store_manager = state.get("vector_store_manager")
    sim_step = state["simulation_step"]
    recent_thought = state.get("previous_thought", "")
    recent_message = ""
    perceived_messages = state.get("perceived_messages", [])
    if perceived_messages and len(perceived_messages) > 0:
        recent_message = perceived_messages[-1].get("content", "")
    if not vector_store_manager or agent is None:
        logger.warning(
            f"Agent {agent_id}: No vector store manager or agent instance available for memory retrieval."
        )
        result = {"rag_summary": "(No memory retrieval available: vector store or agent missing)"}
        logger.info(
            f"[RAG VERIFICATION] Agent {agent_id} RAG Summary after retrieval: {result['rag_summary']}"
        )
        return result
    query_text = f"Memories relevant to: Goal '{agent_goal}'"
    if recent_thought:
        query_text += f" Recent thought: {recent_thought}"
    if recent_message:
        query_text += f" Recent message: {recent_message}"
    logger.debug(
        f"Agent {agent_id} at step {sim_step}: Retrieving memories with query: '{query_text}'"
    )
    try:
        retrieved_memories = await vector_store_manager.aretrieve_relevant_memories(
            agent_id, query=query_text, k=5
        )
        if not retrieved_memories or len(retrieved_memories) == 0:
            logger.info(f"Agent {agent_id}: No relevant memories found for query.")
            result = {"rag_summary": "(No relevant past memories found via RAG)"}
            logger.info(
                f"[RAG VERIFICATION] Agent {agent_id} RAG Summary after retrieval: {result['rag_summary']}"
            )
            return result
        logger.info(f"Agent {agent_id}: Retrieved {len(retrieved_memories)} memories.")
        logger.info(
            f"[RAG VERIFICATION] Agent {agent_id} Retrieved memories: {retrieved_memories}"
        )
        current_role = state.get("current_role", "unknown")
        current_context = f"Current role: {current_role}. Step: {sim_step}."
        if recent_thought:
            current_context += f" Recent thought: {recent_thought}"
        # Use agent's async_generate_l1_summary for summarization
        summary = await agent.async_generate_l1_summary(
            current_role, retrieved_memories, state.get("current_mood", None)
        )
        logger.info(
            f"Agent {agent_id}: Memory summarization complete. Summary length: {len(summary)}"
        )
        logger.debug(f"Agent {agent_id}: Memory summary: '{summary}'")
        result = {"rag_summary": summary}
        logger.info(
            f"[RAG VERIFICATION] Agent {agent_id} RAG Summary after retrieval: {result['rag_summary']}"
        )
        return result
    except Exception as e:
        logger.error(
            f"Agent {agent_id}: Error during memory retrieval and summarization: {e}",
            exc_info=True,
        )
        result = {"rag_summary": "(Memory retrieval failed due to an error)"}
        logger.info(
            f"[RAG VERIFICATION] Agent {agent_id} RAG Summary after retrieval: {result['rag_summary']}"
        )
        return result


async def generate_thought_and_message_node(
    state: AgentTurnState,
) -> dict[str, AgentActionOutput | None]:
    """
    Asynchronously generate the agent's thought and message for the turn using LLM or DSPy.
    Uses AsyncDSPyManager for non-blocking DSPy calls. On timeout or error, returns a failsafe output and logs the issue.
    Must be awaited in the graph.
    """
    agent = state.get("agent_instance")
    agent_id = state["agent_id"]
    state["simulation_step"]
    prev_thought = state.get("previous_thought", None)
    perception = state.get("environment_perception", {})
    perceived_messages = state.get("perceived_messages", [])
    state.get("turn_sentiment_score", 0)
    prompt_modifier = state.get("prompt_modifier", "")
    agent_goal = state.get(
        "agent_goal", "Contribute to the simulation as effectively as possible."
    )
    rag_summary = state.get("rag_summary", "(No relevant past memories found via RAG)")
    knowledge_board_content = state.get("knowledge_board_content", [])
    scenario_description = state.get("scenario_description", "")
    agent_state = state["agent_state"] if "agent_state" in state else state.get("state")
    role = agent_state.role
    raw_role_name = role
    role_description = ROLE_DESCRIPTIONS.get(role, f"A person with the role of {role}.")
    recent_messages_str = "".join(
        [
            "- " + m.get("sender_name", "Someone") + ": " + m.get("content", "") + "\n"
            for m in perceived_messages
        ]
    )
    previous_thoughts_str = prev_thought if prev_thought else "This is your first turn."
    current_situation = (
        "Environment: "
        + perception.get("description", "You are in a collaborative simulation environment.")
        + "\n\n"
        + "Recent messages: "
        + recent_messages_str
        + "\n\n"
        + "Relevant knowledge: "
        + str(knowledge_board_content)
        + "\n\n"
        + "Your previous thoughts: "
        + previous_thoughts_str
        + "\n\n"
        + "Goals: "
        + agent_goal
        + "\n\n"
        + "Relevant memories: "
        + rag_summary
        + "\n\n"
        + "Scenario: "
        + scenario_description
    )
    # PHASE 1: Use DSPy to generate the role-prefixed thought start
    role_prefixed_thought_start = None
    if agent is not None:
        try:
            result = await agent.async_generate_role_prefixed_thought(
                raw_role_name, current_situation
            )
            role_prefixed_thought_start = getattr(result, "thought_process", None) or getattr(
                result, "thought", None
            )
        except Exception:
            role_prefixed_thought_start = None
    if not role_prefixed_thought_start:
        role_prefix = "an" if role[0].lower() in ["a", "e", "i", "o", "u"] else "a"
        role_prefixed_thought_start = (
            f"As {role_prefix} {role}, I need to consider the current situation carefully."
        )
    # PHASE 2: Use DSPy to select action intent and generate justification
    chosen_intent_enum = None
    justification_for_intent = None
    from src.agents.core.agent_state import AgentActionIntent

    available_actions = [intent.value for intent in AgentActionIntent]
    if agent is not None:
        try:
            action_selection_prediction = await agent.async_select_action_intent(
                raw_role_name, current_situation, agent_goal, available_actions
            )
            chosen_intent_str = getattr(action_selection_prediction, "chosen_action_intent", None)
            justification_for_intent = getattr(
                action_selection_prediction, "justification_thought", None
            )
            if chosen_intent_str:
                try:
                    chosen_intent_enum = AgentActionIntent(chosen_intent_str)
                except Exception:
                    chosen_intent_enum = AgentActionIntent.IDLE
        except Exception:
            chosen_intent_enum = None
            justification_for_intent = None
    # PHASE 3: Combine thoughts for the final agent thought
    if justification_for_intent:
        if justification_for_intent.startswith(
            "As a " + raw_role_name + ","
        ) or justification_for_intent.startswith("As an " + raw_role_name + ","):
            prefix_end = justification_for_intent.find(",")
            if prefix_end > 0:
                justification_for_intent = justification_for_intent[prefix_end + 1 :].strip()
        final_agent_thought = role_prefixed_thought_start + " " + justification_for_intent
    else:
        final_agent_thought = role_prefixed_thought_start
    # PHASE 4: Prepare prompt for the main LLM call
    # Set up the base prompt with context
    prompt_parts = []
    prompt_parts.append("# AGENT INFORMATION")
    prompt_parts.append(f"Agent ID: {agent_id}")
    prompt_parts.append(f"Role: {role}")
    prompt_parts.append(f"Role Description: {role_description}")
    prompt_parts.append(f"Goal: {agent_goal}")

    if scenario_description:
        prompt_parts.append("\n# SCENARIO")
        prompt_parts.append(scenario_description)

    # Include environment perception details
    if perception:
        prompt_parts.append("\n# ENVIRONMENT")
        prompt_parts.append(
            perception.get("description", "(No environment description available)")
        )
        if "other_agents" in perception:
            prompt_parts.append(
                _format_other_agents(perception["other_agents"], agent_state.relationships)
            )

    # Include KB content if available
    if knowledge_board_content:
        prompt_parts.append("\n# KNOWLEDGE BOARD")
        prompt_parts.append(_format_knowledge_board(knowledge_board_content))

    # RAG-retrieved memories section
    prompt_parts.append("\n# RELEVANT PAST MEMORIES")
    prompt_parts.append(rag_summary)

    # Messages from previous step
    if perceived_messages:
        prompt_parts.append("\n# MESSAGES FROM OTHERS")
        prompt_parts.append(_format_messages(perceived_messages))

    # Add the relationship context to guide how to interact with others
    prompt_parts.append("\n# RELATIONSHIP CONTEXT")
    prompt_parts.append(prompt_modifier)

    # Previous thought (if any)
    if prev_thought:
        prompt_parts.append("\n# YOUR PREVIOUS THOUGHT")
        prompt_parts.append(prev_thought)

    # Current mood for context
    if hasattr(agent_state, "mood_value"):
        mood_value = agent_state.mood_value
        mood_level = get_mood_level(mood_value)
        descriptive_mood = get_descriptive_mood(mood_value)
        prompt_parts.append("\n# YOUR CURRENT MOOD")
        prompt_parts.append(
            f"Mood level: {mood_level} ({descriptive_mood}, value: {mood_value:.2f})"
        )

    # Include the final combined thought to guide the LLM
    prompt_parts.append("\n# YOUR INTERNAL THOUGHT PROCESS (follow this reasoning):")
    prompt_parts.append(final_agent_thought)

    # Provide guidance on the chosen action intent if available
    if chosen_intent_enum:
        prompt_parts.append(
            "\n# YOUR CHOSEN ACTION INTENT (execute this): " + chosen_intent_enum.value
        )
        prompt_parts.append(
            "Based on your thought process and chosen action intent, formulate your response below."
        )
    else:
        prompt_parts.append("\n# ACTION INTENT OPTIONS")
        prompt_parts.append(
            "Based on your thought process, select one of the following action intents:"
        )
        for intent in AgentActionIntent:
            prompt_parts.append(f"- {intent.value}")

    # Add final instructions for formulating the structured response
    prompt_parts.append(
        "\nRespond with a structured output containing your thought, chosen action intent, and message content (if any)."
    )

    # Combine all prompt parts
    full_prompt = "\n\n".join(prompt_parts)

    # Define the output schema based on the AgentActionOutput model
    output_schema = {
        "thought": "The agent's internal thought or reasoning for the turn.",
        "action_intent": "The agent's primary intent for this turn (choose from: idle, continue_collaboration, propose_idea, ask_clarification, perform_deep_analysis, create_project, join_project, leave_project, send_direct_message).",
        "message_content": "The message to send to other agents, or null if choosing not to send a message.",
        "message_recipient_id": "The ID of the agent this message is directed to. null means broadcast to all agents.",
        "requested_role_change": "Optional: If you wish to request a change to a different role, specify the role name here (e.g., 'Innovator', 'Analyzer', 'Facilitator'). Otherwise, leave as null.",
        "project_name_to_create": "Optional: If you want to create a new project, specify the name here. This is used with the 'create_project' intent.",
        "project_description_for_creation": "Optional: If you want to create a new project, specify the description here. This is used with the 'create_project' intent.",
        "project_id_to_join_or_leave": "Optional: If you want to join or leave a project, specify the project ID here. This is used with the 'join_project' and 'leave_project' intents.",
    }

    # Generate the structured output using the full prompt and schema
    structured_llm_output = generate_structured_output(
        full_prompt,
        response_model=AgentActionOutput,
        model=agent_state.llm_model_name if hasattr(agent_state, "llm_model_name") else None,
    )

    # If we have a structured output, override thought and intent if DSPy provided them
    if structured_llm_output:
        # Always use our combined thought for consistency
        structured_llm_output.thought = final_agent_thought

        # Only override the action intent if DSPy provided one
        if chosen_intent_enum:
            structured_llm_output.action_intent = chosen_intent_enum.value
    else:
        # Handle failure case - create a basic output
        logger.error(f"Agent {agent_id}: Failed to generate structured output, using fallback")
        structured_llm_output = AgentActionOutput(
            thought=final_agent_thought,
            message_content=None,
            message_recipient_id=None,
            action_intent=chosen_intent_enum.value if chosen_intent_enum else "idle",
        )

    # Return the structured output
    return {"structured_output": structured_llm_output}


def process_role_change(agent_state: AgentState, requested_role: str) -> bool:
    """
    Process a role change request for an agent.
    Checks if the agent has sufficient resources (influence points) and meets cooldown requirements.

    Args:
        agent_state (AgentState): The agent's state object
        requested_role (str): The role the agent is requesting to change to

    Returns:
        bool: True if the role change was successful, False otherwise
    """
    # Check if the requested role is valid
    if requested_role not in [ROLE_FACILITATOR, ROLE_INNOVATOR, ROLE_ANALYZER]:
        logger.warning(f"Agent {agent_state.agent_id} requested invalid role: {requested_role}")
        return False

    # Check if it's already the current role
    if requested_role == agent_state.role:
        logger.info(
            f"Agent {agent_state.agent_id} already has role {requested_role}, no change needed."
        )
        return False

    # Check if cooldown period has passed
    if agent_state.steps_in_current_role < agent_state.role_change_cooldown:
        logger.warning(
            f"Agent {agent_state.agent_id} requested role change to {requested_role} but cooldown period not satisfied (needs {agent_state.role_change_cooldown} steps, current: {agent_state.steps_in_current_role})."
        )
        return False

    # Check if the agent has enough IP
    if agent_state.ip < agent_state.role_change_ip_cost:
        logger.warning(
            f"Agent {agent_state.agent_id} requested role change to {requested_role} but had insufficient IP (needed {agent_state.role_change_ip_cost}, had {agent_state.ip})."
        )
        return False

    # All checks passed, proceed with role change
    old_role = agent_state.role

    # Deduct IP cost
    agent_state.ip -= agent_state.role_change_ip_cost

    # Update role
    agent_state.role = requested_role
    agent_state.steps_in_current_role = 0

    # Add to role history
    agent_state.role_history.append((agent_state.last_action_step, requested_role))

    logger.info(
        f"Agent {agent_state.agent_id} changed role from {old_role} to {requested_role}. Spent {agent_state.role_change_ip_cost} IP. Remaining IP: {agent_state.ip}"
    )
    return True


def update_state_node(state: AgentTurnState) -> dict[str, Any]:
    """
    Node for updating the agent's state after action handling.
    This is where passive IP/DU changes and memory consolidation happen.

    Args:
        state (AgentTurnState): Current agent state

    Returns:
        Dict[str, Any]: Updated state
    """
    agent_id = state["agent_id"]
    sim_step = state["simulation_step"]
    action_intent = (
        state.get("structured_output", {}).action_intent
        if state.get("structured_output")
        else "idle"
    )
    message_content = (
        state.get("structured_output", {}).message_content
        if state.get("structured_output")
        else None
    )
    message_recipient_id = (
        state.get("structured_output", {}).message_recipient_id
        if state.get("structured_output")
        else None
    )

    # Get the agent state object
    agent_state = state["state"]

    logger.debug(f"Node 'update_state_node' executing for agent {agent_id} at step {sim_step}")

    # Check if we have a role change request in the structured output
    if state.get("structured_output") and state["structured_output"].requested_role_change:
        # Process the role change request
        requested_role = state["structured_output"].requested_role_change
        logger.info(f"Agent {agent_id} requested role change to: {requested_role}")

        # Call the role change function (it handles validation and IP cost)
        role_change_success = process_role_change(agent_state, requested_role)

        if role_change_success:
            logger.info(f"Agent {agent_id} successfully changed role to {requested_role}")

            # Add a memory about the role change
            agent_state.add_memory(sim_step, "role_change", f"Changed role to {requested_role}")
        else:
            logger.info(
                f"Agent {agent_id} failed to change role to {requested_role} (insufficient IP or cooldown)"
            )

            # Add a memory about the failed role change
            agent_state.add_memory(
                sim_step,
                "resource_constraint",
                f"Attempted to change role to {requested_role} but had insufficient Influence Points or cooldown not satisfied",
            )

    # PASSIVE RESOURCE GENERATION: Generate Data Units based on agent's role
    # Only do this for non-idle actions (to encourage participation)
    generated_du = 0  # Initialize outside the if block
    if action_intent != "idle":
        # Get the role name (e.g., "Facilitator", "Analyzer", "Innovator")
        role_name = agent_state.role

        # Look up the DU generation rate for this role in the config
        du_generation_rate = config.ROLE_DU_GENERATION.get(
            role_name, 1.0
        )  # Default to 1 if not found

        # Generate a float at least 0.5 and at most 1.5 times the base rate
        # This adds some variability to DU generation
        random_factor = 0.5 + random.random()  # Between 0.5 and 1.5
        generated_du = du_generation_rate * random_factor

        # Round to 1 decimal place
        generated_du = round(generated_du, 1)

        # Add the generated DU to the agent's total
        if generated_du > 0:
            previous_du = agent_state.du
            agent_state.du += generated_du
            logger.info(
                f"Agent {agent_id}: Generated {generated_du} DU passively based on role '{role_name}'. DU: {previous_du:.1f} → {agent_state.du:.1f}"
            )

    # MEMORY CONSOLIDATION: First-level hierarchical memory summarization
    # Check if there's enough content in short-term memory to warrant a summary
    if (
        len(agent_state.short_term_memory) >= 3
    ):  # Only summarize if we have at least 3 memory entries
        try:
            # Make sure we have an LLM client available for generating the summary
            if agent_state.llm_client:
                # Extract the most recent memories to summarize (last 10 or all if less)
                recent_memories = list(agent_state.short_term_memory)[-10:]

                # Create a formatted string for DSPy input
                short_term_memory_context = ""
                for memory in recent_memories:
                    # Format depends on the memory type
                    memory_step = memory.get("step", "unknown")
                    memory_type = memory.get("type", "unknown")
                    memory_content = memory.get("content", "No content")

                    # Add formatted memory to the context string
                    short_term_memory_context += f"- Step {memory_step}, {memory_type.replace('_', ' ').title()}: {memory_content}\n"

                # Create an instance of L1SummaryGenerator
                l1_gen = L1SummaryGenerator()

                # Generate the L1 summary using DSPy
                memory_summary = l1_gen.generate_summary(
                    agent_role=agent_state.role,
                    recent_events=short_term_memory_context,
                    current_mood=get_descriptive_mood(agent_state.mood_value)
                    if hasattr(agent_state, "mood_value")
                    else None,
                )

                if memory_summary:
                    # Create metadata for the consolidated memory
                    consolidated_memory = {
                        "step": sim_step,
                        "type": "consolidated_summary",
                        "level": 1,
                        "content": memory_summary,
                        "source": "short_term_memory",
                        "consolidated_entries": len(recent_memories),
                    }

                    # Add the consolidated memory to the agent's memory
                    agent_state.add_memory(sim_step, "consolidated_summary", memory_summary)

                    # If we have a vector store manager, persist this consolidated memory
                    if state.get("vector_store_manager"):
                        try:
                            # Store in vector store for long-term retention and retrieval
                            vector_store = state["vector_store_manager"]
                            vector_store.add_memory(
                                agent_id=agent_id,
                                step=sim_step,
                                event_type="consolidated_summary",
                                content=memory_summary,
                                memory_type="consolidated_summary",
                            )
                            logger.info(
                                f"Agent {agent_id}: Successfully stored consolidated memory in vector store"
                            )
                        except Exception as e:
                            logger.error(
                                f"Agent {agent_id}: Failed to store consolidated memory in vector store: {e}"
                            )

                    logger.info(
                        f"Agent {agent_id}: Generated a level-1 memory consolidation summary at step {sim_step}"
                    )
                    logger.debug(
                        f"Agent {agent_id}: Memory consolidation summary: {memory_summary[:100]}..."
                    )
                else:
                    logger.warning(
                        f"Agent {agent_id}: Failed to generate memory consolidation summary - empty result"
                    )
            else:
                logger.warning(
                    f"Agent {agent_id}: Cannot generate memory consolidation - no LLM client available"
                )
        except Exception as e:
            logger.error(f"Agent {agent_id}: Error during memory consolidation: {e}")

    # Update the state with the potentially modified agent state
    return {
        "state": agent_state,
        "action_intent": action_intent,
        "message_content": message_content,
        "message_recipient_id": message_recipient_id,
    }


def _maybe_prune_l1_memories(state: AgentTurnState) -> dict[str, Any]:
    """
    Checks if it's time to prune Level 1 (daily) summaries based on their age.
    This is called on a more frequent interval than L2 pruning.
    """
    import src.infra.config as config

    agent_id = state["agent_id"]
    sim_step = state["simulation_step"]
    state["state"]
    vector_store = state.get("vector_store_manager")

    # Check if L1 memory pruning is enabled
    if not getattr(config, "MEMORY_PRUNING_ENABLED", False) or not getattr(
        config, "MEMORY_PRUNING_L1_ENABLED", False
    ):
        logger.debug(f"L1 pruning is disabled for agent {agent_id}")
        return state

    # Check if it's time to run L1 pruning based on check interval
    check_interval = getattr(config, "MEMORY_PRUNING_L1_CHECK_INTERVAL_STEPS", 50)
    if sim_step % check_interval != 0:
        return state

    max_age_days = getattr(config, "MEMORY_PRUNING_L1_MAX_AGE_DAYS", 14)
    logger.info(
        f"Agent {agent_id}: Checking for L1 summaries older than {max_age_days} days at step {sim_step}"
    )

    # Find L1 summaries older than the configured threshold
    old_l1_summary_ids = vector_store.get_l1_summaries_older_than(max_age_days)

    if not old_l1_summary_ids:
        logger.info(f"Agent {agent_id}: No old L1 summaries found for pruning")
        return state

    # Log the pruning operation
    logger.info(f"Agent {agent_id}: Pruning {len(old_l1_summary_ids)} old L1 summaries")

    # Delete the old L1 summaries
    success = vector_store.delete_memories_by_ids(old_l1_summary_ids)

    if success:
        logger.info(
            f"Agent {agent_id}: Successfully pruned {len(old_l1_summary_ids)} old L1 summaries"
        )
    else:
        logger.error(f"Agent {agent_id}: Failed to prune old L1 summaries")

    return state


def _maybe_prune_l2_memories(state: AgentTurnState) -> dict[str, Any]:
    """
    Checks if it's time to prune Level 2 (session) summaries based on their age.
    This is called on a less frequent interval than L1 pruning.

    Args:
        state (AgentTurnState): The current agent graph state

    Returns:
        Dict[str, Any]: The updated state after potentially pruning L2 memories
    """
    agent_id = state["agent_id"]
    sim_step = state["simulation_step"]
    state["state"]
    vector_store = state.get("vector_store_manager")

    # Check if L2 memory pruning is enabled and if it's time to check
    if not getattr(config, "MEMORY_PRUNING_ENABLED", False) or not getattr(
        config, "MEMORY_PRUNING_L2_ENABLED", False
    ):
        logger.debug(f"L2 pruning is disabled for agent {agent_id}")
        return state

    # Check if it's time to run L2 pruning based on check interval
    check_interval = getattr(config, "MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS", 100)
    if sim_step % check_interval != 0:
        return state

    max_age_days = getattr(config, "MEMORY_PRUNING_L2_MAX_AGE_DAYS", 30)
    logger.info(
        f"Agent {agent_id}: Checking for L2 summaries older than {max_age_days} days at step {sim_step}"
    )

    # Find L2 summaries older than the configured threshold
    old_l2_summary_ids = vector_store.get_l2_summaries_older_than(max_age_days)

    if not old_l2_summary_ids:
        logger.info(f"Agent {agent_id}: No old L2 summaries found for pruning")
        return state

    # Log the pruning operation
    logger.info(f"Agent {agent_id}: Pruning {len(old_l2_summary_ids)} old L2 summaries")

    # Delete the old L2 summaries
    success = vector_store.delete_memories_by_ids(old_l2_summary_ids)

    if success:
        logger.info(
            f"Agent {agent_id}: Successfully pruned {len(old_l2_summary_ids)} old L2 summaries"
        )
    else:
        logger.error(f"Agent {agent_id}: Failed to prune old L2 summaries")

    return state


async def finalize_message_agent_node(state: AgentTurnState) -> dict[str, Any]:
    """
    Asynchronously finalize the message preparation and handle relationship updates.
    Uses AsyncDSPyManager for non-blocking DSPy calls. On timeout or error, returns a failsafe output and logs the issue.
    Must be awaited in the graph.
    """
    agent = state.get("agent_instance")
    agent_id = state["agent_id"]
    state["simulation_step"]
    message_content = state.get("message_content")
    message_recipient_id = state.get("message_recipient_id")
    action_intent = state.get("action_intent", "idle")
    final_agent_state = state.get("state")
    knowledge_board = state.get("knowledge_board")
    logger.debug(f"Node 'finalize_message_agent_node' executing for agent {agent_id}")
    if action_intent == "propose_idea" and message_content and knowledge_board:
        # ... existing code for propose_idea intent ...
        pass
    elif message_content:
        is_targeted = message_recipient_id is not None
        outgoing_sentiment = analyze_sentiment(message_content)
        if (
            is_targeted
            and outgoing_sentiment
            and message_recipient_id != agent_id
            and agent is not None
        ):
            logger.debug(
                f"Updating relationship for sender {agent_id} with target {message_recipient_id} based on outgoing message sentiment: {outgoing_sentiment}"
            )
            try:
                current_score = final_agent_state.relationships.get(message_recipient_id, 0.0)
                agent1_persona = getattr(final_agent_state, "role", "Unknown")
                agent2_persona = "Unknown"
                if (
                    hasattr(final_agent_state, "all_agents")
                    and message_recipient_id in final_agent_state.all_agents
                ):
                    agent2_persona = getattr(
                        final_agent_state.all_agents[message_recipient_id], "role", "Unknown"
                    )
                interaction_summary = message_content or ""
                dspy_result = await agent.async_update_relationship(
                    current_score,
                    interaction_summary,
                    agent1_persona,
                    agent2_persona,
                    outgoing_sentiment,
                )
                new_score = float(getattr(dspy_result, "new_relationship_score", current_score))
                rationale = getattr(dspy_result, "relationship_change_rationale", None)
                min_score = getattr(final_agent_state, "min_relationship_score", -1.0)
                max_score = getattr(final_agent_state, "max_relationship_score", 1.0)
                new_score = max(min_score, min(max_score, new_score))
                final_agent_state.relationships[message_recipient_id] = new_score
                logger.info(
                    f"[DSPy] Updated relationship ({agent_id}->{message_recipient_id}): {current_score:.2f} -> {new_score:.2f}. Rationale: {rationale}"
                )
            except Exception as e:
                logger.error(f"DSPy RelationshipUpdater failed, falling back to rule-based: {e}")
        return_values = {
            "message_content": message_content,
            "message_recipient_id": message_recipient_id,
            "action_intent": action_intent,
            "updated_agent_state": final_agent_state,
            "is_targeted": is_targeted,
        }
    else:
        logger.debug(f"Agent {agent_id} has no message to send.")
        return_values = {
            "message_content": None,
            "message_recipient_id": None,
            "action_intent": action_intent,
            "updated_agent_state": final_agent_state,
        }
    logger.debug(
        f"FINALIZE_RETURN :: Agent {agent_id}: Returning final state with updated_agent_state included"
    )
    return return_values


def shorten_message(message: str) -> str:
    """Utility function to shorten a message, making it more terse or formal."""
    # Split the message into sentences
    sentences = re.split(r"(?<=[.!?])\s+", message)

    # If we have multiple sentences, just keep the main one(s)
    if len(sentences) > 2:
        # Keep first and maybe last sentence
        return sentences[0] + " " + sentences[-1]
    elif len(sentences) == 2:
        # Keep just the first sentence
        return sentences[0]

    # If it's just one sentence, remove filler words and niceties
    shortened = message
    filler_phrases = [
        "I think that",
        "I believe that",
        "In my opinion",
        "I would like to",
        "It seems to me",
        "If I may say so",
        "I'm happy to",
        "I'm pleased to",
        "I would suggest",
        "Thank you for",
        "I appreciate",
        "If possible",
        "When you get a chance",
        "If you don't mind",
    ]

    for phrase in filler_phrases:
        shortened = shortened.replace(phrase, "")

    # Remove double spaces from removals
    shortened = re.sub(r"\s+", " ", shortened).strip()

    return shortened


def route_broadcast_decision(state: AgentTurnState) -> str:
    """
    Determines whether to broadcast a message or exit the graph.
    The decision is based on the agent's mood and whether a message was generated.
    """
    agent_id = state.get("agent_id")
    updated_state = state.get("updated_state", {})
    current_mood = updated_state.get("mood", "neutral")
    descriptive_mood = updated_state.get("descriptive_mood", "neutral")
    structured_output = state.get("structured_output")
    has_message = structured_output and structured_output.message_content is not None

    logger.debug(
        f"Agent {agent_id} in mood '{current_mood}' (descriptive: '{descriptive_mood}') deciding on broadcasting message..."
    )

    # Only broadcast if:
    # 1. There is a message to broadcast (structured_output.broadcast is not None)
    # 2. And the agent is not unhappy or in a negative/very_negative descriptive mood
    if (
        has_message
        and current_mood != "unhappy"
        and descriptive_mood not in ["negative", "very_negative"]
    ):
        return "broadcast"
    else:
        if has_message:
            logger.info(
                f"Agent {agent_id} suppressing message due to mood: '{current_mood}', descriptive mood: '{descriptive_mood}'"
            )
        return "exit"  # No message to broadcast or agent is unhappy


def route_relationship_context(state: AgentTurnState) -> str:
    """
    Routes to relationship prompt modifier generation based on whether the agent has any
    relationships established yet.
    """
    agent_id = state.get("agent_id")
    current_agent_state = state.get("state")
    relationships = current_agent_state.relationships

    logger.debug(f"Agent {agent_id} relationship router checking for relationships...")
    if relationships:
        logger.debug(f"Agent {agent_id} has {len(relationships)} relationships established.")
        return "has_relationships"
    else:
        logger.debug(f"Agent {agent_id} has no relationships established yet.")
        return "no_relationships"


def handle_propose_idea_node(state: AgentTurnState) -> dict[str, Any]:
    """
    Handles the 'propose_idea' intent by extracting the proposed idea
    and adding it to the Knowledge Board if the idea is valid.
    Also awards IP for successful Knowledge Board posts.
    """
    agent_id = state["agent_id"]
    sim_step = state["simulation_step"]
    structured_output = state.get("structured_output")
    knowledge_board = state.get("knowledge_board")

    # Initialize the variable for idea content
    idea_content = None

    if structured_output and structured_output.message_content:
        idea_content = structured_output.message_content

    # Default return is the unchanged state
    ret_state = {**state}

    # Track if we should award IP (only if idea is valid and successfully posted)
    award_ip = False

    # Access agent's persistent state to get current IP count and Data Units
    agent_persistent_state = state.get("state")
    current_ip = agent_persistent_state.ip
    current_du = agent_persistent_state.du

    # Log the current IP and DU amounts for debugging
    logging.debug(
        f"Agent {agent_id} current IP: {current_ip}, current DU: {current_du}, attempting to propose idea"
    )

    if idea_content:
        # First check if agent has enough DU to post to the Knowledge Board
        if current_du >= PROPOSE_DETAILED_IDEA_DU_COST:
            # Deduct the DU cost for the detailed idea
            agent_persistent_state.du -= PROPOSE_DETAILED_IDEA_DU_COST
            updated_du = agent_persistent_state.du
            logging.info(
                f"Agent {agent_id} spent {PROPOSE_DETAILED_IDEA_DU_COST} DU to post a detailed idea. Remaining DU: {updated_du}."
            )

            # Now check if agent has enough IP to post to the Knowledge Board
            if current_ip >= IP_COST_TO_POST_IDEA:
                # Deduct the cost to post the idea
                agent_persistent_state.ip -= IP_COST_TO_POST_IDEA
                updated_ip = agent_persistent_state.ip
                logging.info(
                    f"Agent {agent_id} spent {IP_COST_TO_POST_IDEA} IP to post an idea. Remaining IP: {updated_ip}."
                )

                # Add the proposed idea to the Knowledge Board
                if knowledge_board:
                    entry = f"{idea_content}"
                    knowledge_board.add_entry(entry, agent_id, sim_step)
                    logging.info(
                        f"KnowledgeBoard: Added entry from Agent {agent_id} at step {sim_step}"
                    )
                    award_ip = True

                # If the idea was successfully posted, award IP
                if award_ip:
                    # Award the IP for a successful proposal
                    agent_persistent_state.ip += IP_AWARD_FOR_PROPOSAL
                    final_ip = agent_persistent_state.ip
                    logging.info(
                        f"Agent {agent_id} earned {IP_AWARD_FOR_PROPOSAL} IP for proposing an idea. New IP: {final_ip}."
                    )
            else:
                # Agent doesn't have enough IP to post to the Knowledge Board
                logging.warning(
                    f"Agent {agent_id} attempted to post idea '{idea_content[:30]}...' but had insufficient IP ({current_ip} IP) for the cost of {IP_COST_TO_POST_IDEA} IP. Idea not posted."
                )

                # Refund the DU since the idea wasn't posted
                agent_persistent_state.du += PROPOSE_DETAILED_IDEA_DU_COST
                logging.info(
                    f"Agent {agent_id} was refunded {PROPOSE_DETAILED_IDEA_DU_COST} DU since the idea wasn't posted due to insufficient IP."
                )
        else:
            # Agent doesn't have enough DU to post the detailed idea
            logging.warning(
                f"Agent {agent_id} attempted to post idea '{idea_content[:30]}...' but had insufficient DU ({current_du} DU) for the cost of {PROPOSE_DETAILED_IDEA_DU_COST} DU. Idea not posted."
            )

    # Store the proposed idea content and updated agent state in the return state
    ret_state["proposed_idea_content"] = idea_content
    ret_state["state"] = agent_persistent_state

    return ret_state


# Add formatting helper functions
def _format_other_agents(
    other_agents_info: list[dict[str, Any]], relationships: dict[str, float]
) -> str:
    """Helper function to format other agents information for the prompt."""
    from src.infra.config import get_relationship_label

    if not other_agents_info:
        return "  You are currently alone."

    lines = []
    lines.append(
        "  Other agents you can interact with (use their ID when sending targeted messages):"
    )
    for agent_info in other_agents_info:
        other_id = agent_info.get("agent_id", "unknown")
        other_name = agent_info.get("name", other_id[:8])
        other_mood = agent_info.get("mood", "unknown")
        agent_info.get("descriptive_mood", "unknown")

        # Get relationship score and label
        relationship_score = relationships.get(other_id, 0.0)
        relationship_label = get_relationship_label(relationship_score)

        # Include the full agent_id for clear targeting
        lines.append(
            f"  - {other_name} (Agent ID: '{other_id}', Mood: {other_mood}, Relationship: {relationship_label} ({relationship_score:.1f}))"
        )

    return "\n".join(lines)


def _format_knowledge_board(board_entries: list[str]) -> str:
    """Helper function to format knowledge board entries for the prompt."""
    if not board_entries:
        return "  (Board is currently empty)"

    lines = []
    lines.append(
        "  You can reference a board entry by its Step and original Agent ID (e.g., 'Regarding Step 3's idea by Agent_XYZ...')."
    )
    for i, entry in enumerate(board_entries):
        lines.append(f"  - {entry}")

    return "\n".join(lines)


def _format_messages(messages: list[dict[str, Any]]) -> str:
    """Helper function to format perceived messages for the prompt."""
    if not messages:
        return "  No messages were perceived in the previous step."

    lines = []
    for msg in messages:
        sender = msg.get("sender_id", "unknown")
        content = msg.get("content", "")
        recipient = msg.get("recipient_id")
        message_type = "(Private to you)" if recipient else "(Broadcast)"
        lines.append("  - " + sender + " " + message_type + ': "' + content + '"')

    return "\n".join(lines)


def get_mood_level(mood_value: float) -> str:
    """
    Gets a textual mood level based on a numerical mood value.

    Args:
        mood_value: Float value between -1.0 and 1.0 representing mood

    Returns:
        A string representing the mood level
    """
    if mood_value < -0.3:
        return "unhappy"
    elif mood_value > 0.3:
        return "happy"
    else:
        return "neutral"


def get_descriptive_mood(mood_value: float) -> str:
    """
    Gets a more detailed descriptive mood based on a numerical mood value.

    Args:
        mood_value: Float value between -1.0 and 1.0 representing mood

    Returns:
        A string with a more detailed mood description
    """
    if mood_value < -0.7:
        return "very unhappy"
    elif mood_value < -0.3:
        return "unhappy"
    elif mood_value < -0.1:
        return "slightly unhappy"
    elif mood_value <= 0.1:
        return "neutral"
    elif mood_value <= 0.3:
        return "slightly happy"
    elif mood_value <= 0.7:
        return "happy"
    else:
        return "very happy"


# Add these new handler functions after the other handler functions


def handle_continue_collaboration_node(state: AgentTurnState) -> dict[str, Any]:
    """
    Handles the 'continue_collaboration' intent.
    Currently a placeholder for future functionality.
    """
    agent_id = state.get("agent_id", "UNKNOWN_HANDLER")
    logger.info(
        f"Agent {agent_id}: Executing 'continue_collaboration' intent handler (currently placeholder)."
    )
    # Potentially add minor relationship boosts or mood adjustments here later if desired
    return state


def handle_idle_node(state: AgentTurnState) -> dict[str, Any]:
    """
    Handles the 'idle' intent.
    Currently a placeholder for future functionality.
    """
    agent_id = state.get("agent_id", "UNKNOWN_HANDLER")
    logger.info(f"Agent {agent_id}: Executing 'idle' intent handler (currently placeholder).")
    # Potentially add slight negative mood decay or relationship decay here later if desired
    return state


def handle_ask_clarification_node(state: AgentTurnState) -> dict[str, Any]:
    """
    Handles the 'ask_clarification' intent.
    Simple clarifications are free, but detailed ones cost DU.
    Checks if the agent has sufficient resources before allowing a detailed clarification.
    """
    agent_id = state.get("agent_id", "UNKNOWN_HANDLER")
    structured_output = state.get("structured_output")
    agent_persistent_state = state.get("state")

    # Only process if there's actual message content
    if structured_output and structured_output.message_content:
        message_content = structured_output.message_content
        original_message = message_content

        # Determine if this is a "detailed" clarification request that should cost DU
        is_detailed = False

        # Check based on length (more than 100 characters is considered detailed)
        if len(message_content) > 100:
            is_detailed = True
            logger.debug(
                f"Agent {agent_id} clarification considered detailed due to length: {len(message_content)} chars"
            )

        # Check for multiple question marks (indicating multiple questions)
        if message_content.count("?") > 1:
            is_detailed = True
            logger.debug(
                f"Agent {agent_id} clarification considered detailed due to multiple questions: {message_content.count('?')} questions"
            )

        # Check for keywords indicating a detailed request
        detailed_keywords = [
            "elaborate",
            "explain in detail",
            "comprehensive",
            "specific details",
            "thorough explanation",
            "step by step",
            "multiple aspects",
            "breakdown",
        ]

        for keyword in detailed_keywords:
            if keyword.lower() in message_content.lower():
                is_detailed = True
                logger.debug(
                    f"Agent {agent_id} clarification considered detailed due to keyword: '{keyword}'"
                )
                break

        # If it's a detailed clarification, check and deduct DU
        if is_detailed:
            current_du = agent_persistent_state.du

            if current_du >= DU_COST_REQUEST_DETAILED_CLARIFICATION:
                # Deduct the DU cost
                new_du = current_du - DU_COST_REQUEST_DETAILED_CLARIFICATION
                agent_persistent_state.du = new_du

                # Log the DU deduction
                logger.info(
                    f"Agent {agent_id} spent {DU_COST_REQUEST_DETAILED_CLARIFICATION} DU for a detailed clarification request. Remaining DU: {new_du}."
                )

                # Store the clarification question in agent state
                agent_persistent_state.last_clarification_question = message_content
            else:
                # Not enough DU for a detailed clarification
                logger.warning(
                    f"Agent {agent_id} attempted a detailed clarification but had insufficient DU ({current_du} < {DU_COST_REQUEST_DETAILED_CLARIFICATION}). Request blocked."
                )

                # Modify the message content to indicate that the request was blocked
                simplified_message = "I wanted to ask a detailed question, but I don't have enough data units (DU) at the moment. Could you provide some basic information instead?"

                # Update the message in the structured output
                structured_output.message_content = simplified_message
                state["structured_output"] = structured_output

                # Store the simplified question
                agent_persistent_state.last_clarification_question = simplified_message
                agent_persistent_state.last_clarification_downgraded = True

                # Add a memory entry about the failed attempt
                memory_content = f"Attempted to ask a detailed clarification: '{original_message[:50]}...' but had insufficient DU ({current_du}/{DU_COST_REQUEST_DETAILED_CLARIFICATION})"
                agent_persistent_state.add_memory(
                    state.get("simulation_step", 0), "resource_constraint", memory_content
                )
        else:
            # Simple clarification, no DU cost
            logger.debug(f"Agent {agent_id} issued a simple clarification request (no DU cost).")

            # Store the clarification question in agent state
            agent_persistent_state.last_clarification_question = message_content

    # Update the state with the potentially modified agent state
    state["state"] = agent_persistent_state
    return state


def handle_deep_analysis_node(state: AgentTurnState) -> dict[str, Any]:
    """
    Handles the 'perform_deep_analysis' intent.
    Deducts the DU cost for performing a deep analysis, especially for Analyzers.
    """
    agent_id = state.get("agent_id", "UNKNOWN_HANDLER")
    agent_persistent_state = state.get("state")
    current_role = agent_persistent_state.role
    current_du = agent_persistent_state.du

    # Check if the agent is an Analyzer (they should get the most benefit from this action)
    is_analyzer = current_role == ROLE_ANALYZER

    # If not an analyzer, log a warning but still allow it (with the same cost for now)
    if not is_analyzer:
        logger.warning(
            f"Agent {agent_id} (Role: {current_role}) is attempting a deep analysis, which is more suited for Analyzers."
        )

    # Check if the agent has enough DU for the deep analysis
    if current_du >= DU_COST_DEEP_ANALYSIS:
        # Deduct the DU cost
        new_du = current_du - DU_COST_DEEP_ANALYSIS
        agent_persistent_state.du = new_du

        # Log the DU deduction
        logger.info(
            f"Agent {agent_id} (Role: {current_role}) spent {DU_COST_DEEP_ANALYSIS} DU to perform deep analysis. Remaining DU: {new_du}."
        )

        # Note: The outcome of this analysis might lead to DU earning through the
        # "successful analysis" or "constructive reference" mechanisms in update_state_node
    else:
        # Not enough DU to perform the deep analysis
        logger.warning(
            f"Agent {agent_id} attempted deep analysis but had insufficient DU ({current_du} < {DU_COST_DEEP_ANALYSIS}). Action may be less effective or not fully registered."
        )
        # We don't modify the agent state here as they don't have enough DU to spend

    # Update the state with the potentially modified agent state
    state["state"] = agent_persistent_state
    return state


def handle_create_project_node(state: AgentTurnState) -> dict[str, Any]:
    """
    Handles the 'create_project' intent.
    Allows an agent to create a new project, with a cost in both IP and DU.
    """
    from src.infra import config

    agent_id = state.get("agent_id", "UNKNOWN_HANDLER")
    structured_output = state.get("structured_output")
    agent_persistent_state = state.get("state")

    # Extract the project name and description to create
    project_name = structured_output.project_name_to_create if structured_output else None
    project_description = (
        structured_output.project_description_for_creation if structured_output else None
    )

    if not project_name:
        logger.warning(
            f"Agent {agent_id} attempted to create a project but didn't specify a project name"
        )
        return state

    # Check if the agent has enough IP and DU to create a project
    current_ip = agent_persistent_state.ip
    current_du = agent_persistent_state.du

    if current_ip < config.IP_COST_CREATE_PROJECT:
        logger.warning(
            f"Agent {agent_id} attempted to create project '{project_name}' but had insufficient IP ({current_ip} < {config.IP_COST_CREATE_PROJECT})"
        )
        return state

    if current_du < config.DU_COST_CREATE_PROJECT:
        logger.warning(
            f"Agent {agent_id} attempted to create project '{project_name}' but had insufficient DU ({current_du} < {config.DU_COST_CREATE_PROJECT})"
        )
        return state

    # Get the simulation from the environment perception
    # This requires the simulation instance to be passed via run_turn to environment_perception
    simulation = state.get("environment_perception", {}).get("simulation")

    if not simulation:
        logger.error(
            f"Agent {agent_id} couldn't create project '{project_name}' because simulation reference is missing"
        )
        return state

    # Create the project
    project_id = simulation.create_project(project_name, agent_id, project_description)

    if project_id:
        # Project created successfully, deduct costs
        agent_persistent_state.ip -= config.IP_COST_CREATE_PROJECT
        agent_persistent_state.du -= config.DU_COST_CREATE_PROJECT

        # Set the agent's current project affiliation
        agent_persistent_state.current_project_id = project_id
        agent_persistent_state.current_project_affiliation = (
            project_name  # Use project NAME, not ID
        )

        # Update project history
        agent_persistent_state.project_history.append(
            (state.get("simulation_step", 0), project_id)
        )

        desc_info = f" with description: '{project_description}'" if project_description else ""
        logger.info(
            f"Agent {agent_id} created project '{project_name}'{desc_info} (ID: {project_id}) at a cost of {config.IP_COST_CREATE_PROJECT} IP and {config.DU_COST_CREATE_PROJECT} DU"
        )
    else:
        logger.warning(f"Agent {agent_id} failed to create project '{project_name}'")

    # Update the state with the potentially modified agent state
    state["state"] = agent_persistent_state
    return state


def handle_join_project_node(state: AgentTurnState) -> dict[str, Any]:
    """
    Handles the 'join_project' intent.
    Allows an agent to join an existing project, with a cost in both IP and DU.
    """
    from src.infra import config

    agent_id = state.get("agent_id", "UNKNOWN_HANDLER")
    structured_output = state.get("structured_output")
    agent_persistent_state = state.get("state")

    # Extract the project ID to join
    project_id = structured_output.project_id_to_join_or_leave if structured_output else None

    if not project_id:
        logger.warning(
            f"Agent {agent_id} attempted to join a project but didn't specify a project ID"
        )
        return state

    # Check if the agent is already in a project
    current_project = agent_persistent_state.current_project_id
    if current_project:
        logger.warning(
            f"Agent {agent_id} attempted to join project '{project_id}' but is already a member of project '{current_project}'"
        )
        return state

    # Check if the agent has enough IP and DU to join a project
    current_ip = agent_persistent_state.ip
    current_du = agent_persistent_state.du

    if current_ip < config.IP_COST_JOIN_PROJECT:
        logger.warning(
            f"Agent {agent_id} attempted to join project '{project_id}' but had insufficient IP ({current_ip} < {config.IP_COST_JOIN_PROJECT})"
        )
        return state

    if current_du < config.DU_COST_JOIN_PROJECT:
        logger.warning(
            f"Agent {agent_id} attempted to join project '{project_id}' but had insufficient DU ({current_du} < {config.DU_COST_JOIN_PROJECT})"
        )
        return state

    # Get the simulation from the environment perception
    simulation = state.get("environment_perception", {}).get("simulation")

    if not simulation:
        logger.error(
            f"Agent {agent_id} couldn't join project '{project_id}' because simulation reference is missing"
        )
        return state

    # Get project name first in case joining fails
    project_name = simulation.projects.get(project_id, {}).get("name", "Unknown Project")

    # Attempt to join the project
    success = simulation.join_project(project_id, agent_id)

    if success:
        # Project joined successfully, deduct costs
        agent_persistent_state.ip -= config.IP_COST_JOIN_PROJECT
        agent_persistent_state.du -= config.DU_COST_JOIN_PROJECT

        # Set the agent's current project affiliation
        agent_persistent_state.current_project_id = project_id
        agent_persistent_state.current_project_affiliation = (
            project_name  # Use project NAME, not ID
        )

        # Update project history
        agent_persistent_state.project_history.append(
            (state.get("simulation_step", 0), project_id)
        )

        logger.info(
            f"Agent {agent_id} joined project '{project_name}' (ID: {project_id}) at a cost of {config.IP_COST_JOIN_PROJECT} IP and {config.DU_COST_JOIN_PROJECT} DU"
        )
    else:
        logger.warning(f"Agent {agent_id} failed to join project with ID '{project_id}'")

    # Update the state with the potentially modified agent state
    state["state"] = agent_persistent_state
    return state


def handle_leave_project_node(state: AgentTurnState) -> dict[str, Any]:
    """
    Handles the 'leave_project' intent.
    Allows an agent to leave a project they are currently a member of.
    """
    agent_id = state.get("agent_id", "UNKNOWN_HANDLER")
    structured_output = state.get("structured_output")
    agent_persistent_state = state.get("state")

    # Extract the project ID to leave
    project_id = structured_output.project_id_to_join_or_leave if structured_output else None

    # If no project ID specified, use the agent's current project
    if not project_id:
        project_id = agent_persistent_state.current_project_id
        if not project_id:
            logger.warning(
                f"Agent {agent_id} attempted to leave a project but is not a member of any project"
            )
            return state

    # Get the simulation from the environment perception
    simulation = state.get("environment_perception", {}).get("simulation")

    if not simulation:
        logger.error(
            f"Agent {agent_id} couldn't leave project '{project_id}' because simulation reference is missing"
        )
        return state

    # Get project details for logging before leaving
    project_name = simulation.projects.get(project_id, {}).get("name", "Unknown Project")

    # Attempt to leave the project
    success = simulation.leave_project(project_id, agent_id)

    if success:
        # Clear the agent's current project affiliation in both fields to keep them in sync
        agent_persistent_state.current_project_id = None
        agent_persistent_state.current_project_affiliation = None

        # Update project history
        agent_persistent_state.project_history.append((state.get("simulation_step", 0), None))

        logger.info(f"Agent {agent_id} left project '{project_name}' (ID: {project_id})")
    else:
        logger.warning(f"Agent {agent_id} failed to leave project with ID '{project_id}'")

    # Update the state with the potentially modified agent state
    state["state"] = agent_persistent_state
    return state


def handle_send_direct_message_node(state: AgentTurnState) -> dict[str, Any]:
    """
    Handles the 'send_direct_message' intent.
    Processes a targeted message to a specific agent.
    """
    agent_id = state.get("agent_id", "UNKNOWN_HANDLER")
    structured_output = state.get("structured_output")

    # Extract message details
    message_content = structured_output.message_content if structured_output else None
    message_recipient_id = structured_output.message_recipient_id if structured_output else None

    if not message_recipient_id:
        logger.warning(
            f"Agent {agent_id} attempted to send direct message but didn't specify a recipient"
        )
        # Fallback to a broadcast message
        return state

    logger.info(
        f"Agent {agent_id} sending direct message to {message_recipient_id}: '{message_content[:30]}...'"
    )

    # Set flag to ensure this is processed as a targeted message
    state["is_targeted"] = True

    return state


# Update the route_action_intent function to handle all intents
def route_action_intent(state: AgentTurnState) -> str:
    """
    Determines the next node based on the agent's action_intent.
    """
    structured_output: Optional[AgentActionOutput] = state.get("structured_output")
    intent = structured_output.action_intent if structured_output else "idle"

    logger.debug(f"Agent {state['agent_id']}: Routing based on action_intent '{intent}'...")

    if intent == "propose_idea":
        return "handle_propose_idea"
    elif intent == "ask_clarification":
        return "handle_ask_clarification"
    elif intent == "continue_collaboration":
        return "handle_continue_collaboration"
    elif intent == "idle":
        return "handle_idle"
    elif intent == "perform_deep_analysis":
        return "handle_deep_analysis"
    elif intent == "create_project":
        return "handle_create_project"
    elif intent == "join_project":
        return "handle_join_project"
    elif intent == "leave_project":
        return "handle_leave_project"
    elif intent == "send_direct_message":
        return "handle_send_direct_message"
    else:
        # For unknown intents, go to update_state
        return "update_state"


# --- Graph Definition ---


def _maybe_consolidate_memories(state: AgentTurnState) -> dict[str, Any]:
    """
    Checks if it's time to consolidate L1 summaries into L2 summaries at the current step.
    This node generates Level 2 (chapter) summaries from Level 1 summaries after a configured
    number of steps have passed since the last L2 consolidation.

    Args:
        state (AgentTurnState): The current agent graph state

    Returns:
        Dict[str, Any]: The updated state after potentially consolidating memories
    """
    agent_id = state["agent_id"]
    sim_step = state["simulation_step"]
    agent_state = state["state"]
    vector_store = state.get("vector_store_manager")

    # Define the L2 consolidation interval (every 10 steps by default)
    L2_CONSOLIDATION_INTERVAL = 10

    # Check if we're at a step where L2 consolidation should happen
    if sim_step % L2_CONSOLIDATION_INTERVAL != 0 or sim_step == 0:
        return state

    # Check if we have access to the vector store for retrieving L1 summaries
    if not vector_store:
        logger.warning(
            f"Agent {agent_id}: Cannot perform L2 memory consolidation - no vector store available"
        )
        return state

    # Since we're at a consolidation step, we'll create a Level 2 summary from
    # all Level 1 summaries since the last consolidation

    # Get the last Level 2 consolidation step (init to 0 if not set)
    last_l2_step = getattr(agent_state, "last_level_2_consolidation_step", 0)

    logger.info(
        f"Agent {agent_id}: Performing L2 memory consolidation at step {sim_step}. Last L2 consolidation was at step {last_l2_step}"
    )

    # Retrieve all Level 1 summaries since the last L2 consolidation
    f"recent consolidated summaries for agent {agent_id}"

    # Define the step range for the L1 summaries we want
    # We want summaries after the last L2 consolidation up to the current step
    step_filter = {"$gt": last_l2_step, "$lte": sim_step}

    # Get the relevant L1 summaries from the vector store
    l1_summaries = vector_store.retrieve_filtered_memories(
        agent_id=agent_id,
        filters={"memory_type": "consolidated_summary", "step": step_filter},
        limit=50,  # Get all summaries in the range, up to 50
    )

    # If we don't have any L1 summaries to consolidate, return early
    if not l1_summaries or len(l1_summaries) == 0:
        logger.info(f"Agent {agent_id}: No Level 1 summaries found for L2 consolidation")
        return state

    logger.info(
        f"Agent {agent_id}: Found {len(l1_summaries)} Level 1 summaries to consolidate into a Level 2 summary"
    )

    # Prepare the context string from the L1 summaries
    l1_summaries_context = ""

    # Sort the summaries by step for chronological order
    sorted_summaries = sorted(l1_summaries, key=lambda x: x.get("step", 0))

    for summary in sorted_summaries:
        step = summary.get("step", "unknown")
        content = summary.get("content", "No content")
        l1_summaries_context += f"- Step {step}, Consolidated Summary: {content}\n"

    # Prepare optional inputs for the L2SummaryGenerator

    # Get descriptive mood if available
    current_mood = (
        get_descriptive_mood(agent_state.mood_value)
        if hasattr(agent_state, "mood_value")
        else None
    )

    # Analyze mood trend (simple approach for now)
    overall_mood_trend = None
    if hasattr(agent_state, "mood_history") and len(agent_state.mood_history) >= 2:
        recent_moods = [mood for step, mood in agent_state.mood_history[-5:]]
        # Simple trend detection
        if all(mood == recent_moods[0] for mood in recent_moods):
            overall_mood_trend = f"Consistently {recent_moods[0]}"
        elif recent_moods[-1] > recent_moods[0]:
            overall_mood_trend = f"Improving from {recent_moods[0]} to {recent_moods[-1]}"
        elif recent_moods[-1] < recent_moods[0]:
            overall_mood_trend = f"Declining from {recent_moods[0]} to {recent_moods[-1]}"
        else:
            overall_mood_trend = f"Fluctuating with recent {recent_moods[-1]}"

    # Format agent goals
    agent_goals = None
    if hasattr(agent_state, "goals") and agent_state.goals:
        agent_goals = "Current goals: " + ", ".join(
            [goal.get("description", "") for goal in agent_state.goals if "description" in goal]
        )

    # Create an instance of L2SummaryGenerator
    l2_gen = L2SummaryGenerator()

    # Generate the L2 summary using DSPy
    try:
        l2_summary_text = l2_gen.generate_summary(
            agent_role=agent_state.role,
            l1_summaries_context=l1_summaries_context,
            overall_mood_trend=overall_mood_trend,
            agent_goals=agent_goals,
        )

        if not l2_summary_text:
            # If DSPy generation failed, fallback to direct LLM call (if available)
            if agent_state.llm_client:
                logger.warning(
                    f"Agent {agent_id}: DSPy L2 summary generation failed, falling back to direct LLM"
                )

                # Build a prompt for the LLM
                prompt = f"""You are helping generate a Level 2 (L2) memory summary for an agent with role {agent_state.role}.
                
                Based on the following series of Level 1 summaries, create a comprehensive yet concise Level 2 summary that captures the key insights, themes, and developments across this period.
                
                Level 1 Summaries:
                {l1_summaries_context}
                
                """

                if overall_mood_trend:
                    prompt += f"Overall mood trend during this period: {overall_mood_trend}\n\n"

                if agent_goals:
                    prompt += f"Agent's goals: {agent_goals}\n\n"

                prompt += "Generate a comprehensive L2 summary that synthesizes these experiences into meaningful insights:"

                # Call the LLM directly
                l2_summary_text = generate_text(prompt, model=config.LLM_NAME_SHORT)
            else:
                logger.error(
                    f"Agent {agent_id}: Cannot generate L2 summary - DSPy failed and no LLM client available"
                )
                return state

    except Exception as e:
        logger.error(f"Agent {agent_id}: Failed to generate L2 summary: {e}")
        return state

    # If we have a valid L2 summary, store it
    if l2_summary_text:
        # Create metadata for the L2 summary
        l2_summary = {
            "step": sim_step,
            "type": "chapter_summary",  # This is a Level 2 summary
            "level": 2,
            "content": l2_summary_text,
            "source": "l1_summaries_consolidation",
            "consolidated_entries": len(l1_summaries),
            "consolidation_period": f"{last_l2_step + 1}-{sim_step}",
        }

        # Add the L2 summary to the agent's memory
        agent_state.add_memory(sim_step, "chapter_summary", l2_summary_text)

        # Update the last L2 consolidation step
        agent_state.last_level_2_consolidation_step = sim_step

        # Store the L2 summary in the vector store
        try:
            vector_store.add_memory(
                agent_id=agent_id,
                step=sim_step,
                event_type="chapter_summary",  # Type for filtering
                content=l2_summary_text,
                memory_type="chapter_summary",  # Type for filtering
                level=2,
                consolidation_period=f"{last_l2_step + 1}-{sim_step}",
                consolidated_entries=len(l1_summaries),
            )
            logger.info(
                f"Agent {agent_id}: Successfully stored Level 2 chapter summary in vector store"
            )
        except Exception as e:
            logger.error(
                f"Agent {agent_id}: Failed to store Level 2 chapter summary in vector store: {e}"
            )

        logger.info(
            f"Agent {agent_id}: Generated a Level 2 chapter summary at step {sim_step}, consolidating {len(l1_summaries)} Level 1 summaries"
        )
        logger.debug(f"Agent {agent_id}: Chapter summary content: {l2_summary_text[:150]}...")
    else:
        logger.warning(
            f"Agent {agent_id}: Failed to generate Level 2 chapter summary - empty result"
        )

    return {"state": agent_state}


def _maybe_prune_l1_memories_mus(state: AgentTurnState) -> dict[str, Any]:
    """
    Checks if it's time to prune Level 1 (consolidated) memories using MUS-based pruning.
    Only runs if enabled and at the configured interval.
    """
    import src.infra.config as config

    agent_id = state["agent_id"]
    sim_step = state["simulation_step"]
    vector_store = state.get("vector_store_manager")

    if not getattr(config, "MEMORY_PRUNING_L1_MUS_ENABLED", False):
        return state

    check_interval = getattr(config, "MEMORY_PRUNING_L1_MUS_CHECK_INTERVAL_STEPS", 50)
    if sim_step % check_interval != 0:
        return state

    logger.info(f"Agent {agent_id}: Performing MUS-based L1 pruning check at step {sim_step}")

    mus_threshold = getattr(config, "MEMORY_PRUNING_L1_MUS_THRESHOLD", 0.3)
    min_age_days = getattr(config, "MEMORY_PRUNING_L1_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION", 7)

    if not vector_store:
        logger.warning(
            f"Agent {agent_id}: No vector store manager available for MUS-based L1 pruning."
        )
        return state

    ids_to_prune = vector_store.get_l1_memories_for_mus_pruning(mus_threshold, min_age_days)

    if ids_to_prune:
        logger.info(
            f"Agent {agent_id}: MUS-based L1 pruning: deleting {len(ids_to_prune)} L1 summaries (sample: {ids_to_prune[:3]})"
        )
        success = vector_store.delete_memories_by_ids(ids_to_prune)
        if success:
            logger.info(
                f"Agent {agent_id}: Successfully pruned {len(ids_to_prune)} L1 summaries by MUS."
            )
        else:
            logger.error(f"Agent {agent_id}: Failed to prune L1 summaries by MUS.")
    else:
        logger.info(
            f"Agent {agent_id}: No L1 summaries eligible for MUS-based pruning at this check."
        )

    return state


def _maybe_prune_l2_memories_mus(state: AgentTurnState) -> dict[str, Any]:
    """
    Checks if it's time to prune Level 2 (chapter) memories using MUS-based pruning.
    Only runs if enabled and at the configured interval.

    Args:
        state (AgentTurnState): The current agent graph state

    Returns:
        Dict[str, Any]: The updated state after potentially pruning L2 memories by MUS
    """
    import src.infra.config as config

    agent_id = state["agent_id"]
    sim_step = state["simulation_step"]
    vector_store = state.get("vector_store_manager")

    # Check if MUS-based L2 memory pruning is enabled
    if not getattr(config, "MEMORY_PRUNING_L2_MUS_ENABLED", False):
        return state

    # Check if it's time to run MUS-based L2 pruning based on check interval
    check_interval = getattr(config, "MEMORY_PRUNING_L2_MUS_CHECK_INTERVAL_STEPS", 100)
    if sim_step % check_interval != 0:
        return state

    logger.info(f"Agent {agent_id}: Performing MUS-based L2 pruning check at step {sim_step}")

    mus_threshold = getattr(config, "MEMORY_PRUNING_L2_MUS_THRESHOLD", 0.3)
    min_age_days = getattr(config, "MEMORY_PRUNING_L2_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION", 14)

    if not vector_store:
        logger.warning(
            f"Agent {agent_id}: No vector store manager available for MUS-based L2 pruning."
        )
        return state

    # Get L2 summaries that meet the MUS pruning criteria
    ids_to_prune = vector_store.get_l2_memories_for_mus_pruning(mus_threshold, min_age_days)

    if ids_to_prune:
        logger.info(
            f"Agent {agent_id}: MUS-based L2 pruning: deleting {len(ids_to_prune)} L2 summaries (sample: {ids_to_prune[:3]})"
        )
        success = vector_store.delete_memories_by_ids(ids_to_prune)
        if success:
            logger.info(
                f"Agent {agent_id}: Successfully pruned {len(ids_to_prune)} L2 summaries by MUS."
            )
        else:
            logger.error(f"Agent {agent_id}: Failed to prune L2 summaries by MUS.")
    else:
        logger.info(
            f"Agent {agent_id}: No L2 summaries eligible for MUS-based pruning at this check."
        )

    return state


def create_basic_agent_graph():
    """
    Builds the agent turn graph with intent routing.
    Flow: Analyze Sentiment -> Prepare Prompt -> Retrieve Memories -> Generate Output -> Route Intent -> [Handle Intent] -> Update State -> Finalize Message -> END
    """
    workflow = StateGraph(AgentTurnState)

    # Add the nodes
    workflow.add_node("analyze_sentiment", analyze_perception_sentiment_node)
    workflow.add_node("prepare_relationship_prompt", prepare_relationship_prompt_node)
    # Add new RAG node for memory retrieval and summarization
    workflow.add_node("retrieve_memories", retrieve_and_summarize_memories_node)
    workflow.add_node("generate_action_output", generate_thought_and_message_node)
    workflow.add_node("handle_propose_idea", handle_propose_idea_node)  # Specific intent handler
    workflow.add_node(
        "handle_continue_collaboration", handle_continue_collaboration_node
    )  # Specific intent handler
    workflow.add_node("handle_idle", handle_idle_node)  # Specific intent handler
    workflow.add_node(
        "handle_ask_clarification", handle_ask_clarification_node
    )  # Specific intent handler
    workflow.add_node("handle_deep_analysis", handle_deep_analysis_node)  # Specific intent handler
    workflow.add_node(
        "handle_create_project", handle_create_project_node
    )  # Specific intent handler
    workflow.add_node("handle_join_project", handle_join_project_node)  # Specific intent handler
    workflow.add_node("handle_leave_project", handle_leave_project_node)  # Specific intent handler
    workflow.add_node(
        "handle_send_direct_message", handle_send_direct_message_node
    )  # New intent handler
    workflow.add_node("update_state", update_state_node)  # Unified state update
    workflow.add_node(
        "prune_l2_memories", _maybe_prune_l2_memories
    )  # L2 memory age-based pruning node
    workflow.add_node(
        "prune_l2_mus_memories", _maybe_prune_l2_memories_mus
    )  # MUS-based L2 memory pruning node
    workflow.add_node(
        "prune_l1_mus_memories", _maybe_prune_l1_memories_mus
    )  # MUS-based L1 memory pruning node
    workflow.add_node(
        "prune_l1_memories", _maybe_prune_l1_memories
    )  # L1 memory age-based pruning node
    workflow.add_node(
        "consolidate_memories", _maybe_consolidate_memories
    )  # Memory consolidation node
    workflow.add_node(
        "finalize_message", finalize_message_agent_node
    )  # Final decision on message sending

    # Define edges
    workflow.set_entry_point("analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "prepare_relationship_prompt")
    workflow.add_edge("prepare_relationship_prompt", "retrieve_memories")
    workflow.add_edge("retrieve_memories", "generate_action_output")

    # Route to the appropriate handler based on the intent
    workflow.add_conditional_edges(
        "generate_action_output",
        route_action_intent,  # This function returns the NAME of the next node
        {
            # The keys here MUST match the exact strings returned by route_action_intent
            "handle_propose_idea": "handle_propose_idea",
            "handle_ask_clarification": "handle_ask_clarification",
            "handle_continue_collaboration": "handle_continue_collaboration",
            "handle_idle": "handle_idle",
            "handle_deep_analysis": "handle_deep_analysis",
            "handle_create_project": "handle_create_project",
            "handle_join_project": "handle_join_project",
            "handle_leave_project": "handle_leave_project",
            "handle_send_direct_message": "handle_send_direct_message",
            "update_state": "update_state",  # Handles the 'else' case from route_action_intent
        },
    )

    # All action handlers go to update_state
    workflow.add_edge("handle_propose_idea", "update_state")
    workflow.add_edge("handle_continue_collaboration", "update_state")
    workflow.add_edge("handle_idle", "update_state")
    workflow.add_edge("handle_ask_clarification", "update_state")
    workflow.add_edge("handle_deep_analysis", "update_state")
    workflow.add_edge("handle_create_project", "update_state")
    workflow.add_edge("handle_join_project", "update_state")
    workflow.add_edge("handle_leave_project", "update_state")
    workflow.add_edge("handle_send_direct_message", "update_state")

    # Update state goes to L2 memory pruning (age-based)
    workflow.add_edge("update_state", "prune_l2_memories")

    # Add the new L2 MUS-based pruning node to the memory management pipeline
    workflow.add_edge("prune_l2_memories", "prune_l2_mus_memories")

    # Continue with the rest of the memory management pipeline
    workflow.add_edge("prune_l2_mus_memories", "prune_l1_mus_memories")
    workflow.add_edge("prune_l1_mus_memories", "prune_l1_memories")
    workflow.add_edge("prune_l1_memories", "consolidate_memories")
    workflow.add_edge("consolidate_memories", "finalize_message")

    # Finalize message is the end of the graph
    workflow.add_edge("finalize_message", END)

    # Compile the workflow
    return workflow.compile()


# Compile the graph
basic_agent_graph_compiled = create_basic_agent_graph()


def is_dspy_lm_configured() -> bool:
    """Checks if a DSPy LM is configured in settings."""
    lm = getattr(dspy.settings, "lm", None)
    if lm is None:
        logger.debug("[DSPy] LM not configured at time of check.")
        return False
    return True
