# mypy: ignore-errors
# mypy: disable-error-code=unused-ignore
# src/agents/graphs/basic_agent_graph.py
"""
Defines the basic LangGraph structure for an agent's turn.
"""

import logging
import os
import random
import sys
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from src.agents.core.agent_state import AgentState
from src.agents.core.roles import ROLE_ANALYZER, ROLE_FACILITATOR, ROLE_INNOVATOR

# Import L1SummaryGenerator for DSPy-based L1 summary generation
from src.agents.dspy_programs.l1_summary_generator import L1SummaryGenerator

# Import L2SummaryGenerator for DSPy-based L2 summary generation
from src.infra import config  # Import config for role change parameters
from src.infra.llm_client import analyze_sentiment, generate_structured_output

from .agent_graph_builder import build_graph

# Module logger
logger = logging.getLogger(__name__)

# At the top of the file, after other DSPy imports
try:
    import os  # Redundant but harmless

    from dspy import Predict

    # Mypy: Suppressing noise from transformers internal error
    # Use the robust DSPy relationship updater loader with fallback logic
    from src.agents.dspy_programs.relationship_updater import get_relationship_updater

    _relationship_updater = get_relationship_updater()

    _REL_UPDATER_PATH = os.path.join(
        os.path.dirname(__file__),
        "../dspy_programs/compiled/optimized_relationship_updater.json",
    )
    if os.path.exists(_REL_UPDATER_PATH):
        _optimized_relationship_updater = Predict.load(
            _relationship_updater,
            _REL_UPDATER_PATH,
        )
    else:
        _optimized_relationship_updater = None
except Exception as e:
    _optimized_relationship_updater = None
    logger.error("Failed to load OptimizedRelationshipUpdater: %s", e)


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
            f"Failed to load DSPy optimized_action_selector: {e}. "
            "Action intent selection will use fallback.",
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
    dspy = None
except Exception as e:  # Catch any other exception during DSPy setup
    logging.critical(f"CRITICAL ERROR during general DSPy setup: {e}", exc_info=True)
    DSPY_AVAILABLE = False
    DSPY_ACTION_SELECTOR_AVAILABLE = False
    optimized_action_selector = None
    dspy = None


# Use TYPE_CHECKING to avoid circular import issues
if TYPE_CHECKING:
    pass  # No specific type hints needed here for now

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


# Define the state the graph will operate on during a single agent turn
class AgentTurnState(TypedDict):
    """Represents the state passed into and modified by the agent's graph turn."""

    agent_id: str
    current_state: dict[str, object]  # The agent's full state dictionary
    simulation_step: int  # The current step number from the simulation
    previous_thought: str | None  # The thought from the *last* turn
    environment_perception: dict[str, object]  # Perception data from the environment
    perceived_messages: list[dict[str, object]]  # Messages perceived from last step
    memory_history_list: list[dict[str, object]]  # Field for memory history list
    turn_sentiment_score: int  # Field for aggregated sentiment score
    prompt_modifier: str  # Field for relationship-based prompt adjustments
    structured_output: AgentActionOutput | None  # Holds the parsed LLM output object
    agent_goal: str  # The agent's goal for the simulation
    updated_state: dict[str, object]  # Output field: The updated state after the turn
    vector_store_manager: object | None  # For persisting memories to vector store
    rag_summary: str  # Summarized memories from vector store
    knowledge_board_content: list[str]  # Current entries on the knowledge board
    knowledge_board: object | None  # The knowledge board instance for posting entries
    scenario_description: str  # Description of the simulation scenario
    current_role: str  # The agent's current role in the simulation
    influence_points: int  # The agent's current Influence Points
    steps_in_current_role: int  # Steps taken in the current role
    data_units: int  # The agent's current Data Units
    current_project_affiliation: str | None  # The agent's current project ID (if any)
    available_projects: dict[str, object]  # Dictionary of available projects
    state: AgentState  # The agent's structured state object (new Pydantic model)
    collective_ip: float | None  # Total IP across all agents in the simulation
    collective_du: float | None  # Total DU across all agents in the simulation


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
        f"Node 'analyze_perception_sentiment_node' executing for agent {agent_id} "
        f"at step {sim_step}"
    )

    total_sentiment_score = 0
    analyzed_count = 0

    if not perceived_messages:
        logger.debug("  No messages perceived, sentiment score remains 0.")
        return {"turn_sentiment_score": 0}

    for msg in perceived_messages:
        sender_id = msg.get("sender_id", "unknown")
        message_content = msg.get("content", None)
        if sender_id == agent_id:  # Skip own messages
            continue
        if isinstance(message_content, str):
            sentiment = analyze_sentiment(message_content)
            if sentiment == "positive":
                total_sentiment_score += 1
            elif sentiment == "negative":
                total_sentiment_score -= 1
            analyzed_count += 1

    logger.info(
        f"Agent {agent_id}: Aggregated sentiment score from {analyzed_count} perceived messages: "
        f"{total_sentiment_score}"
    )
    return {"turn_sentiment_score": total_sentiment_score}


def prepare_relationship_prompt_node(state: AgentTurnState) -> dict[str, str]:
    """
    Prepares a detailed relationship prompt modifier.
    """
    from src.infra.config import get_relationship_label  # Local import

    agent_state = state["state"]
    relationships = agent_state.relationships

    modifier = "You have neutral relationships. Maintain a balanced approach."
    if relationships:
        descriptions = [
            f"- Agent_{other_id}: {get_relationship_label(score)} (Score: {score:.2f})"
            for other_id, score in relationships.items()
        ]
        summary = "RELATIONSHIPS:\n" + "\n".join(descriptions)

        guidance = [
            "RELATIONSHIP INFLUENCE GUIDANCE:",
            "Adjust tone: warm for allies, neutral for neutral, cautious for negative.",
            "Target positive relations for collaboration, be cautious with negative ones.",
            "Prioritize input from allies in decisions.",
        ]
        modifier = f"{summary}\n\n" + "\n".join(guidance)

    return {"prompt_modifier": modifier}


async def retrieve_and_summarize_memories_node(state: AgentTurnState) -> dict[str, str]:
    """
    Asynchronously retrieve and summarize memories.
    """
    agent = state.get("agent_instance")  # Passed from Simulation if Agent uses this graph
    agent_id = state["agent_id"]
    agent_goal = state.get("agent_goal", "Contribute effectively.")
    vector_store_manager = state.get("vector_store_manager")
    sim_step = state["simulation_step"]
    previous_thought = state.get("previous_thought", "")

    # Construct query from goal and recent thought/messages
    query_parts = [f"Memories relevant to: Goal '{agent_goal}'"]
    if previous_thought:
        query_parts.append(f"Recent thought: {previous_thought}")
    # Consider adding recent messages to query context if impactful
    query_text = ". ".join(query_parts)

    logger.debug(
        f"Agent {agent_id} (Step {sim_step}): Retrieving memories with query: '{query_text}'"
    )

    if (
        not vector_store_manager or not agent
    ):  # Agent instance needed for its async_generate_l1_summary
        logger.warning(
            f"Agent {agent_id}: Vector store or agent instance missing for memory retrieval."
        )
        return {"rag_summary": "(No memory retrieval: store or agent missing)"}

    try:
        if not hasattr(vector_store_manager, "aretrieve_relevant_memories"):
            logger.error("Vector store manager missing 'aretrieve_relevant_memories'.")
            return {"rag_summary": "(Retrieval method missing)"}

        retrieved_memories = await vector_store_manager.aretrieve_relevant_memories(
            agent_id, query=query_text, k=5
        )

        if not retrieved_memories:
            logger.info(f"Agent {agent_id}: No relevant memories found.")
            return {"rag_summary": "(No relevant past memories found)"}

        logger.info(f"Agent {agent_id}: Retrieved {len(retrieved_memories)} memories.")
        # Ensure agent has the async_generate_l1_summary method
        if not hasattr(agent, "async_generate_l1_summary"):
            logger.error("Agent instance missing 'async_generate_l1_summary' method.")
            return {"rag_summary": "(Agent summary method missing)"}

        current_role = state.get("current_role", "unknown")
        # Pass memories as string, assuming the method handles parsing if needed
        # Ensure that retrieved_memories (list of dicts) is correctly processed into a string for summary
        memories_content_list = [
            str(mem.get("content", ""))
            for mem in retrieved_memories
            if isinstance(mem, dict) and mem.get("content")
        ]
        memories_str = "\n".join(memories_content_list)

        # Get current mood from the AgentState object within the turn state
        current_mood_value = (
            state["state"].mood_value if hasattr(state["state"], "mood_value") else 0.0
        )
        current_mood_descriptive = get_descriptive_mood(current_mood_value)

        summary_result_obj = await agent.async_generate_l1_summary(
            current_role, memories_str, current_mood_descriptive
        )
        # Extract summary text from DSPy result (assuming it has a 'summary' attribute)
        summary_text = getattr(
            summary_result_obj, "summary", "(Summarization error or empty summary)"
        )

        logger.info(f"Agent {agent_id}: Memory summary generated (length: {len(summary_text)}).")
        return {"rag_summary": summary_text}

    except Exception as e:
        logger.error(
            f"Agent {agent_id}: Error in memory retrieval/summarization: {e}", exc_info=True
        )
        return {"rag_summary": "(Memory retrieval/summary failed)"}


async def generate_thought_and_message_node(
    state: AgentTurnState,
) -> dict[str, AgentActionOutput | None]:
    """
    Asynchronously generate the agent's thought and message.
    """
    agent = state.get("agent_instance")
    agent_id = state["agent_id"]
    agent_state_obj = state["state"]  # This is the AgentState Pydantic model

    # Logging for diagnostics
    logger.info(f"Agent {agent_id}: Type of agent_state_obj in graph: {type(agent_state_obj)}")
    if isinstance(agent_state_obj, dict):
        logger.info(
            f"Agent {agent_id}: agent_state_obj is a dict. Keys: {list(agent_state_obj.keys())}"
        )
    elif isinstance(agent_state_obj, BaseModel):  # Check if it's a Pydantic model
        logger.info(
            f"Agent {agent_id}: agent_state_obj is a Pydantic model. Fields: {list(AgentState.model_fields.keys())}"
        )
        logger.info(f"Agent {agent_id}: agent_state_obj dir(): {dir(agent_state_obj)}")
        logger.info(
            f"Agent {agent_id}: agent_state_obj vars(): {vars(agent_state_obj) if hasattr(agent_state_obj, '__dict__') else 'N/A'}"
        )

    role = agent_state_obj.role
    agent_goal = state.get("agent_goal", "Contribute effectively.")
    scenario_description = state.get("scenario_description", "")
    perception = state.get("environment_perception", {})
    knowledge_board_content = state.get("knowledge_board_content", [])
    perceived_messages = state.get("perceived_messages", [])
    rag_summary = state.get("rag_summary", "(No relevant memories)")
    previous_thought = state.get("previous_thought")
    prompt_modifier = state.get("prompt_modifier", "")  # Relationship context

    newline_char = "\n"

    # Prepare recent messages string
    if perceived_messages:
        message_strings = [
            f"- {m.get('sender_name', 'Someone')}: {m.get('content', '')}{newline_char}"
            for m in perceived_messages
        ]
        recent_messages_str = "".join(message_strings)
    else:
        recent_messages_str = "None"

    # Current situation string
    situation_parts = [
        f"Environment: {perception.get('description', 'N/A')}",
        f"Recent messages: {recent_messages_str}",
        f"Knowledge Board: {' | '.join(knowledge_board_content) if knowledge_board_content else 'Empty'}",
        f"Your previous thought: {previous_thought if previous_thought else 'First turn.'}",
        f"Your Goals: {agent_goal}",
        f"Relevant Memories: {rag_summary}",
        f"Scenario: {scenario_description}",
    ]
    current_situation = "\n\n".join(situation_parts)

    # Phase 1: Role-prefixed thought start (DSPy)
    role_thought_start = f"As a {role}, considering the situation..."
    if agent and hasattr(agent, "async_generate_role_prefixed_thought"):
        try:
            thought_result = await agent.async_generate_role_prefixed_thought(
                role, current_situation
            )
            # Ensure thought_result is not None before getattr
            if thought_result:
                role_thought_start = getattr(thought_result, "thought", role_thought_start)
        except Exception as e:
            logger.error(f"Agent {agent_id}: Error in async_generate_role_prefixed_thought: {e}")

    # Phase 2: Action intent selection (DSPy)
    chosen_intent_val = "idle"
    intent_justification = "Defaulting to idle."
    from src.agents.core.agent_state import AgentActionIntent  # Local import for clarity

    available_actions_str = [intent.value for intent in AgentActionIntent]

    if agent and hasattr(agent, "async_select_action_intent"):
        try:
            logger.debug(
                f"Agent {agent_id}: PRE-AWAIT agent.async_select_action_intent. Type: {type(agent.async_select_action_intent)}"
            )  # New log
            action_result = await agent.async_select_action_intent(
                role, current_situation, agent_goal, available_actions_str
            )
            # Ensure action_result is not None
            if action_result:
                chosen_intent_val = getattr(action_result, "chosen_action_intent", "idle")
                intent_justification = getattr(
                    action_result, "justification_thought", "No justification provided."
                )
        except Exception as e:
            logger.error(
                f"Agent {agent_id}: Error in async_select_action_intent: {e}", exc_info=True
            )  # Added exc_info

    # Phase 3: Combine thoughts
    final_agent_thought = f"{role_thought_start} {intent_justification}"

    # Phase 4: Prepare prompt for main LLM call (structured output)
    prompt_elements = [
        f"# AGENT: {agent_id} (Role: {role}, Goal: {agent_goal})",
        f"# SCENARIO: {scenario_description}",
        f"# PERCEPTION: {perception.get('description', 'N/A')}",
        f"# OTHER AGENTS: {_format_other_agents(perception.get('other_agents', []), agent_state_obj.relationships)}",
        f"# KNOWLEDGE BOARD:\n{_format_knowledge_board(knowledge_board_content)}",
        f"# PAST MEMORIES (Summary):\n{rag_summary}",
        f"# RECENT MESSAGES:\n{_format_messages(perceived_messages)}",
        f"# RELATIONSHIP CONTEXT:\n{prompt_modifier}",
        f"# PREVIOUS THOUGHT:\n{previous_thought if previous_thought else 'N/A'}",
        f"# CURRENT MOOD: {getattr(agent_state_obj, 'mood', 'unknown')} (Value: {(getattr(agent_state_obj, 'mood_value', 0.0)):.2f})",
        f"# YOUR INTERNAL THOUGHT (follow this reasoning):\n{final_agent_thought}",
        f"# YOUR CHOSEN ACTION INTENT (execute this): {chosen_intent_val}",
        "Based on your thought and chosen intent, formulate your structured response.",
    ]
    full_prompt = "\n\n".join(prompt_elements)

    llm_model_name = getattr(agent_state_obj, "llm_model_name", None) or config.DEFAULT_LLM_MODEL
    structured_llm_output = generate_structured_output(
        full_prompt, AgentActionOutput, model=str(llm_model_name)
    )

    if structured_llm_output:
        structured_llm_output.thought = final_agent_thought  # Override with our generated thought
        structured_llm_output.action_intent = chosen_intent_val  # Override with DSPy chosen intent
    else:
        logger.error(
            f"Agent {agent_id}: LLM failed to generate structured output. Using fallback."
        )
        structured_llm_output = AgentActionOutput(
            thought=final_agent_thought,
            action_intent=chosen_intent_val,
            message_content=None,  # Ensure None if no message
        )
    return {"structured_output": structured_llm_output}


def process_role_change(agent_state: AgentState, requested_role: str) -> bool:
    # (Implementation from original file)
    if requested_role not in VALID_ROLES:
        logger.warning(f"Agent {agent_state.agent_id} requested invalid role: {requested_role}")
        return False
    if requested_role == agent_state.role:
        return False
    if agent_state.steps_in_current_role < agent_state.role_change_cooldown:
        return False
    if agent_state.ip < agent_state.role_change_ip_cost:
        return False
    old_role = agent_state.role
    agent_state.ip -= agent_state.role_change_ip_cost
    agent_state.role = requested_role
    agent_state.steps_in_current_role = 0
    agent_state.role_history.append((int(agent_state.last_action_step or 0), requested_role))
    logger.info(f"Agent {agent_state.agent_id} changed role from {old_role} to {requested_role}.")
    return True


def update_state_node(state: AgentTurnState) -> dict[str, Any]:
    agent_id = state["agent_id"]
    sim_step = state["simulation_step"]
    structured_output = state.get("structured_output")
    action_intent = structured_output.action_intent if structured_output else "idle"
    agent_state_obj = state["state"]

    if structured_output and structured_output.requested_role_change:
        if process_role_change(agent_state_obj, structured_output.requested_role_change):
            agent_state_obj.add_memory(
                sim_step,
                "role_change",
                f"Changed role to {structured_output.requested_role_change}",
            )
        else:
            agent_state_obj.add_memory(
                sim_step, "resource_constraint", "Failed role change attempt"
            )

    if action_intent != "idle":
        role_name = agent_state_obj.role
        du_gen_rate = ROLE_DU_GENERATION.get(role_name, 1.0)  # Use constant
        generated_du = round(du_gen_rate * (0.5 + random.random()), 1)
        if generated_du > 0:
            agent_state_obj.du += generated_du
            logger.info(
                f"Agent {agent_id}: Generated {generated_du} DU. Total DU: {agent_state_obj.du:.1f}"
            )
        # Increment steps taken in current role after performing an action
        agent_state_obj.steps_in_current_role += 1

    if (
        len(agent_state_obj.short_term_memory) >= 3
        and hasattr(agent_state_obj, "llm_client")
        and agent_state_obj.llm_client
    ):
        try:
            l1_gen = L1SummaryGenerator()
            recent_events_str = "\n".join(
                [
                    str(mem.get("content", ""))
                    for mem in list(agent_state_obj.short_term_memory)[-10:]
                ]
            )
            # Use public property with hasattr
            mood_val_for_summary = 0.0
            if hasattr(agent_state_obj, "mood_value"):
                mood_val_for_summary = getattr(agent_state_obj, "mood_value", 0.0)
            mood_desc = get_descriptive_mood(mood_val_for_summary)

            # Call the dspy program
            summary_prediction = l1_gen(
                agent_role=agent_state_obj.role,
                recent_events=recent_events_str,
                current_mood=mood_desc,
            )
            memory_summary = getattr(summary_prediction, "summary", None)

            if memory_summary:
                agent_state_obj.add_memory(sim_step, "consolidated_summary", memory_summary)
                vector_store = state.get("vector_store_manager")
                if vector_store and hasattr(vector_store, "add_memory"):
                    vector_store.add_memory(
                        agent_id,
                        sim_step,
                        "consolidated_summary",
                        memory_summary,
                        "consolidated_summary",
                    )  # Added memory_type
                logger.info(f"Agent {agent_id}: Generated L1 memory summary.")
        except Exception as e:
            logger.error(
                f"Agent {agent_id}: Error during L1 memory consolidation: {e}", exc_info=True
            )  # Added exc_info

    updated_state_dict = dict(state)
    updated_state_dict["state"] = agent_state_obj
    if hasattr(agent_state_obj, "du"):
        updated_state_dict["data_units"] = int(agent_state_obj.du)
    return updated_state_dict


def _maybe_prune_l1_memories(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


def _maybe_prune_l2_memories(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


async def finalize_message_agent_node(state: AgentTurnState) -> dict[str, Any]:
    agent = state.get("agent_instance")
    agent_id = state["agent_id"]
    structured_output = state.get("structured_output")
    final_agent_state_obj = state["state"]

    if not structured_output or not final_agent_state_obj:
        logger.debug(f"Agent {agent_id}: No structured output or agent state to finalize.")
        return {
            "message_content": None,
            "message_recipient_id": None,
            "action_intent": "idle",
            "updated_agent_state": final_agent_state_obj,
        }

    message_content = structured_output.message_content
    message_recipient_id = structured_output.message_recipient_id
    action_intent = structured_output.action_intent

    if message_content and message_recipient_id and message_recipient_id != agent_id and agent:
        sentiment = analyze_sentiment(message_content)
        sentiment_score_map = {"positive": 0.5, "neutral": 0.0, "negative": -0.5}
        interaction_sentiment_score = sentiment_score_map.get(sentiment, 0.0)

        if hasattr(agent, "async_update_relationship"):
            try:
                current_rel_score = final_agent_state_obj.relationships.get(
                    message_recipient_id, 0.0
                )
                agent1_persona_str = str(final_agent_state_obj.role or "Unknown")
                # Simplified agent2_persona retrieval for now
                # In a real sim, this would come from querying the state of the other agent
                agent2_state = (
                    state.get("environment_perception", {})
                    .get("all_agents_map", {})
                    .get(message_recipient_id)
                )
                agent2_persona_str = (
                    str(agent2_state.role)
                    if agent2_state and hasattr(agent2_state, "role")
                    else "Unknown"
                )

                dspy_result = await agent.async_update_relationship(
                    current_relationship_score=current_rel_score,
                    interaction_summary=message_content[:100],
                    agent1_persona=agent1_persona_str,
                    agent2_persona=agent2_persona_str,  # This needs to be the actual persona of agent 2
                    interaction_sentiment=interaction_sentiment_score,
                )
                # Ensure dspy_result is not None before getattr
                new_score = current_rel_score  # Default to current score
                rationale = "N/A"
                if dspy_result:
                    new_score = float(
                        getattr(dspy_result, "new_relationship_score", current_rel_score)
                    )
                    rationale = getattr(dspy_result, "relationship_change_rationale", "N/A")

                final_agent_state_obj.relationships[message_recipient_id] = max(
                    final_agent_state_obj.min_relationship_score,
                    min(final_agent_state_obj.max_relationship_score, new_score),
                )
                logger.info(
                    f"[DSPy] Rel update ({agent_id}->{message_recipient_id}): {current_rel_score:.2f} -> {new_score:.2f}. Rationale: {rationale}"
                )
            except Exception as e:
                logger.error(
                    f"DSPy RelationshipUpdater failed for {agent_id}->{message_recipient_id}: {e}",
                    exc_info=True,
                )

    return {
        "message_content": message_content,
        "message_recipient_id": message_recipient_id,
        "action_intent": action_intent,
        "updated_agent_state": final_agent_state_obj,
        "is_targeted": message_recipient_id is not None,
    }


def _format_other_agents(
    other_agents_info: list[dict[str, Any]], relationships: dict[str, float]
) -> str:
    from src.infra.config import get_relationship_label

    if not other_agents_info:
        return "  You are currently alone."
    lines = ["  Other agents you can interact with (use their ID when sending targeted messages):"]
    for agent_info in other_agents_info:
        other_id = agent_info.get("agent_id", "unknown")
        other_name = agent_info.get("name", other_id[:8])
        other_mood = agent_info.get("mood", "unknown")
        relationship_score = relationships.get(other_id, 0.0)
        relationship_label = get_relationship_label(relationship_score)
        lines.append(
            f"  - {other_name} (Agent ID: '{other_id}', Mood: {other_mood}, Relationship: {relationship_label} ({relationship_score:.1f}))"
        )
    return "\n".join(lines)


def _format_knowledge_board(board_entries: list[str]) -> str:
    if not board_entries:
        return "  (Board is currently empty)"
    lines = [
        "  You can reference a board entry by its Step and original Agent ID (e.g., 'Regarding Step 3's idea by Agent_XYZ...')."
    ]
    for entry in board_entries:
        lines.append(f"  - {entry}")
    return "\n".join(lines)


def _format_messages(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "  No messages were perceived in the previous step."
    lines = []
    for msg in messages:
        sender = msg.get("sender_id", "unknown")
        content = msg.get("content", "")
        recipient = msg.get("recipient_id")
        message_type = "(Private to you)" if recipient else "(Broadcast)"
        lines.append(f'  - {sender} {message_type}: "{content}"')
    return "\n".join(lines)


def get_mood_level(mood_value: float) -> str:
    if mood_value < -0.3:
        return "unhappy"
    elif mood_value > 0.3:
        return "happy"
    return "neutral"


def get_descriptive_mood(mood_value: float) -> str:
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
    return "very happy"


def shorten_message(message: str) -> str:
    return message


def route_broadcast_decision(state: AgentTurnState) -> str:
    structured_output = state.get("structured_output")
    if structured_output and structured_output.message_content:
        return "broadcast"
    return "exit"


def route_relationship_context(state: AgentTurnState) -> str:
    # Ensure state["state"] (AgentState object) is accessed for relationships
    agent_state_obj = state.get("state")
    if agent_state_obj and agent_state_obj.relationships:
        return "has_relationships"
    return "no_relationships"


def handle_propose_idea_node(state: AgentTurnState) -> dict[str, Any]:
    agent_id = state["agent_id"]
    sim_step = state["simulation_step"]
    structured_output = state.get("structured_output")
    kb = state.get("knowledge_board")
    agent_state_obj = state["state"]
    idea_content = structured_output.message_content if structured_output else None

    if idea_content and kb and hasattr(kb, "add_entry"):
        # Simplified cost/award logic
        agent_state_obj.du -= PROPOSE_DETAILED_IDEA_DU_COST
        agent_state_obj.ip -= IP_COST_TO_POST_IDEA
        agent_state_obj.ip += IP_AWARD_FOR_PROPOSAL
        kb.add_entry(f"Idea from {agent_id}: {idea_content}", agent_id, sim_step)
        logger.info(f"Agent {agent_id} posted idea to KB: {idea_content[:50]}...")

    ret_state = dict(state)  # Ensure a copy for modification
    ret_state["state"] = agent_state_obj  # Put back AgentState obj
    return ret_state


def handle_continue_collaboration_node(state: AgentTurnState) -> dict[str, Any]:
    agent_state_obj = state["state"]  # Deduct IP cost for broadcasting a message
    agent_state_obj.ip -= agent_state_obj.ip_cost_per_message
    return dict(state)


def handle_idle_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


def handle_ask_clarification_node(state: AgentTurnState) -> dict[str, Any]:
    agent_state_obj = state["state"]
    # Simplified: Assume DU cost is handled if needed by other logic or is minor.
    # In a full version, check message length/keywords for DU_COST_REQUEST_DETAILED_CLARIFICATION
    ret_state = dict(state)
    ret_state["state"] = agent_state_obj
    return ret_state


def handle_deep_analysis_node(state: AgentTurnState) -> dict[str, Any]:
    agent_state_obj = state["state"]
    agent_state_obj.du -= DU_COST_DEEP_ANALYSIS  # Example cost
    logger.info(
        f"Agent {state['agent_id']} performed deep analysis, DU cost {DU_COST_DEEP_ANALYSIS}."
    )
    ret_state = dict(state)
    ret_state["state"] = agent_state_obj
    return ret_state


def handle_create_project_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


def handle_join_project_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


def handle_leave_project_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


def handle_send_direct_message_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


def route_action_intent(state: AgentTurnState) -> str:
    structured_output = state.get("structured_output")
    intent = structured_output.action_intent if structured_output else "idle"

    route_map = {
        "propose_idea": "handle_propose_idea",
        "ask_clarification": "handle_ask_clarification",
        "continue_collaboration": "handle_continue_collaboration",
        "idle": "handle_idle",
        "perform_deep_analysis": "handle_deep_analysis",
        "create_project": "handle_create_project",
        "join_project": "handle_join_project",
        "leave_project": "handle_leave_project",
        "send_direct_message": "handle_send_direct_message",
    }
    return route_map.get(intent, "update_state")


def _maybe_consolidate_memories(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


# --- Graph Definition ---

def compile_agent_graph() -> Any:
    """Compile and return the Basic Agent Graph executor."""
    from .agent_graph_builder import build_graph

    try:
        graph_builder = build_graph()
        executor = graph_builder.compile()
        logger.info(
            "AGENT_GRAPH_COMPILATION_SUCCESS: Basic Agent Graph compiled and assigned to executor."
        )
        return executor
    except Exception as e:  # pragma: no cover - compilation rarely fails
        logger.critical(
            f"AGENT_GRAPH_COMPILATION_ERROR: CRITICAL ERROR during graph compilation in basic_agent_graph.py: {e}",
            exc_info=True,
        )
        return None