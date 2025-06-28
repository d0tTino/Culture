# src/agents/graphs/basic_agent_graph.py
"""
Defines the basic LangGraph structure for an agent's turn.
"""

import hashlib
import json
import logging
import random
import sys
from math import sqrt
from typing import TYPE_CHECKING, Any, cast

from src.agents.core.agent_controller import AgentController
from src.agents.core.agent_graph_types import AgentTurnState
from src.agents.core.agent_state import AgentState
from src.agents.core.mood_utils import get_descriptive_mood
from src.agents.core.role_embeddings import ROLE_EMBEDDINGS
from src.agents.core.roles import ensure_profile

# Import L1SummaryGenerator for DSPy-based L1 summary generation
from src.agents.dspy_programs.l1_summary_generator import L1SummaryGenerator

# Import L2SummaryGenerator for DSPy-based L2 summary generation
from src.infra import config  # Import config for role change parameters

from .interaction_handlers import (  # noqa: F401 - imported for re-export
    handle_create_project_node,
    handle_join_project_node,
    handle_leave_project_node,
    handle_send_direct_message_node,
)

# Module logger
logger = logging.getLogger(__name__)


def compute_trace_hash(state: AgentTurnState) -> str:
    """Return a deterministic hash for a LangGraph invocation."""
    relevant = {
        "agent_id": state.get("agent_id"),
        "simulation_step": state.get("simulation_step"),
    }
    agent_state = state.get("state")
    if isinstance(agent_state, AgentState):
        relevant["current_role"] = getattr(agent_state, "current_role", None)
    payload = json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


try:
    from pathlib import Path

    from dspy_ai import Predict

    # Mypy: Suppressing noise from transformers internal error
    # Use the robust DSPy relationship updater loader with fallback logic
    from src.agents.dspy_programs.relationship_updater import get_relationship_updater

    _relationship_updater = get_relationship_updater()

    _REL_UPDATER_PATH = (
        Path(__file__).resolve().parent
        / "../dspy_programs/compiled/optimized_relationship_updater.json"
    )
    if _REL_UPDATER_PATH.exists():
        _optimized_relationship_updater = Predict.load(
            _relationship_updater,
            str(_REL_UPDATER_PATH),
        )
    else:
        _optimized_relationship_updater = None
except Exception as e:
    _optimized_relationship_updater = None
    logger.error("Failed to load OptimizedRelationshipUpdater: %s", e)


# Add dspy imports with detailed error handling
try:
    logging.info("Attempting to import DSPy modules...")
    sys.path.insert(0, str(Path(".").resolve()))  # Ensure the root directory is in the path
    import dspy_ai as dspy
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

# --- Node Functions ---


def _get_current_role(agent_state: AgentState) -> str:
    if hasattr(agent_state, "current_role"):
        cur = getattr(agent_state, "current_role")
        if hasattr(cur, "name"):
            return cast(str, cur.name)
        return cast(str, cur)
    return cast(str, getattr(agent_state, "role"))


def _set_current_role(agent_state: AgentState, role: str) -> None:
    if hasattr(agent_state, "current_role"):
        agent_state.current_role = ensure_profile(role)
    else:
        setattr(agent_state, "role", role)


def _embedding_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Return cosine similarity between two vectors."""
    if not vec1 or not vec2:
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sqrt(sum(a * a for a in vec1))
    norm2 = sqrt(sum(b * b for b in vec2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


def process_role_change(agent_state: AgentState, requested_role: str) -> bool:
    resolved_role, _ = ROLE_EMBEDDINGS.best_role(requested_role)
    if resolved_role is None:
        logger.warning(f"Agent {agent_state.agent_id} requested invalid role: {requested_role}")
        return False
    requested_role = resolved_role
    current_role = _get_current_role(agent_state)
    if requested_role == current_role:
        return False
    requested_emb = ROLE_EMBEDDINGS.role_vectors.get(requested_role)
    similarity = (
        _embedding_similarity(agent_state.role_embedding, requested_emb) if requested_emb else 0.0
    )
    if similarity < 0.7:
        return False
    if agent_state.steps_in_current_role < agent_state.role_change_cooldown:
        return False
    if agent_state.ip < agent_state.role_change_ip_cost:
        return False
    old_role = current_role
    start_ip = agent_state.ip
    agent_state.ip -= agent_state.role_change_ip_cost
    try:
        from src.infra.ledger import ledger

        ledger.log_change(
            agent_state.agent_id,
            agent_state.ip - start_ip,
            0.0,
            "role_change",
        )
    except Exception:  # pragma: no cover - optional
        logger.debug("Ledger logging failed", exc_info=True)
    _set_current_role(agent_state, requested_role)
    agent_state.steps_in_current_role = 0
    agent_state.role_history.append((int(agent_state.last_action_step or 0), requested_role))
    agent_state.role_reputation[requested_role] = similarity
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
            AgentController(agent_state_obj).add_memory(
                f"Changed role to {structured_output.requested_role_change}",
                {"step": sim_step, "type": "role_change"},
            )
        else:
            AgentController(agent_state_obj).add_memory(
                "Failed role change attempt",
                {"step": sim_step, "type": "resource_constraint"},
            )

    if action_intent != "idle":
        role_name = _get_current_role(agent_state_obj)
        role_du_conf = config.ROLE_DU_GENERATION.get(role_name, {"base": 1.0})
        if isinstance(role_du_conf, dict):
            du_gen_rate = role_du_conf.get("base", 1.0)
        else:
            du_gen_rate = float(role_du_conf)

        generated_du = round(du_gen_rate * (0.5 + random.random()), 1)
        if generated_du > 0:
            start_du = agent_state_obj.du
            agent_state_obj.du += generated_du
            logger.info(
                f"Agent {agent_id}: Generated {generated_du} DU. Total DU: {agent_state_obj.du:.1f}"
            )
            try:
                from src.infra.ledger import ledger

                ledger.log_change(
                    agent_state_obj.agent_id,
                    0.0,
                    agent_state_obj.du - start_du,
                    "du_generation",
                )
            except Exception:  # pragma: no cover - optional
                logger.debug("Ledger logging failed", exc_info=True)
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
            summary_prediction = l1_gen.generate_summary(
                agent_role=_get_current_role(agent_state_obj),
                recent_events=recent_events_str,
                current_mood=mood_desc,
            )
            memory_summary = getattr(summary_prediction, "summary", None)

            if memory_summary:
                AgentController(agent_state_obj).add_memory(
                    memory_summary,
                    {"step": sim_step, "type": "consolidated_summary"},
                )
                vector_store = state.get("vector_store_manager")
                if vector_store and hasattr(vector_store, "add_memory"):
                    vector_store.add_memory(
                        agent_id,
                        sim_step,
                        "consolidated_summary",
                        memory_summary,
                        "consolidated_summary",
                    )
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
    manager = state.get("semantic_manager")
    step = int(state.get("simulation_step", 0))
    interval = int(
        config.get_config_value_with_override("SEMANTIC_MEMORY_CONSOLIDATION_INTERVAL_STEPS", 24)
    )
    if manager and interval > 0 and step % interval == 0:
        try:
            manager.consolidate_memories(state["agent_id"])
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Semantic consolidation failed: %s", exc, exc_info=True)
    return dict(state)


# --- Graph Definition ---


def build_graph() -> Any:
    """Wrapper for backward compatibility with older tests."""
    from .agent_graph_builder import build_graph as _build_graph

    return _build_graph()


def compile_agent_graph() -> Any:
    """Return the compiled Basic Agent Graph executor."""
    try:
        executor = build_graph()

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
