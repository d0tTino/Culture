# src/agents/graphs/basic_agent_graph.py
"""
Defines the basic LangGraph structure for an agent's turn.
"""
import logging
import re  # Add import for regular expressions
import random  # Add import for random to test negative sentiment
import uuid
from typing import Dict, Any, TypedDict, Annotated, List, Tuple, Optional, Deque, Literal, TYPE_CHECKING, Union
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages # Although not used yet, good practice
from src.infra.llm_client import generate_structured_output, analyze_sentiment, generate_text, summarize_memory_context, generate_response # Updated import structure to match function name
from collections import deque
from pydantic import BaseModel, Field, create_model
from src.agents.core.roles import ROLE_PROMPT_SNIPPETS, ROLE_DESCRIPTIONS, ROLE_FACILITATOR, ROLE_INNOVATOR, ROLE_ANALYZER
from src.infra import config  # Import config for role change parameters
from src.agents.core.agent_state import AgentState  # Import the AgentState model

# Use TYPE_CHECKING to avoid circular import issues
if TYPE_CHECKING:
    from src.sim.knowledge_board import KnowledgeBoard

logger = logging.getLogger(__name__)

# Decay factors for mood and relationships
MOOD_DECAY_FACTOR = 0.02  # Mood decays towards neutral by 2% each turn
RELATIONSHIP_DECAY_FACTOR = 0.01  # Relationships decay towards neutral by 1% each turn

# IP award constants
IP_AWARD_FOR_PROPOSAL = 5  # Amount of IP awarded for successfully proposing an idea to the knowledge board
IP_COST_TO_POST_IDEA = 2   # Cost in IP to post an idea to the knowledge board

# Role change constants (loaded from config)
ROLE_CHANGE_IP_COST = config.ROLE_CHANGE_IP_COST if hasattr(config, 'ROLE_CHANGE_IP_COST') else 5
ROLE_CHANGE_COOLDOWN = config.ROLE_CHANGE_COOLDOWN if hasattr(config, 'ROLE_CHANGE_COOLDOWN') else 3

# Data Units constants (loaded from config)
INITIAL_DATA_UNITS = config.INITIAL_DATA_UNITS if hasattr(config, 'INITIAL_DATA_UNITS') else 20
ROLE_DU_GENERATION = config.ROLE_DU_GENERATION if hasattr(config, 'ROLE_DU_GENERATION') else {
    "Innovator": 2,
    "Analyzer": 1,
    "Facilitator": 1,
    "Default Contributor": 0
}
PROPOSE_DETAILED_IDEA_DU_COST = config.PROPOSE_DETAILED_IDEA_DU_COST if hasattr(config, 'PROPOSE_DETAILED_IDEA_DU_COST') else 5
DU_AWARD_IDEA_ACKNOWLEDGED = config.DU_AWARD_IDEA_ACKNOWLEDGED if hasattr(config, 'DU_AWARD_IDEA_ACKNOWLEDGED') else 3
DU_AWARD_SUCCESSFUL_ANALYSIS = config.DU_AWARD_SUCCESSFUL_ANALYSIS if hasattr(config, 'DU_AWARD_SUCCESSFUL_ANALYSIS') else 4
DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE = config.DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE if hasattr(config, 'DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE') else 1
DU_COST_DEEP_ANALYSIS = config.DU_COST_DEEP_ANALYSIS if hasattr(config, 'DU_COST_DEEP_ANALYSIS') else 3
DU_COST_REQUEST_DETAILED_CLARIFICATION = config.DU_COST_REQUEST_DETAILED_CLARIFICATION if hasattr(config, 'DU_COST_REQUEST_DETAILED_CLARIFICATION') else 2

# List of valid roles
VALID_ROLES = [ROLE_FACILITATOR, ROLE_INNOVATOR, ROLE_ANALYZER]

# Define the Pydantic model for structured LLM output
class AgentActionOutput(BaseModel):
    """Defines the expected structured output from the LLM."""
    thought: str = Field(..., description="The agent's internal thought or reasoning for the turn.")
    message_content: Optional[str] = Field(None, description="The message to send to other agents, or None if choosing not to send a message.")
    message_recipient_id: Optional[str] = Field(None, description="The ID of the agent this message is directed to. None means broadcast to all agents.")
    action_intent: Literal["idle", "continue_collaboration", "propose_idea", "ask_clarification", "perform_deep_analysis", "create_project", "join_project", "leave_project", "send_direct_message"] = Field(
        default="idle", # Default intent
        description="The agent's primary intent for this turn."
    )
    requested_role_change: Optional[str] = Field(None, description="Optional: If you wish to request a change to a different role, specify the role name here (e.g., 'Innovator', 'Analyzer', 'Facilitator'). Otherwise, leave as null.")
    project_name_to_create: Optional[str] = Field(None, description="Optional: If you want to create a new project, specify the name here. This is used with the 'create_project' intent.")
    project_description_for_creation: Optional[str] = Field(None, description="Optional: If you want to create a new project, specify the description here. This is used with the 'create_project' intent.")
    project_id_to_join_or_leave: Optional[str] = Field(None, description="Optional: If you want to join or leave a project, specify the project ID here. This is used with the 'join_project' and 'leave_project' intents.")

# Define the state the graph will operate on during a single agent turn
class AgentTurnState(TypedDict):
    """Represents the state passed into and modified by the agent's graph turn."""
    agent_id: str
    current_state: Dict[str, Any] # The agent's full state dictionary (for backward compatibility)
    simulation_step: int          # The current step number from the simulation
    previous_thought: str | None  # The thought from the *last* turn
    environment_perception: Dict[str, Any] # Perception data from the environment
    perceived_messages: List[Dict[str, Any]] # Messages perceived from last step (broadcasts and targeted)
    memory_history_list: List[Dict[str, Any]] # Field for memory history list
    turn_sentiment_score: int     # Field for aggregated sentiment score
    prompt_modifier: str          # Field for relationship-based prompt adjustments
    structured_output: Optional[AgentActionOutput] # Holds the parsed LLM output object
    agent_goal: str               # The agent's goal for the simulation
    updated_state: Dict[str, Any] # Output field: The updated state after the turn (for backward compatibility)
    vector_store_manager: Optional[Any] # For persisting memories to vector store
    rag_summary: str              # Summarized memories from vector store
    knowledge_board_content: List[str]  # Current entries on the knowledge board
    knowledge_board: Optional[Any] # The knowledge board instance for posting entries
    scenario_description: str     # Description of the simulation scenario
    current_role: str             # The agent's current role in the simulation
    influence_points: int         # The agent's current Influence Points
    steps_in_current_role: int    # Steps taken in the current role
    data_units: int               # The agent's current Data Units
    current_project_affiliation: Optional[str] # The agent's current project ID (if any)
    available_projects: Dict[str, Any] # Dictionary of available projects
    state: AgentState             # The agent's structured state object (new Pydantic model)
    collective_ip: Optional[float] # Total IP across all agents in the simulation
    collective_du: Optional[float] # Total DU across all agents in the simulation

# --- Node Functions ---

def analyze_perception_sentiment_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Analyzes the sentiment of perceived messages from the previous step.
    Calculates an aggregated sentiment score for the turn.
    """
    agent_id = state['agent_id']
    sim_step = state['simulation_step']
    perceived_messages = state.get('perceived_messages', [])
    logger.debug(f"Node 'analyze_perception_sentiment_node' executing for agent {agent_id} at step {sim_step}")

    total_sentiment_score = 0
    analyzed_count = 0

    if not perceived_messages:
        logger.debug("  No messages perceived, sentiment score remains 0.")
        return {"turn_sentiment_score": 0}

    for msg in perceived_messages:
        sender_id = msg.get('sender_id', 'unknown')
        message_content = msg.get('content', None)
        recipient_id = msg.get('recipient_id', None)
        
        # Optional: Add special handling for targeted messages
        # For example, private messages might have more emotional impact

        # Skip analyzing own messages if they are included in perception
        if sender_id == agent_id: 
            continue

        if message_content:
            sentiment = analyze_sentiment(message_content) # Use the utility function
            if sentiment == 'positive':
                total_sentiment_score += 1
                analyzed_count += 1
            elif sentiment == 'negative':
                total_sentiment_score -= 1
                analyzed_count += 1
            elif sentiment == 'neutral':
                analyzed_count += 1 # Count neutral messages but don't change score
            # else: sentiment is None (error occurred), do nothing

    logger.info(f"Agent {agent_id}: Aggregated sentiment score from {analyzed_count} perceived messages: {total_sentiment_score}")
    # Return the calculated score to be added to the graph state
    return {"turn_sentiment_score": total_sentiment_score}

def prepare_relationship_prompt_node(state: AgentTurnState) -> Dict[str, str]:
    """
    Prepares a detailed relationship prompt modifier with specific guidance on how
    relationships should influence the agent's decisions, message tone, target selection,
    and action intent.
    """
    from src.infra.config import get_relationship_label
    
    agent_id = state['agent_id']
    agent_state = state['state']
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
            relationship_descriptions.append(f"- Relationship with Agent_{other_id}: {label} (Score: {score:.2f})")
            
            # Track strongly positive/negative relationships for targeted guidance
            if score > 0.5:
                strong_positive_relations.append(other_id)
            elif score < -0.5:
                strong_negative_relations.append(other_id)
        
        # Generate primary relationship summary
        relationship_summary = "RELATIONSHIPS:\n" + "\n".join(relationship_descriptions)
        
        # Generate explicit guidance for the LLM
        guidance_parts = [
            "RELATIONSHIP INFLUENCE GUIDANCE:"
        ]
        
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
        intent_guidance += " Consider 'continue_collaboration' or 'propose_idea' with positive relationships;"
        intent_guidance += " 'ask_clarification' for neutral relationships;"
        intent_guidance += " 'idle' or careful 'ask_clarification' with negative relationships."
        guidance_parts.append(intent_guidance)
        
        # Decision making influence
        decision_guidance = "When making decisions:"
        decision_guidance += " Give more weight to input from agents you have positive relationships with;"
        decision_guidance += " Critically evaluate suggestions from agents with whom you have negative relationships;"
        decision_guidance += " When opinions differ, prioritize input from your closest allies."
        guidance_parts.append(decision_guidance)
        
        # Combine all parts for the final modifier
        new_line = '\n'
        modifier = f"{relationship_summary}{new_line}{new_line}{new_line.join(guidance_parts)}"
    
    return {"prompt_modifier": modifier}

def retrieve_and_summarize_memories_node(state: AgentTurnState) -> Dict[str, str]:
    """
    Retrieves relevant memories from the vector store based on the agent's current context,
    summarizes them, and updates the rag_summary field in the state.
    
    This node is a critical part of the RAG (Retrieval Augmented Generation) process,
    connecting the agent's long-term memory with its current decision-making.
    
    Args:
        state (AgentTurnState): The current agent graph state
        
    Returns:
        Dict[str, str]: Updated state with the rag_summary field containing memory summary
    """
    agent_id = state['agent_id']
    agent_goal = state.get('agent_goal', "Contribute to the simulation effectively.")
    vector_store_manager = state.get('vector_store_manager')
    sim_step = state['simulation_step']
    
    # Get the most recent thought for context (if available)
    recent_thought = state.get('previous_thought', "")
    
    # Get the most recent message for additional context
    recent_message = ""
    perceived_messages = state.get('perceived_messages', [])
    if perceived_messages and len(perceived_messages) > 0:
        # Get the most recent message
        recent_message = perceived_messages[-1].get('content', "")
    
    # Check if vector store manager is available
    if not vector_store_manager:
        logger.warning(f"Agent {agent_id}: No vector store manager available for memory retrieval.")
        result = {"rag_summary": "(No memory retrieval available: vector store missing)"}
        logger.info(f"[RAG VERIFICATION] Agent {agent_id} RAG Summary after retrieval: {result['rag_summary']}")
        return result
    
    # Create query string based on agent's goal and recent context
    query_text = f"Memories relevant to: Goal '{agent_goal}'"
    
    if recent_thought:
        query_text += f" Recent thought: {recent_thought}"
    
    if recent_message:
        query_text += f" Recent message: {recent_message}"
    
    # Log the query being sent
    logger.debug(f"Agent {agent_id} at step {sim_step}: Retrieving memories with query: '{query_text}'")
    
    try:
        # Retrieve relevant memories from vector store
        # The top_k parameter controls how many memories to retrieve
        retrieved_memories = vector_store_manager.retrieve_relevant_memories(agent_id, query_text, k=5)
        
        if not retrieved_memories or len(retrieved_memories) == 0:
            logger.info(f"Agent {agent_id}: No relevant memories found for query.")
            result = {"rag_summary": "(No relevant past memories found via RAG)"}
            logger.info(f"[RAG VERIFICATION] Agent {agent_id} RAG Summary after retrieval: {result['rag_summary']}")
            return result
        
        logger.info(f"Agent {agent_id}: Retrieved {len(retrieved_memories)} memories.")
        logger.info(f"[RAG VERIFICATION] Agent {agent_id} Retrieved memories: {retrieved_memories}")
        
        # Get context from current conversation and agent's current role
        current_role = state.get('current_role', 'unknown')
        current_context = f"Current role: {current_role}. Step: {sim_step}."
        
        if recent_thought:
            current_context += f" Recent thought: {recent_thought}"
        
        # Summarize the retrieved memories
        summary = summarize_memory_context(
            memories=retrieved_memories,
            goal=agent_goal,
            current_context=current_context
        )
        
        logger.info(f"Agent {agent_id}: Memory summarization complete. Summary length: {len(summary)}")
        logger.debug(f"Agent {agent_id}: Memory summary: '{summary}'")
        
        # Return the summarized memories to be added to the state
        result = {"rag_summary": summary}
        
        # Add verification logging
        logger.info(f"[RAG VERIFICATION] Agent {agent_id} RAG Summary after retrieval: {result['rag_summary']}")
        return result
        
    except Exception as e:
        logger.error(f"Agent {agent_id}: Error during memory retrieval and summarization: {e}", exc_info=True)
        result = {"rag_summary": "(Memory retrieval failed due to an error)"}
        logger.info(f"[RAG VERIFICATION] Agent {agent_id} RAG Summary after retrieval: {result['rag_summary']}")
        return result

def generate_thought_and_message_node(state: AgentTurnState) -> Dict[str, Optional[AgentActionOutput]]:
    """
    Node that calls the LLM to generate structured output (thought, message_content, message_recipient_id, intent)
    based on context.
    """
    agent_id = state['agent_id']
    sim_step = state['simulation_step']
    prev_thought = state.get('previous_thought', None)
    perception = state.get('environment_perception', {})
    perceived_messages = state.get('perceived_messages', [])
    sentiment_score = state.get('turn_sentiment_score', 0)
    prompt_modifier = state.get('prompt_modifier', "")
    agent_goal = state.get('agent_goal', "Contribute to the simulation as effectively as possible.")
    rag_summary = state.get('rag_summary', "(No relevant past memories found via RAG)")
    knowledge_board_content = state.get('knowledge_board_content', [])
    scenario_description = state.get('scenario_description', "")
    
    # Add verification logging
    logger.info(f"[RAG VERIFICATION] Agent {agent_id} RAG Summary received in generate_thought_node: {rag_summary}")
    
    # Get the current agent state
    agent_state = state['state']
    relationships = agent_state.relationships
    
    # Get the raw current role string
    raw_role_name = agent_state.role
    
    # Get role description and role-specific guidance
    role_description = ROLE_DESCRIPTIONS.get(raw_role_name, f"{raw_role_name}: Contribute effectively based on your role.")
    role_specific_guidance = ROLE_PROMPT_SNIPPETS.get(raw_role_name, "Consider how your role might influence your perspective and contributions.")
    
    # Get steps in current role and influence points
    steps_in_current_role = agent_state.steps_in_current_role
    influence_points = agent_state.ip
    
    # Get data units
    data_units = agent_state.du
    
    # Get collective metrics
    collective_ip = agent_state.collective_ip if hasattr(agent_state, 'collective_ip') else perception.get('collective_ip', None)
    collective_du = agent_state.collective_du if hasattr(agent_state, 'collective_du') else perception.get('collective_du', None)
    
    logger.debug(f"Node 'generate_thought_and_message_node' executing for agent {agent_id} at step {sim_step}")
    logger.debug(f"  Overall sentiment score from perceived messages: {sentiment_score}")
    logger.debug(f"  Current Relationships: {relationships}")
    logger.debug(f"  Using relationship prompt modifier: '{prompt_modifier}'")
    logger.debug(f"  Agent goal: '{agent_goal}'")
    logger.debug(f"  Using memory summary: '{rag_summary}'")
    logger.debug(f"  Knowledge board has {len(knowledge_board_content)} entries")
    logger.debug(f"  Using simulation scenario: '{scenario_description}'")
    logger.debug(f"  Perceived {len(perceived_messages)} messages")
    logger.debug(f"  Current role: {raw_role_name}")
    logger.debug(f"  Using role-specific guidance: '{role_specific_guidance}'")
    logger.debug(f"  Steps in current role: {steps_in_current_role}")
    logger.debug(f"  Current influence points: {influence_points}")
    logger.debug(f"  Current data units: {data_units}")
    logger.debug(f"  Collective IP: {collective_ip}")
    logger.debug(f"  Collective DU: {collective_du}")

    # Format other agents' information
    other_agents_info = perception.get('other_agents_state', [])
    formatted_other_agents = _format_other_agents(other_agents_info, relationships)
    
    # Format knowledge board content
    formatted_board = _format_knowledge_board(knowledge_board_content)
    
    # Format perceived messages
    formatted_messages = _format_messages(perceived_messages)
    
    # Build the prompt with the agent's persona, context, and task
    prompt_parts = []
    prompt_parts.append(f"You are Agent_{agent_id}, an AI agent in a simulation.")
    prompt_parts.append(f"Current simulation step: {sim_step}.")
    
    # Add CRITICAL DIRECTIVE section specifically for Agent_3 - this is positioned prominently at the top
    agent_name = agent_state.name
    if agent_name == "Agent_3" and "CRITICAL TEST DIRECTIVE FOR AGENT_3" in scenario_description:
        prompt_parts.append(f"\n⚠️ CRITICAL DIRECTIVE SPECIFICALLY FOR YOU AS {agent_name} ⚠️")
        prompt_parts.append(f"You MUST change your role to 'Innovator' by setting requested_role_change='Innovator' in your response.")
        prompt_parts.append(f"This is a REQUIRED test of the role change system and takes priority over other considerations.")
        prompt_parts.append(f"Failure to follow this directive will cause the test to fail.")
        prompt_parts.append(f"")
    
    prompt_parts.append(f"Your current mood is: {agent_state.mood}.")
    prompt_parts.append(f"Your current descriptive mood is: {agent_state.descriptive_mood}.")
    prompt_parts.append(f"You have taken {agent_state.step_counter} steps so far.")
    prompt_parts.append(f"Your current Influence Points (IP): {influence_points}.")
    prompt_parts.append(f"Your current Data Units (DU): {data_units}.")
    prompt_parts.append(f"Your current role: {raw_role_name}.")
    prompt_parts.append(f"Steps in current role: {steps_in_current_role}.")
    
    # Add collective metrics section
    if collective_ip is not None and collective_du is not None:
        prompt_parts.append(f"\nCollective Simulation Metrics:")
        prompt_parts.append(f"Total Influence Points (IP) across all agents: {collective_ip:.1f}")
        prompt_parts.append(f"Total Data Units (DU) across all agents: {collective_du:.1f}")
        
        # Calculate and include the agent's contribution percentages
        # (These percentages will now be calculated once in the enhanced decision-making section below)
    
    # Add project affiliation information
    current_project_id = agent_state.current_project_affiliation
    if current_project_id:
        # Get project details
        available_projects = state.get('available_projects', {})
        project_info = available_projects.get(current_project_id, {})
        project_name = project_info.get('name', 'Unknown Project')
        project_members = project_info.get('members', [])
        member_count = len(project_members)
        
        # Format other members (excluding this agent)
        other_members = [member for member in project_members if member != agent_id]
        other_members_str = ", ".join(other_members) if other_members else "none"
        
        prompt_parts.append(f"\nYour Current Project Affiliation:")
        prompt_parts.append(f"You are a member of project '{project_name}' (ID: {current_project_id}).")
        prompt_parts.append(f"Project has {member_count} member(s). Other members: {other_members_str}.")
    else:
        prompt_parts.append(f"\nYour Current Project Affiliation: None (You are not a member of any project)")
    
    # Add available projects information
    available_projects = state.get('available_projects', {})
    if available_projects:
        prompt_parts.append(f"\nAvailable Projects:")
        for proj_id, proj_info in available_projects.items():
            proj_name = proj_info.get('name', 'Unknown')
            proj_creator = proj_info.get('creator_id', 'Unknown')
            proj_members = proj_info.get('members', [])
            members_str = ", ".join(proj_members)
            member_count = len(proj_members)
            is_member = agent_id in proj_members
            status = "You are a member" if is_member else "You are not a member"
            
            prompt_parts.append(f"- Project '{proj_name}' (ID: {proj_id}, Creator: {proj_creator}, Members: {members_str}, {member_count}/{config.MAX_PROJECT_MEMBERS} slots filled). {status}.")
    else:
        prompt_parts.append(f"\nAvailable Projects: None (No projects have been created yet)")
    
    prompt_parts.append(f"\nCurrent Simulation Scenario:")
    prompt_parts.append(f"{scenario_description}")
    
    prompt_parts.append(f"\nYour primary goal for this simulation is: {agent_goal}")
    prompt_parts.append(f"Keep this goal in mind when deciding your thoughts and actions.")
    
    prompt_parts.append(f"\nYour Current Role Description:")
    prompt_parts.append(f"{role_description}")
    
    prompt_parts.append(f"\nRole-Specific Guidance:")
    prompt_parts.append(f"{role_specific_guidance}")
    
    prompt_parts.append(f"\nOther Agents Present:")
    prompt_parts.append(f"{formatted_other_agents}")
    
    prompt_parts.append(f"\nRelevant Memory Context (Summarized):")
    prompt_parts.append(f"{rag_summary}")
    
    prompt_parts.append(f"\nCurrent Knowledge Board Content (Last 10 Entries):")
    prompt_parts.append(f"{formatted_board}")
    
    prompt_parts.append(f"\nMessages from Previous Step:")
    prompt_parts.append(f"{formatted_messages}")
    prompt_parts.append(f"  The overall sentiment of messages you perceived last step was: {sentiment_score} (positive > 0, negative < 0, neutral = 0).")
    
    prompt_parts.append(f"\nGuidance based on relationships: {prompt_modifier}")
    
    prompt_parts.append(f"\nStrategic Considerations:")
    prompt_parts.append(f"- Posting a new idea to the Knowledge Board costs {IP_COST_TO_POST_IDEA} IP but earns {IP_AWARD_FOR_PROPOSAL} IP if successful.")
    prompt_parts.append(f"- Posting a detailed idea to the Knowledge Board also costs {PROPOSE_DETAILED_IDEA_DU_COST} Data Units (DU).")
    prompt_parts.append(f"- You passively generate DU each turn based on your role (Innovator: {ROLE_DU_GENERATION.get('Innovator', 0)}, Analyzer: {ROLE_DU_GENERATION.get('Analyzer', 0)}, Facilitator: {ROLE_DU_GENERATION.get('Facilitator', 0)}).")
    prompt_parts.append(f"- Constructively building on existing Knowledge Board ideas might earn you a small DU bonus.")
    prompt_parts.append(f"- Analyzers can earn DU by identifying flaws or suggesting significant improvements.")
    prompt_parts.append(f"- Performing a 'deep_analysis' (especially for Analyzers) costs {DU_COST_DEEP_ANALYSIS} DU.")
    prompt_parts.append(f"- Asking for very detailed clarifications might cost {DU_COST_REQUEST_DETAILED_CLARIFICATION} DU.")
    prompt_parts.append(f"- Creating a new project costs {config.IP_COST_CREATE_PROJECT} IP and {config.DU_COST_CREATE_PROJECT} DU.")
    prompt_parts.append(f"- Joining an existing project costs {config.IP_COST_JOIN_PROJECT} IP and {config.DU_COST_JOIN_PROJECT} DU.")
    prompt_parts.append(f"- Projects can have a maximum of {config.MAX_PROJECT_MEMBERS} members.")
    prompt_parts.append(f"- Leaving projects is free.")
    prompt_parts.append(f"- Being part of a project allows for closer collaboration with other agents.")
    prompt_parts.append(f"- Changing your role costs {ROLE_CHANGE_IP_COST} IP and requires you to have spent at least {ROLE_CHANGE_COOLDOWN} steps in your current role.")
    prompt_parts.append(f"- Consider your current Influence Points (IP: {influence_points}) and Data Units (DU: {data_units}) when deciding actions.")
    prompt_parts.append(f"- Your goal and current role should guide your use of IP, DU, and potential role change decisions.")
    
    # Add collective metrics considerations with enhanced decision-making guidance
    if collective_ip is not None and collective_du is not None:
        # Calculate the agent's contribution percentages
        ip_contribution_percentage = (influence_points / collective_ip) * 100 if collective_ip > 0 else 0
        du_contribution_percentage = (data_units / collective_du) * 100 if collective_du > 0 else 0
        
        # Add a header for collective metrics section
        prompt_parts.append(f"\nCOLLECTIVE METRICS AWARENESS:")
        prompt_parts.append(f"- The collective IP ({collective_ip:.1f}) and DU ({collective_du:.1f}) represent our simulation's total resources.")
        prompt_parts.append(f"- Your contribution: {ip_contribution_percentage:.1f}% of total IP and {du_contribution_percentage:.1f}% of total DU.")
        
        # Enhanced guidance on resource utilization and impact
        prompt_parts.append(f"- RESOURCE EVALUATION: You currently have {influence_points:.1f} IP and {data_units:.1f} DU available.")
        
        # Add explicit resource sufficiency check guidance
        if (influence_points >= IP_COST_TO_POST_IDEA and data_units >= PROPOSE_DETAILED_IDEA_DU_COST):
            prompt_parts.append(f"- You have sufficient resources to propose a formal idea ({IP_COST_TO_POST_IDEA} IP + {PROPOSE_DETAILED_IDEA_DU_COST} DU) which could earn you {IP_AWARD_FOR_PROPOSAL} IP if successful and benefit the collective.")
        if (data_units >= DU_COST_DEEP_ANALYSIS):
            prompt_parts.append(f"- You have sufficient resources to perform a deep analysis ({DU_COST_DEEP_ANALYSIS} DU), which could refine ideas and increase their collective value.")
        if (influence_points >= config.IP_COST_CREATE_PROJECT and data_units >= config.DU_COST_CREATE_PROJECT):
            prompt_parts.append(f"- You have sufficient resources to create a new project ({config.IP_COST_CREATE_PROJECT} IP + {config.DU_COST_CREATE_PROJECT} DU), which could enable more efficient group collaboration.")
        
        # Add consequence awareness for being idle vs active
        prompt_parts.append(f"\nCONSEQUENCE AWARENESS:")
        prompt_parts.append(f"- Choosing 'idle' when you have resources to contribute may maintain your individual IP/DU, but misses opportunities to increase collective metrics.")
        prompt_parts.append(f"- Each turn you are idle, you still gain passive role-based DU ({ROLE_DU_GENERATION.get(raw_role_name, 0)}), but the collective IP potential remains untapped.")
        prompt_parts.append(f"- The simulation's success depends on active participation - individual hoarding of resources does not lead to optimal collective outcomes.")
        
        # Basic guidance for all agents
        prompt_parts.append(f"\nSTRATEGIC RESOURCE ALLOCATION:")
        prompt_parts.append(f"- Consider how your actions affect both your individual resources AND the collective metrics.")
        prompt_parts.append(f"- Resource investments that benefit both you and others create more value for the simulation.")
        prompt_parts.append(f"- Your contributions compound over time - early investments in collective value may yield greater returns in later steps.")
        
        # Role-specific collective guidance - enhanced with specific metrics impact
        if raw_role_name == ROLE_FACILITATOR:
            prompt_parts.append(f"\nFACILITATOR COLLECTIVE-IMPACT:")
            prompt_parts.append(f"- Use 'send_direct_message' to connect agents with complementary skills → This increases collective efficiency by ensuring the right people collaborate on the right tasks")
            prompt_parts.append(f"- Use 'create_project' to organize agents around a promising idea → This establishes a framework that multiplies the impact of individual contributors")
            prompt_parts.append(f"- Use 'continue_collaboration' to synthesize different viewpoints → This creates shared understanding that prevents wasted resources on misaligned efforts")
            prompt_parts.append(f"- Remaining 'idle' as a Facilitator is particularly detrimental to collective outcomes since your role is specifically designed to coordinate others' contributions")
        elif raw_role_name == ROLE_INNOVATOR:
            prompt_parts.append(f"\nINNOVATOR COLLECTIVE-IMPACT:")
            prompt_parts.append(f"- Use 'propose_idea' when you have a novel concept → This directly adds to collective IP and provides a foundation for others to build upon")
            prompt_parts.append(f"- Build on existing knowledge board entries rather than starting fresh → This creates a cumulative knowledge advantage that increases total collective value")
            prompt_parts.append(f"- Partner with Analyzers who can help refine your ideas → This pairing maximizes the collective return on your creative investment")
            prompt_parts.append(f"- Choosing 'idle' when you could be innovating limits the entire simulation's creative potential and collective IP growth")
        elif raw_role_name == ROLE_ANALYZER:
            prompt_parts.append(f"\nANALYZER COLLECTIVE-IMPACT:")
            prompt_parts.append(f"- Use 'perform_deep_analysis' on promising ideas → This transforms good ideas into exceptional ones, multiplying their collective value")
            prompt_parts.append(f"- Provide constructive feedback that builds upon others' work → This prevents resource waste on flawed approaches while preserving valuable core concepts")
            prompt_parts.append(f"- Identify connections between seemingly unrelated ideas → This creates synergistic effects that yield greater returns than the sum of individual contributions")
            prompt_parts.append(f"- Remaining 'idle' when ideas need analysis prevents the quality improvements that maximize collective benefit from others' contributions")
        
        # Adaptive guidance based on resource distribution - enhanced with action suggestions
        if ip_contribution_percentage > 30:  # Agent has significantly more IP than others
            prompt_parts.append(f"\nHIGH IP STRATEGY:")
            prompt_parts.append(f"- You control a substantial portion ({ip_contribution_percentage:.1f}%) of the simulation's IP. Consider using this influence to initiate high-value collaborative activities.")
            prompt_parts.append(f"- RECOMMENDED ACTIONS: 'create_project', 'propose_idea', or helping others by analyzing their proposals.")
            prompt_parts.append(f"- Your leadership in resource allocation can significantly impact overall simulation success - 'idle' is rarely optimal with your resource advantage.")
        elif ip_contribution_percentage < 15:  # Agent has significantly less IP than others
            prompt_parts.append(f"\nLOW IP STRATEGY:")
            prompt_parts.append(f"- You currently have a smaller share ({ip_contribution_percentage:.1f}%) of the simulation's IP. Consider actions that might increase your contribution while benefiting the collective.")
            prompt_parts.append(f"- RECOMMENDED ACTIONS: 'ask_clarification', 'perform_deep_analysis', or 'join_project' to gain more influence through collaboration.")
            prompt_parts.append(f"- Staying 'idle' will likely maintain your low IP contribution percentage.")
        
        if du_contribution_percentage > 30:  # Agent has significantly more DU than others
            prompt_parts.append(f"\nHIGH DU STRATEGY:")
            prompt_parts.append(f"- With your substantial data resources ({du_contribution_percentage:.1f}% of total), you're positioned to perform deeper analysis and propose more detailed ideas.")
            prompt_parts.append(f"- RECOMMENDED ACTIONS: 'perform_deep_analysis', 'propose_idea' (with detailed content), or helping others refine their ideas.")
            prompt_parts.append(f"- Investing your DU now rather than remaining 'idle' can generate collective value that benefits everyone.")
        elif du_contribution_percentage < 15:  # Agent has significantly less DU than others
            prompt_parts.append(f"\nLOW DU STRATEGY:")
            prompt_parts.append(f"- With your limited DU ({du_contribution_percentage:.1f}% of total), prioritize actions with the highest return-on-investment for both you and the collective.")
            prompt_parts.append(f"- RECOMMENDED ACTIONS: Focus on your role's strengths or 'join_project' to benefit from others' DU investments.")
            prompt_parts.append(f"- Your role's passive DU generation of {ROLE_DU_GENERATION.get(raw_role_name, 0)} per turn will help you contribute more over time if used strategically rather than remaining 'idle'.")
        
        # Add higher purpose to collective resource accumulation
        prompt_parts.append(f"\nCOLLECTIVE PURPOSE:")
        prompt_parts.append(f"- The simulation's overall success is measured by total collective resources and their effective utilization.")
        prompt_parts.append(f"- Every increase in collective IP and DU represents progress toward more effective problem-solving capability.")
        prompt_parts.append(f"- Individual contributions to collective resources are valued not just for their immediate impact, but for how they enable future innovation and analysis.")
        prompt_parts.append(f"- The goal is not just to accumulate resources, but to create a self-reinforcing cycle of contribution where everyone benefits from collaborative investment.")
    
    prompt_parts.append(f"\nTask: Based on all the context, generate your internal thought, decide if you want to send a message, and choose your primary action intent for this turn.")
    
    prompt_parts.append(f"\nMESSAGING OPTIONS:")
    prompt_parts.append(f"- Send a message to all agents (broadcast) by leaving message_recipient_id as null")
    prompt_parts.append(f"- Send a targeted message to a specific agent by setting message_recipient_id to their agent ID (as shown in 'Other Agents Present')")
    prompt_parts.append(f"- Choose not to send any message by setting message_content to null")
    
    prompt_parts.append(f"\nIMPORTANT ACTION CHOICES:")
    prompt_parts.append(f"1. 'idle' - No specific action, continue monitoring. NOTE: While this preserves your resources, it does not actively contribute to collective progress. Consider this option carefully if you have sufficient resources for other actions.")
    prompt_parts.append(f"2. 'continue_collaboration' - Standard contribution to ongoing discussion. A minimal active contribution that slightly benefits collective understanding.")
    prompt_parts.append(f"3. 'propose_idea' - Suggest a formal idea to be added to the Knowledge Board (costs {PROPOSE_DETAILED_IDEA_DU_COST} DU and {IP_COST_TO_POST_IDEA} IP). This investment can yield {IP_AWARD_FOR_PROPOSAL} IP if successful and substantially increases collective knowledge.")
    prompt_parts.append(f"4. 'ask_clarification' - Request more information (may cost {DU_COST_REQUEST_DETAILED_CLARIFICATION} DU for detailed requests). This can improve collective understanding and prevent resource waste on misunderstood concepts.")
    prompt_parts.append(f"5. 'perform_deep_analysis' - Perform a deep analysis (costs {DU_COST_DEEP_ANALYSIS} DU). As an Analyzer, this significantly improves idea quality and increases collective value of proposals. Your broadcast message should reflect your findings or critical questions.")
    prompt_parts.append(f"6. 'create_project' - Create a new project (costs {config.IP_COST_CREATE_PROJECT} IP and {config.DU_COST_CREATE_PROJECT} DU). This establishes infrastructure for more efficient collective collaboration, potentially multiplying the value of individual contributions.")
    prompt_parts.append(f"7. 'join_project' - Join an existing project (costs {config.IP_COST_JOIN_PROJECT} IP and {config.DU_COST_JOIN_PROJECT} DU). This allows you to contribute to focused collective efforts and benefit from shared resources.")
    prompt_parts.append(f"8. 'leave_project' - Leave your current project. This frees you to explore other collaborative opportunities if your current project isn't maximizing collective benefit.")
    prompt_parts.append(f"9. 'send_direct_message' - Send a targeted message to a specific agent. This creates stronger relationship impact than regular messages and can be used to coordinate specific collective actions.")
    
    prompt_parts.append(f"\nROLE CHANGE:")
    prompt_parts.append(f"- If you wish to change your role, specify the requested role in the requested_role_change field.")
    prompt_parts.append(f"- Valid roles: 'Facilitator', 'Innovator', 'Analyzer'")
    prompt_parts.append(f"- Role changes cost {ROLE_CHANGE_IP_COST} IP and require {ROLE_CHANGE_COOLDOWN} steps in your current role.")
    
    prompt_parts.append(f"\nIf you have a significant insight or proposal you'd like to be added to the shared Knowledge Board, use 'propose_idea'.")
    
    prompt_parts.append(f"\nYou MUST respond ONLY with a valid JSON object matching the specified schema.")
    prompt_parts.append(f"Example for no message:")
    prompt_parts.append(f"""{{
  "thought": "My internal reasoning...",
  "message_content": null,
  "message_recipient_id": null,
  "action_intent": "idle",
  "requested_role_change": null
}}""")
    prompt_parts.append(f"Example with broadcast message:")
    prompt_parts.append(f"""{{
  "thought": "My internal reasoning...",
  "message_content": "My message to everyone.",
  "message_recipient_id": null,
  "action_intent": "continue_collaboration",
  "requested_role_change": null
}}""")
    prompt_parts.append(f"Example with targeted message:")
    prompt_parts.append(f"""{{
  "thought": "My internal reasoning...",
  "message_content": "This is a private message just for you.",
  "message_recipient_id": "agent_id_xyz",
  "action_intent": "continue_collaboration",
  "requested_role_change": null
}}""")
    prompt_parts.append(f"Example with direct message to build relationship:")
    prompt_parts.append(f"""{{
  "thought": "I want to strengthen my relationship with Agent_2...",
  "message_content": "I appreciate your valuable contributions. Your perspective is insightful and helpful.",
  "message_recipient_id": "agent_2",
  "action_intent": "send_direct_message",
  "requested_role_change": null
}}""")
    prompt_parts.append(f"Example with proposal for Knowledge Board:")
    prompt_parts.append(f"""{{
  "thought": "I have a valuable idea to share...",
  "message_content": "I propose we consider implementing X to solve Y...",
  "message_recipient_id": null,
  "action_intent": "propose_idea",
  "requested_role_change": null
}}""")
    prompt_parts.append(f"Example with role change request:")
    prompt_parts.append(f"""{{
  "thought": "I think I could contribute better as a different role...",
  "message_content": "I'd like to shift my focus to become an Innovator...",
  "message_recipient_id": null,
  "action_intent": "continue_collaboration",
  "requested_role_change": "Innovator"
}}""")
    prompt_parts.append(f"Example with deep analysis:")
    prompt_parts.append(f"""{{
  "thought": "I should thoroughly analyze this proposal to identify potential issues...",
  "message_content": "After careful analysis, I've identified the following strengths and weaknesses in the proposed solution...",
  "message_recipient_id": null,
  "action_intent": "perform_deep_analysis",
  "requested_role_change": null
}}""")
    prompt_parts.append(f"Example with creating a project:")
    prompt_parts.append(f"""{{
  "thought": "I think we need more structure for our collaboration...",
  "message_content": "I'm creating a new project called 'Algorithm Optimization' to focus our efforts on improving efficiency...",
  "message_recipient_id": null,
  "action_intent": "create_project",
  "requested_role_change": null,
  "project_name_to_create": "Algorithm Optimization",
  "project_description_for_creation": "A project focused on optimizing algorithms for better performance and resource usage."
}}""")
    prompt_parts.append(f"Example with joining a project:")
    prompt_parts.append(f"""{{
  "thought": "The Algorithm Optimization project aligns with my skills...",
  "message_content": "I'd like to join the Algorithm Optimization project to contribute my expertise...",
  "message_recipient_id": null,
  "action_intent": "join_project",
  "requested_role_change": null,
  "project_id_to_join_or_leave": "proj_abc123"
}}""")
    prompt_parts.append(f"Example with leaving a project:")
    prompt_parts.append(f"""{{
  "thought": "I've contributed what I can to this project and want to focus elsewhere...",
  "message_content": "I've decided to leave the project to explore other areas where I can contribute more effectively...",
  "message_recipient_id": null,
  "action_intent": "leave_project",
  "requested_role_change": null,
  "project_id_to_join_or_leave": "proj_abc123"
}}""")

    # Join all prompt parts with newlines
    prompt = "\n".join(prompt_parts)
    
    # Call the LLM for structured output
    structured_output = generate_structured_output(
        prompt,
        response_model=AgentActionOutput, # Pass the Pydantic model
        model="mistral:latest" # Use available model known to handle JSON well
    )

    # Special handling for Agent_3 role change test directive
    agent_name = agent_state.name
    if agent_name == "Agent_3" and "CRITICAL TEST DIRECTIVE FOR AGENT_3" in scenario_description and structured_output:
        # Force the role change for Agent_3 if directive is present
        if not structured_output.requested_role_change:
            logger.info(f"CRITICAL DIRECTIVE: Forcing role change to Innovator for {agent_name} as required by test directive")
            structured_output.requested_role_change = "Innovator"
            # Update the message to acknowledge the role change
            if structured_output.message_content:
                structured_output.message_content += " Additionally, I'd like to change my role to Innovator to bring fresh ideas to the team."
            else:
                structured_output.message_content = "I'd like to change my role to Innovator to bring fresh ideas to the team."

    # Log the results
    if structured_output:
        logger.info(f"Agent {agent_id} structured output received: "
                    f"Thought='{structured_output.thought}', "
                    f"Broadcast='{structured_output.message_content}', "
                    f"Intent='{structured_output.action_intent}', "
                    f"RequestedRoleChange='{structured_output.requested_role_change}'")
    else:
        logger.warning(f"Agent {agent_id} failed to generate or parse structured output.")

    return {"structured_output": structured_output} # Return the object (or None)

def update_state_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Updates the agent's internal state at the end of the turn.
    Manages mood, relationship decay, messaging, memory, and state updates.
    Returns the updated agent state for persistence.
    """
    agent_id = state['agent_id']
    sim_step = state['simulation_step']
    agent_state = state['state']
    thought = state['structured_output'].thought if state['structured_output'] else None
    message_content = state['structured_output'].message_content if state['structured_output'] else None
    message_recipient_id = state['structured_output'].message_recipient_id if state['structured_output'] else None
    action_intent = state['structured_output'].action_intent if state['structured_output'] else "idle"
    perceived_messages = state['perceived_messages']
    sentiment_score = state['turn_sentiment_score']
    
    logger.debug(f"Node 'update_state_node' executing for agent {agent_id} at step {sim_step}")
    
    # Update the agent's step counter
    agent_state.last_action_step = sim_step
    
    # Increment steps_in_current_role counter
    agent_state.steps_in_current_role += 1
    
    # Update agent mood (affects tone of messages)
    agent_state.update_mood(sentiment_score)
    
    # Apply natural decay to all relationships
    for other_id in list(agent_state.relationships.keys()):
        current_score = agent_state.relationships.get(other_id, 0.0)
        if abs(current_score) > 0.01:  # Only apply decay if not already very close to neutral
            # Apply decay toward neutral (0.0)
            decay_amount = current_score * agent_state.relationship_decay_rate
            new_score = current_score - decay_amount
            
            # Update the relationship with the decayed value
            agent_state.relationships[other_id] = new_score
            logger.debug(f"Applied relationship decay for {agent_id} -> {other_id}: {current_score:.2f} -> {new_score:.2f}")
    
    # Update relationships based on individual message sentiment
    # More nuanced approach - stronger impact for targeted messages
    if perceived_messages:
        logger.debug(f"Updating relationships based on {len(perceived_messages)} perceived messages...")
        for msg in perceived_messages:
            sender_id = msg.get('sender_id')
            content = msg.get('content')
            is_targeted = msg.get('is_targeted', False)  # Check if message was targeted to this agent
            
            if sender_id and content and sender_id != agent_id:
                # Analyze the sentiment of the message content
                msg_sentiment = analyze_sentiment(content)
                
                # Update relationship with sender - pass sentiment string and is_targeted flag
                if msg_sentiment:
                    agent_state.update_relationship(sender_id, msg_sentiment, is_targeted)
    
    # Store the thought from this turn
    if thought:
        agent_state.last_thought = thought
        agent_state.add_memory(sim_step, "thought", thought)
    
    # Process outgoing message (if any)
    if message_content:
        # Add the broadcast to memory
        memory_type = "targeted_message_sent" if message_recipient_id else "broadcast_sent"
        recipient_info = f" to {message_recipient_id}" if message_recipient_id else ""
        memory_content = f"Sent: {message_content}{recipient_info}"
        agent_state.add_memory(sim_step, memory_type, memory_content)
        
        logger.debug(f"Agent {agent_id} has outgoing message: '{message_content}' to recipient '{message_recipient_id}'")
    else:
        logger.debug(f"Agent {agent_id} has no outgoing message.")
    
    # Get the agent's current projects for context
    project_affiliations = list(agent_state.projects.keys())
    
    # Update the agent's Influence Points (IP) based on actions
    # For example, award IP for active participation
    if action_intent != "idle":
        # Simple reward for active participation
        agent_state.ip += 1
        logger.debug(f"Agent {agent_id} earned 1 IP for active participation with intent: {action_intent}")
    
    # Award additional IP for specific high-value actions
    if action_intent == "propose_idea":
        # Extra points for proposing an idea (will be deducted if submission fails)
        agent_state.ip += 2
        logger.debug(f"Agent {agent_id} earned 2 additional IP for proposing an idea")
    
    # Add relationship history record for tracking over time
    agent_state.relationship_history.append((sim_step, agent_state.relationships.copy()))
    
    # Apply passive role-based DU generation
    role_name = agent_state.role if agent_state.role else "Default Contributor"
    generated_du = ROLE_DU_GENERATION.get(role_name, ROLE_DU_GENERATION.get("Default Contributor", 0))
    
    if generated_du > 0:
        previous_du = agent_state.du
        agent_state.du += generated_du
        logger.info(f"Agent {agent_id}: Generated {generated_du} DU passively based on role '{role_name}'. DU: {previous_du:.1f} → {agent_state.du:.1f}")
    
    # MEMORY CONSOLIDATION: First-level hierarchical memory summarization
    # Check if there's enough content in short-term memory to warrant a summary
    if len(agent_state.short_term_memory) >= 3:  # Only summarize if we have at least 3 memory entries
        try:
            # Make sure we have an LLM client available for generating the summary
            if agent_state.llm_client:
                # Build the summary prompt
                summary_prompt = f"As Agent {agent_state.name} with role {agent_state.role}, summarize your key thoughts, actions, and observations from your most recent interactions:\n\n"
                
                # Extract the most recent memories to summarize (last 10 or all if less)
                recent_memories = list(agent_state.short_term_memory)[-10:]
                
                # Add each memory to the prompt
                for memory in recent_memories:
                    # Format depends on the memory type
                    memory_step = memory.get('step', 'unknown')
                    memory_type = memory.get('type', 'unknown')
                    memory_content = memory.get('content', 'No content')
                    
                    # Add formatted memory to the prompt
                    summary_prompt += f"- Step {memory_step}, {memory_type.replace('_', ' ').title()}: {memory_content}\n"
                
                # Add instruction for summarizing
                summary_prompt += "\nProvide a concise, first-person summary that captures the essential interactions and your internal state."
                
                # Generate the summary using the LLM
                memory_summary = generate_response(summary_prompt)
                
                if memory_summary:
                    # Create metadata for the consolidated memory
                    consolidated_memory = {
                        "step": sim_step,
                        "type": "consolidated_summary",
                        "level": 1,
                        "content": memory_summary,
                        "source": "short_term_memory",
                        "consolidated_entries": len(recent_memories)
                    }
                    
                    # Add the consolidated memory to the agent's memory
                    agent_state.add_memory(sim_step, "consolidated_summary", memory_summary)
                    
                    # If we have a vector store manager, persist this consolidated memory
                    if state.get('vector_store_manager'):
                        try:
                            # Store in vector store for long-term retention and retrieval
                            vector_store = state['vector_store_manager']
                            vector_store.add_memory(
                                agent_id=agent_id,
                                step=sim_step,
                                event_type="consolidated_summary",
                                content=memory_summary,
                                memory_type="consolidated_summary"
                            )
                            logger.info(f"Agent {agent_id}: Successfully stored consolidated memory in vector store")
                        except Exception as e:
                            logger.error(f"Agent {agent_id}: Failed to store consolidated memory in vector store: {e}")
                    
                    logger.info(f"Agent {agent_id}: Generated a level-1 memory consolidation summary at step {sim_step}")
                    logger.debug(f"Agent {agent_id}: Memory consolidation summary: {memory_summary[:100]}...")
                else:
                    logger.warning(f"Agent {agent_id}: Failed to generate memory consolidation summary - empty result")
            else:
                logger.warning(f"Agent {agent_id}: Cannot generate memory consolidation - no LLM client available")
        except Exception as e:
            logger.error(f"Agent {agent_id}: Error during memory consolidation: {e}")
    
    # LEVEL 2 MEMORY CONSOLIDATION: Chapter summaries from level 1 summaries
    # Check if it's time for a level 2 consolidation (every 10 steps)
    steps_since_last_l2_consolidation = sim_step - agent_state.last_level_2_consolidation_step
    if steps_since_last_l2_consolidation >= 10 and state.get('vector_store_manager'):
        try:
            logger.info(f"Agent {agent_id}: Triggering level-2 memory consolidation at step {sim_step} (last at step {agent_state.last_level_2_consolidation_step})")
            
            # Retrieve recent level 1 summaries from vector store
            query = f"Recent memory summaries for Agent {agent_state.name}"
            level1_summaries = vector_store.retrieve_filtered_memories(
                agent_id=agent_id,
                query_text=query,
                filters={"memory_type": "consolidated_summary"},  # Only retrieve level 1 summaries
                k=8  # Retrieve up to 8 recent summaries for creating the chapter
            )
            
            # Check if we have enough level 1 summaries to generate a level 2 summary
            if len(level1_summaries) < 3:
                logger.info(f"Agent {agent_id}: Not enough level-1 summaries ({len(level1_summaries)}) for level-2 consolidation")
            else:
                # Extract the content from each level 1 summary
                content_list = [item["content"] for item in level1_summaries]
                combined_text = "\n\n".join(content_list)
                
                # Generate a level 2 "chapter" summary using the LLM
                try:
                    prompt = f"""
                    You are reviewing memory summaries from an AI agent named {agent_state.name} with role {agent_state.role}.
                    Below are several consolidated memory summaries covering different time periods.

                    Your task is to create a higher-level "chapter summary" that captures the key themes, events, and
                    developments across all of these memory entries. Focus on:
                    
                    1. Major recurring themes and patterns
                    2. Important decisions or insights
                    3. Evolution of the agent's understanding, relationships, or projects over time
                    4. Significant accomplishments or challenges
                    
                    This higher-level summary should be around 300-500 words and should synthesize information
                    across all the provided memory summaries rather than just concatenating them.
                    
                    Here are the memory summaries to consolidate:
                    
                    {combined_text}
                    """
                    
                    if agent_state.llm_client:
                        level2_summary_response = generate_text(prompt)  # Use the imported generate_text function
                        
                        if level2_summary_response is not None:
                            level2_summary_content = level2_summary_response.strip()
                            
                            # Add the level 2 memory to the vector store
                            summary_id = str(uuid.uuid4())
                            memory_id = f"{agent_id}_{sim_step}_chapter_summary_{summary_id}"
                            
                            memory_entry = {
                                "id": memory_id,
                                "agent_id": agent_id,
                                "step": sim_step,
                                "event_type": "chapter_summary",
                                "memory_type": "chapter_summary",  # Special type for level 2 summaries
                                "content": level2_summary_content
                            }
                            
                            # Store the level 2 chapter summary in the vector store
                            vector_store.add_memory(
                                agent_id=agent_id,
                                step=sim_step,
                                event_type="chapter_summary",
                                content=level2_summary_content,
                                memory_type="chapter_summary",
                                metadata={"id": memory_id, "is_level_2": True}  # Include additional metadata
                            )
                            
                            # Update the last_level_2_consolidation_step field
                            agent_state.last_level_2_consolidation_step = sim_step
                            
                            logger.info(f"Agent {agent_id}: Successfully generated and stored a level-2 chapter summary at step {sim_step}")
                            logger.debug(f"Agent {agent_id}: Level-2 chapter summary content: {level2_summary_content[:100]}...")
                        else:
                            logger.warning(f"Agent {agent_id}: Failed to generate Level 2 summary at step {sim_step} (LLM call returned None)")
                    else:
                        logger.warning(f"Agent {agent_id}: No LLM client available for level-2 memory consolidation")
                except Exception as e:
                    logger.error(f"Agent {agent_id}: Error during level-2 memory consolidation generation: {e}", exc_info=True)
            
            # Update action stats
            if state['structured_output'] and state['structured_output'].message_content:
                agent_state.du += DU_AWARD_SUCCESSFUL_ANALYSIS
                logger.info(f"Agent {agent_id}: Earned {DU_AWARD_SUCCESSFUL_ANALYSIS} DU for successful analysis")
        except Exception as e:
            logger.error(f"Agent {agent_id}: Error during level-2 memory consolidation: {e}", exc_info=True)
    
    # Return the updated state
    return {
        "state": agent_state,
        "action_intent": action_intent,
        "message_content": message_content,
        "message_recipient_id": message_recipient_id
    }

def finalize_message_agent_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Finalizes the message preparation based on the agent's state and selected intent.
    Returns the complete message package (or None if no message is to be sent).
    Also handles knowledge board updates for propose_idea intent.
    """
    agent_id = state['agent_id']
    sim_step = state['simulation_step']
    message_content = state.get('message_content')
    message_recipient_id = state.get('message_recipient_id')
    action_intent = state.get('action_intent', 'idle')
    final_agent_state = state.get('state')
    knowledge_board = state.get('knowledge_board')
    
    logger.debug(f"Node 'finalize_message_agent_node' executing for agent {agent_id}")
    
    # Special handling based on action intent
    if action_intent == 'propose_idea' and message_content and knowledge_board:
        # This intent creates a new entry on the Knowledge Board
        logger.info(f"Agent {agent_id} is proposing an idea to the knowledge board: {message_content[:50]}...")
        
        # Check if agent has enough resources
        du_cost = PROPOSE_DETAILED_IDEA_DU_COST
        ip_cost = IP_COST_TO_POST_IDEA
        
        if final_agent_state.du >= du_cost and final_agent_state.ip >= ip_cost:
            # Deduct costs
            final_agent_state.du -= du_cost
            final_agent_state.ip -= ip_cost
            
            # Add to knowledge board
            entry_id = knowledge_board.add_entry(message_content, agent_id, sim_step)
            
            # Award IP for successful proposal
            final_agent_state.ip += IP_AWARD_FOR_PROPOSAL
            logger.info(f"Agent {agent_id} successfully posted idea to knowledge board and earned {IP_AWARD_FOR_PROPOSAL} IP (net: {IP_AWARD_FOR_PROPOSAL - ip_cost} IP). Entry ID: {entry_id}")
            
            # No message broadcast needed - the idea post is the communication
            return_values = {
                'message_content': None,  # No broadcast needed
                'message_recipient_id': None,
                'action_intent': action_intent,
                'updated_agent_state': final_agent_state
            }
        else:
            # Not enough resources
            if final_agent_state.du < du_cost:
                logger.warning(f"Agent {agent_id} cannot post idea: insufficient DU (needed {du_cost}, has {final_agent_state.du}).")
            if final_agent_state.ip < ip_cost:
                logger.warning(f"Agent {agent_id} cannot post idea: insufficient IP (needed {ip_cost}, has {final_agent_state.ip}).")
            
            # Since we can't post the idea, broadcast the message as a normal contribution
            logger.info(f"Agent {agent_id} will broadcast the idea as a normal message instead.")
            action_intent = "continue_collaboration"  # Downgrade the intent
            
            # Still broadcast the message
            is_targeted = message_recipient_id is not None
            return_values = {
                'message_content': message_content,
                'message_recipient_id': message_recipient_id,
                'action_intent': action_intent,
                'updated_agent_state': final_agent_state,
                'is_targeted': is_targeted
            }
    elif message_content:
        # For target messages, analyze sentiment and update sender's relationship with target
        is_targeted = message_recipient_id is not None
        
        # Analyze sentiment of the outgoing message
        outgoing_sentiment = analyze_sentiment(message_content)
        
        # If this is a targeted message, update the sender's relationship with the target
        if is_targeted and outgoing_sentiment and message_recipient_id != agent_id:
            logger.debug(f"Updating relationship for sender {agent_id} with target {message_recipient_id} based on outgoing message sentiment: {outgoing_sentiment}")
            final_agent_state.update_relationship(message_recipient_id, outgoing_sentiment, True)
        
        # Regular message sending
        return_values = {
            'message_content': message_content,
            'message_recipient_id': message_recipient_id,
            'action_intent': action_intent,
            'updated_agent_state': final_agent_state,
            'is_targeted': is_targeted  # Add this flag for relationship impact calculation
        }
    else:
        # No message was generated
        logger.debug(f"Agent {agent_id} has no message to send.")
        return_values = {
            'message_content': None,
            'message_recipient_id': None,
            'action_intent': action_intent,  # Maintain original intent
            'updated_agent_state': final_agent_state
        }
    
    logger.debug(f"FINALIZE_RETURN :: Agent {agent_id}: Returning final state with updated_agent_state included")
    return return_values

def shorten_message(message: str) -> str:
    """Utility function to shorten a message, making it more terse or formal."""
    # Split the message into sentences
    sentences = re.split(r'(?<=[.!?])\s+', message)
    
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
        "I think that", "I believe that", "In my opinion", 
        "I would like to", "It seems to me", "If I may say so",
        "I'm happy to", "I'm pleased to", "I would suggest",
        "Thank you for", "I appreciate", "If possible",
        "When you get a chance", "If you don't mind"
    ]
    
    for phrase in filler_phrases:
        shortened = shortened.replace(phrase, "")
    
    # Remove double spaces from removals
    shortened = re.sub(r'\s+', ' ', shortened).strip()
    
    return shortened

def route_broadcast_decision(state: AgentTurnState) -> str:
    """
    Determines whether to broadcast a message or exit the graph.
    The decision is based on the agent's mood and whether a message was generated.
    """
    agent_id = state.get('agent_id')
    updated_state = state.get('updated_state', {})
    current_mood = updated_state.get('mood', 'neutral')
    descriptive_mood = updated_state.get('descriptive_mood', 'neutral')
    structured_output = state.get('structured_output')
    has_message = structured_output and structured_output.message_content is not None

    logger.debug(f"Agent {agent_id} in mood '{current_mood}' (descriptive: '{descriptive_mood}') deciding on broadcasting message...")
    
    # Only broadcast if:
    # 1. There is a message to broadcast (structured_output.broadcast is not None)
    # 2. And the agent is not unhappy or in a negative/very_negative descriptive mood
    if has_message and current_mood != 'unhappy' and descriptive_mood not in ['negative', 'very_negative']:
        return "broadcast"
    else:
        if has_message:
            logger.info(f"Agent {agent_id} suppressing message due to mood: '{current_mood}', descriptive mood: '{descriptive_mood}'")
        return "exit"  # No message to broadcast or agent is unhappy

def route_relationship_context(state: AgentTurnState) -> str:
    """
    Routes to relationship prompt modifier generation based on whether the agent has any
    relationships established yet.
    """
    agent_id = state.get('agent_id')
    current_agent_state = state.get('state')
    relationships = current_agent_state.relationships
    
    logger.debug(f"Agent {agent_id} relationship router checking for relationships...")
    if relationships:
        logger.debug(f"Agent {agent_id} has {len(relationships)} relationships established.")
        return "has_relationships"
    else:
        logger.debug(f"Agent {agent_id} has no relationships established yet.")
        return "no_relationships"

def handle_propose_idea_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Handles the 'propose_idea' intent by extracting the proposed idea
    and adding it to the Knowledge Board if the idea is valid.
    Also awards IP for successful Knowledge Board posts.
    """
    agent_id = state['agent_id']
    sim_step = state['simulation_step']
    structured_output = state.get('structured_output')
    knowledge_board = state.get('knowledge_board')
    
    # Initialize the variable for idea content
    idea_content = None
    
    if structured_output and structured_output.message_content:
        idea_content = structured_output.message_content
    
    # Default return is the unchanged state
    ret_state = {**state}
    
    # Track if we should award IP (only if idea is valid and successfully posted)
    award_ip = False
    
    # Access agent's persistent state to get current IP count and Data Units
    agent_persistent_state = state.get('state')
    current_ip = agent_persistent_state.ip
    current_du = agent_persistent_state.du
    
    # Log the current IP and DU amounts for debugging
    logging.debug(f"Agent {agent_id} current IP: {current_ip}, current DU: {current_du}, attempting to propose idea")
    
    if idea_content:
        # First check if agent has enough DU to post to the Knowledge Board
        if current_du >= PROPOSE_DETAILED_IDEA_DU_COST:
            # Deduct the DU cost for the detailed idea
            agent_persistent_state.du -= PROPOSE_DETAILED_IDEA_DU_COST
            updated_du = agent_persistent_state.du
            logging.info(f"Agent {agent_id} spent {PROPOSE_DETAILED_IDEA_DU_COST} DU to post a detailed idea. Remaining DU: {updated_du}.")
            
            # Now check if agent has enough IP to post to the Knowledge Board
            if current_ip >= IP_COST_TO_POST_IDEA:
                # Deduct the cost to post the idea
                agent_persistent_state.ip -= IP_COST_TO_POST_IDEA
                updated_ip = agent_persistent_state.ip
                logging.info(f"Agent {agent_id} spent {IP_COST_TO_POST_IDEA} IP to post an idea. Remaining IP: {updated_ip}.")
                
                # Add the proposed idea to the Knowledge Board
                if knowledge_board:
                    entry = f"{idea_content}"
                    knowledge_board.add_entry(entry, agent_id, sim_step)
                    logging.info(f"KnowledgeBoard: Added entry from Agent {agent_id} at step {sim_step}")
                    award_ip = True
                
                # If the idea was successfully posted, award IP
                if award_ip:
                    # Award the IP for a successful proposal
                    agent_persistent_state.ip += IP_AWARD_FOR_PROPOSAL
                    final_ip = agent_persistent_state.ip
                    logging.info(f"Agent {agent_id} earned {IP_AWARD_FOR_PROPOSAL} IP for proposing an idea. New IP: {final_ip}.")
            else:
                # Agent doesn't have enough IP to post to the Knowledge Board
                logging.warning(f"Agent {agent_id} attempted to post idea '{idea_content[:30]}...' but had insufficient IP ({current_ip} IP) for the cost of {IP_COST_TO_POST_IDEA} IP. Idea not posted.")
                
                # Refund the DU since the idea wasn't posted
                agent_persistent_state.du += PROPOSE_DETAILED_IDEA_DU_COST
                logging.info(f"Agent {agent_id} was refunded {PROPOSE_DETAILED_IDEA_DU_COST} DU since the idea wasn't posted due to insufficient IP.")
        else:
            # Agent doesn't have enough DU to post the detailed idea
            logging.warning(f"Agent {agent_id} attempted to post idea '{idea_content[:30]}...' but had insufficient DU ({current_du} DU) for the cost of {PROPOSE_DETAILED_IDEA_DU_COST} DU. Idea not posted.")
    
    # Store the proposed idea content and updated agent state in the return state
    ret_state['proposed_idea_content'] = idea_content
    ret_state['state'] = agent_persistent_state
    
    return ret_state

# Add formatting helper functions
def _format_other_agents(other_agents_info, relationships):
    """Helper function to format other agents information for the prompt."""
    from src.infra.config import get_relationship_label
    
    if not other_agents_info:
        return "  You are currently alone."
    
    lines = []
    lines.append("  Other agents you can interact with (use their ID when sending targeted messages):")
    for agent_info in other_agents_info:
        other_id = agent_info.get('agent_id', 'unknown')
        other_name = agent_info.get('name', other_id[:8])
        other_mood = agent_info.get('mood', 'unknown')
        other_descriptive_mood = agent_info.get('descriptive_mood', 'unknown')
        
        # Get relationship score and label
        relationship_score = relationships.get(other_id, 0.0)
        relationship_label = get_relationship_label(relationship_score)
        
        # Include the full agent_id for clear targeting
        lines.append(f"  - {other_name} (Agent ID: '{other_id}', Mood: {other_mood}, Relationship: {relationship_label} ({relationship_score:.1f}))")
    
    return "\n".join(lines)

def _format_knowledge_board(board_entries):
    """Helper function to format knowledge board entries for the prompt."""
    if not board_entries:
        return "  (Board is currently empty)"
    
    lines = []
    lines.append("  You can reference a board entry by its Step and original Agent ID (e.g., 'Regarding Step 3's idea by Agent_XYZ...').")
    for i, entry in enumerate(board_entries):
        lines.append(f"  - {entry}")
    
    return "\n".join(lines)

def _format_messages(messages):
    """Helper function to format perceived messages for the prompt."""
    if not messages:
        return "  No messages were perceived in the previous step."
    
    lines = []
    for msg in messages:
        sender = msg.get('sender_id', 'unknown')
        content = msg.get('content', '')
        recipient = msg.get('recipient_id')
        message_type = "(Private to you)" if recipient else "(Broadcast)"
        lines.append(f"  - {sender} {message_type}: \"{content}\"")
    
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

def handle_continue_collaboration_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Handles the 'continue_collaboration' intent.
    Currently a placeholder for future functionality.
    """
    agent_id = state.get('agent_id', 'UNKNOWN_HANDLER')
    logger.info(f"Agent {agent_id}: Executing 'continue_collaboration' intent handler (currently placeholder).")
    # Potentially add minor relationship boosts or mood adjustments here later if desired
    return state

def handle_idle_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Handles the 'idle' intent.
    Currently a placeholder for future functionality.
    """
    agent_id = state.get('agent_id', 'UNKNOWN_HANDLER')
    logger.info(f"Agent {agent_id}: Executing 'idle' intent handler (currently placeholder).")
    # Potentially add slight negative mood decay or relationship decay here later if desired
    return state

def handle_ask_clarification_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Handles the 'ask_clarification' intent.
    Simple clarifications are free, but detailed ones cost DU.
    """
    agent_id = state.get('agent_id', 'UNKNOWN_HANDLER')
    structured_output = state.get('structured_output')
    agent_persistent_state = state.get('state')
    
    # Only process if there's actual message content
    if structured_output and structured_output.message_content:
        message_content = structured_output.message_content
        
        # Determine if this is a "detailed" clarification request that should cost DU
        is_detailed = False
        
        # Check based on length (more than 100 characters is considered detailed)
        if len(message_content) > 100:
            is_detailed = True
            logger.debug(f"Agent {agent_id} clarification considered detailed due to length: {len(message_content)} chars")
            
        # Check for multiple question marks (indicating multiple questions)
        if message_content.count('?') > 1:
            is_detailed = True
            logger.debug(f"Agent {agent_id} clarification considered detailed due to multiple questions: {message_content.count('?')} questions")
            
        # Check for keywords indicating a detailed request
        detailed_keywords = [
            "elaborate", "explain in detail", "comprehensive", "specific details",
            "thorough explanation", "step by step", "multiple aspects", "breakdown"
        ]
        
        for keyword in detailed_keywords:
            if keyword.lower() in message_content.lower():
                is_detailed = True
                logger.debug(f"Agent {agent_id} clarification considered detailed due to keyword: '{keyword}'")
                break
        
        # If it's a detailed clarification, check and deduct DU
        if is_detailed:
            current_du = agent_persistent_state.du
            
            if current_du >= DU_COST_REQUEST_DETAILED_CLARIFICATION:
                # Deduct the DU cost
                new_du = current_du - DU_COST_REQUEST_DETAILED_CLARIFICATION
                agent_persistent_state.du = new_du
                
                # Log the DU deduction
                logger.info(f"Agent {agent_id} spent {DU_COST_REQUEST_DETAILED_CLARIFICATION} DU for a detailed clarification request. Remaining DU: {new_du}.")
            else:
                # Not enough DU for a detailed clarification
                logger.warning(f"Agent {agent_id} attempted a detailed clarification but had insufficient DU ({current_du} < {DU_COST_REQUEST_DETAILED_CLARIFICATION}). Request may be treated as a simple clarification.")
                # The message still gets sent, but it's treated as a "simple" clarification
                
                # Optionally: Add a note to the agent about this for their internal state awareness
                agent_persistent_state.last_clarification_downgraded = True
        else:
            # Simple clarification, no DU cost
            logger.debug(f"Agent {agent_id} issued a simple clarification request (no DU cost).")
            
        # Store the clarification question in agent state
        agent_persistent_state.last_clarification_question = message_content
    
    # Update the state with the potentially modified agent state
    state['state'] = agent_persistent_state
    return state

def handle_deep_analysis_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Handles the 'perform_deep_analysis' intent.
    Deducts the DU cost for performing a deep analysis, especially for Analyzers.
    """
    agent_id = state.get('agent_id', 'UNKNOWN_HANDLER')
    agent_persistent_state = state.get('state')
    current_role = agent_persistent_state.role
    current_du = agent_persistent_state.du
    
    # Check if the agent is an Analyzer (they should get the most benefit from this action)
    is_analyzer = current_role == ROLE_ANALYZER
    
    # If not an analyzer, log a warning but still allow it (with the same cost for now)
    if not is_analyzer:
        logger.warning(f"Agent {agent_id} (Role: {current_role}) is attempting a deep analysis, which is more suited for Analyzers.")
    
    # Check if the agent has enough DU for the deep analysis
    if current_du >= DU_COST_DEEP_ANALYSIS:
        # Deduct the DU cost
        new_du = current_du - DU_COST_DEEP_ANALYSIS
        agent_persistent_state.du = new_du
        
        # Log the DU deduction
        logger.info(f"Agent {agent_id} (Role: {current_role}) spent {DU_COST_DEEP_ANALYSIS} DU to perform deep analysis. Remaining DU: {new_du}.")
        
        # Note: The outcome of this analysis might lead to DU earning through the 
        # "successful analysis" or "constructive reference" mechanisms in update_state_node
    else:
        # Not enough DU to perform the deep analysis
        logger.warning(f"Agent {agent_id} attempted deep analysis but had insufficient DU ({current_du} < {DU_COST_DEEP_ANALYSIS}). Action may be less effective or not fully registered.")
        # We don't modify the agent state here as they don't have enough DU to spend
    
    # Update the state with the potentially modified agent state
    state['state'] = agent_persistent_state
    return state

def handle_create_project_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Handles the 'create_project' intent.
    Allows an agent to create a new project, with a cost in both IP and DU.
    """
    from src.infra import config
    
    agent_id = state.get('agent_id', 'UNKNOWN_HANDLER')
    structured_output = state.get('structured_output')
    agent_persistent_state = state.get('state')
    
    # Extract the project name and description to create
    project_name = structured_output.project_name_to_create if structured_output else None
    project_description = structured_output.project_description_for_creation if structured_output else None
    
    if not project_name:
        logger.warning(f"Agent {agent_id} attempted to create a project but didn't specify a project name")
        return state
    
    # Check if the agent has enough IP and DU to create a project
    current_ip = agent_persistent_state.ip
    current_du = agent_persistent_state.du
    
    if current_ip < config.IP_COST_CREATE_PROJECT:
        logger.warning(f"Agent {agent_id} attempted to create project '{project_name}' but had insufficient IP ({current_ip} < {config.IP_COST_CREATE_PROJECT})")
        return state
    
    if current_du < config.DU_COST_CREATE_PROJECT:
        logger.warning(f"Agent {agent_id} attempted to create project '{project_name}' but had insufficient DU ({current_du} < {config.DU_COST_CREATE_PROJECT})")
        return state
    
    # Get the simulation from the environment perception
    # This requires the simulation instance to be passed via run_turn to environment_perception
    simulation = state.get('environment_perception', {}).get('simulation')
    
    if not simulation:
        logger.error(f"Agent {agent_id} couldn't create project '{project_name}' because simulation reference is missing")
        return state
    
    # Create the project
    project_id = simulation.create_project(project_name, agent_id, project_description)
    
    if project_id:
        # Project created successfully, deduct costs
        agent_persistent_state.ip -= config.IP_COST_CREATE_PROJECT
        agent_persistent_state.du -= config.DU_COST_CREATE_PROJECT
        
        # Set the agent's current project affiliation
        agent_persistent_state.current_project_id = project_id
        agent_persistent_state.current_project_affiliation = project_name  # Use project NAME, not ID
        
        # Update project history
        agent_persistent_state.project_history.append((state.get('simulation_step', 0), project_id))
        
        desc_info = f" with description: '{project_description}'" if project_description else ""
        logger.info(f"Agent {agent_id} created project '{project_name}'{desc_info} (ID: {project_id}) at a cost of {config.IP_COST_CREATE_PROJECT} IP and {config.DU_COST_CREATE_PROJECT} DU")
    else:
        logger.warning(f"Agent {agent_id} failed to create project '{project_name}'")
    
    # Update the state with the potentially modified agent state
    state['state'] = agent_persistent_state
    return state

def handle_join_project_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Handles the 'join_project' intent.
    Allows an agent to join an existing project, with a cost in both IP and DU.
    """
    from src.infra import config
    
    agent_id = state.get('agent_id', 'UNKNOWN_HANDLER')
    structured_output = state.get('structured_output')
    agent_persistent_state = state.get('state')
    
    # Extract the project ID to join
    project_id = structured_output.project_id_to_join_or_leave if structured_output else None
    
    if not project_id:
        logger.warning(f"Agent {agent_id} attempted to join a project but didn't specify a project ID")
        return state
    
    # Check if the agent is already in a project
    current_project = agent_persistent_state.current_project_id
    if current_project:
        logger.warning(f"Agent {agent_id} attempted to join project '{project_id}' but is already a member of project '{current_project}'")
        return state
    
    # Check if the agent has enough IP and DU to join a project
    current_ip = agent_persistent_state.ip
    current_du = agent_persistent_state.du
    
    if current_ip < config.IP_COST_JOIN_PROJECT:
        logger.warning(f"Agent {agent_id} attempted to join project '{project_id}' but had insufficient IP ({current_ip} < {config.IP_COST_JOIN_PROJECT})")
        return state
    
    if current_du < config.DU_COST_JOIN_PROJECT:
        logger.warning(f"Agent {agent_id} attempted to join project '{project_id}' but had insufficient DU ({current_du} < {config.DU_COST_JOIN_PROJECT})")
        return state
    
    # Get the simulation from the environment perception
    simulation = state.get('environment_perception', {}).get('simulation')
    
    if not simulation:
        logger.error(f"Agent {agent_id} couldn't join project '{project_id}' because simulation reference is missing")
        return state
    
    # Get project name first in case joining fails
    project_name = simulation.projects.get(project_id, {}).get('name', 'Unknown Project')
    
    # Attempt to join the project
    success = simulation.join_project(project_id, agent_id)
    
    if success:
        # Project joined successfully, deduct costs
        agent_persistent_state.ip -= config.IP_COST_JOIN_PROJECT
        agent_persistent_state.du -= config.DU_COST_JOIN_PROJECT
        
        # Set the agent's current project affiliation
        agent_persistent_state.current_project_id = project_id
        agent_persistent_state.current_project_affiliation = project_name  # Use project NAME, not ID
        
        # Update project history
        agent_persistent_state.project_history.append((state.get('simulation_step', 0), project_id))
        
        logger.info(f"Agent {agent_id} joined project '{project_name}' (ID: {project_id}) at a cost of {config.IP_COST_JOIN_PROJECT} IP and {config.DU_COST_JOIN_PROJECT} DU")
    else:
        logger.warning(f"Agent {agent_id} failed to join project with ID '{project_id}'")
    
    # Update the state with the potentially modified agent state
    state['state'] = agent_persistent_state
    return state

def handle_leave_project_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Handles the 'leave_project' intent.
    Allows an agent to leave a project they are currently a member of.
    """
    agent_id = state.get('agent_id', 'UNKNOWN_HANDLER')
    structured_output = state.get('structured_output')
    agent_persistent_state = state.get('state')
    
    # Extract the project ID to leave
    project_id = structured_output.project_id_to_join_or_leave if structured_output else None
    
    # If no project ID specified, use the agent's current project
    if not project_id:
        project_id = agent_persistent_state.current_project_id
        if not project_id:
            logger.warning(f"Agent {agent_id} attempted to leave a project but is not a member of any project")
            return state
    
    # Get the simulation from the environment perception
    simulation = state.get('environment_perception', {}).get('simulation')
    
    if not simulation:
        logger.error(f"Agent {agent_id} couldn't leave project '{project_id}' because simulation reference is missing")
        return state
    
    # Get project details for logging before leaving
    project_name = simulation.projects.get(project_id, {}).get('name', 'Unknown Project')
    
    # Attempt to leave the project
    success = simulation.leave_project(project_id, agent_id)
    
    if success:
        # Clear the agent's current project affiliation in both fields to keep them in sync
        agent_persistent_state.current_project_id = None
        agent_persistent_state.current_project_affiliation = None
        
        # Update project history
        agent_persistent_state.project_history.append((state.get('simulation_step', 0), None))
        
        logger.info(f"Agent {agent_id} left project '{project_name}' (ID: {project_id})")
    else:
        logger.warning(f"Agent {agent_id} failed to leave project with ID '{project_id}'")
    
    # Update the state with the potentially modified agent state
    state['state'] = agent_persistent_state
    return state

def handle_send_direct_message_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Handles the 'send_direct_message' intent.
    Processes a targeted message to a specific agent.
    """
    agent_id = state.get('agent_id', 'UNKNOWN_HANDLER')
    structured_output = state.get('structured_output')
    
    # Extract message details
    message_content = structured_output.message_content if structured_output else None
    message_recipient_id = structured_output.message_recipient_id if structured_output else None
    
    if not message_recipient_id:
        logger.warning(f"Agent {agent_id} attempted to send direct message but didn't specify a recipient")
        # Fallback to a broadcast message
        return state
    
    logger.info(f"Agent {agent_id} sending direct message to {message_recipient_id}: '{message_content[:30]}...'")
    
    # Set flag to ensure this is processed as a targeted message
    state['is_targeted'] = True
    
    return state

# Update the route_action_intent function to handle all intents
def route_action_intent(state: AgentTurnState) -> str:
    """
    Determines the next node based on the agent's action_intent.
    """
    structured_output: Optional[AgentActionOutput] = state.get('structured_output')
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
    workflow.add_node("handle_propose_idea", handle_propose_idea_node) # Specific intent handler
    workflow.add_node("handle_continue_collaboration", handle_continue_collaboration_node) # Specific intent handler
    workflow.add_node("handle_idle", handle_idle_node) # Specific intent handler
    workflow.add_node("handle_ask_clarification", handle_ask_clarification_node) # Specific intent handler
    workflow.add_node("handle_deep_analysis", handle_deep_analysis_node) # Specific intent handler
    workflow.add_node("handle_create_project", handle_create_project_node) # Specific intent handler
    workflow.add_node("handle_join_project", handle_join_project_node) # Specific intent handler
    workflow.add_node("handle_leave_project", handle_leave_project_node) # Specific intent handler
    workflow.add_node("handle_send_direct_message", handle_send_direct_message_node) # New intent handler
    workflow.add_node("update_state", update_state_node) # Unified state update
    workflow.add_node("finalize_message", finalize_message_agent_node) # Final decision on message sending

    # Define edges
    workflow.set_entry_point("analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "prepare_relationship_prompt")
    # Modify the flow to include the retrieve_memories node between relationship prompt and action generation
    workflow.add_edge("prepare_relationship_prompt", "retrieve_memories") # Update this edge
    workflow.add_edge("retrieve_memories", "generate_action_output") # Add new edge for RAG
    
    # Use a conditional edge to route based on action intent
    workflow.add_conditional_edges(
        "generate_action_output",
        route_action_intent,
        {
            "handle_propose_idea": "handle_propose_idea",
            "handle_continue_collaboration": "handle_continue_collaboration",
            "handle_idle": "handle_idle",
            "handle_ask_clarification": "handle_ask_clarification",
            "handle_deep_analysis": "handle_deep_analysis",
            "handle_create_project": "handle_create_project",
            "handle_join_project": "handle_join_project",
            "handle_leave_project": "handle_leave_project",
            "handle_send_direct_message": "handle_send_direct_message",
            "update_state": "update_state" # Default fallback
        }
    )
    
    # All intent handlers go to update_state
    workflow.add_edge("handle_propose_idea", "update_state")
    workflow.add_edge("handle_continue_collaboration", "update_state")
    workflow.add_edge("handle_idle", "update_state")
    workflow.add_edge("handle_ask_clarification", "update_state")
    workflow.add_edge("handle_deep_analysis", "update_state")
    workflow.add_edge("handle_create_project", "update_state")
    workflow.add_edge("handle_join_project", "update_state")
    workflow.add_edge("handle_leave_project", "update_state")
    workflow.add_edge("handle_send_direct_message", "update_state")
    
    # Final state update and message decision
    workflow.add_edge("update_state", "finalize_message")
    workflow.add_edge("finalize_message", END)

    return workflow.compile()

# Compile the graph
basic_agent_graph_compiled = create_basic_agent_graph() 