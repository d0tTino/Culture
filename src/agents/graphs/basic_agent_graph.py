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
from src.infra.llm_client import generate_structured_output, analyze_sentiment, generate_text, summarize_memory_context # Updated import structure to match function name
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
    action_intent: Literal["idle", "continue_collaboration", "propose_idea", "ask_clarification", "perform_deep_analysis", "create_project", "join_project", "leave_project"] = Field(
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
    Prepares a sentiment-based prompt modifier based on the agent's relationship scores
    with other agents.
    """
    agent_id = state['agent_id']
    agent_state = state['state']
    relationships = agent_state.relationships
    
    # Default neutral modifier
    modifier = "Maintain a neutral stance in your messages."
    
    # Only generate relationship-based prompts if we have relationships
    if relationships:
        # Find most positive and most negative relationships
        most_positive_score = -1.1
        most_positive_agent = None
        most_negative_score = 1.1
        most_negative_agent = None
        
        for other_id, score in relationships.items():
            if score > most_positive_score:
                most_positive_score = score
                most_positive_agent = other_id
            if score < most_negative_score:
                most_negative_score = score
                most_negative_agent = other_id
        
        # Use these values to craft the prompt modifier
        if most_positive_agent and most_positive_score > 0.3:
            if most_negative_agent and most_negative_score < -0.3:
                # Have both strong positive and negative relationships
                modifier = f"Be especially friendly and supportive toward {most_positive_agent} (relationship: {most_positive_score:.1f}) while being more cautious and measured in interactions with {most_negative_agent} (relationship: {most_negative_score:.1f})."
                logger.debug(f"Agent {agent_id}: Applying 'mixed' prompt modifier.")
            else:
                # Only have strong positive relationship
                modifier = f"Be especially friendly and supportive toward {most_positive_agent} (relationship: {most_positive_score:.1f}). Show enthusiasm and consider their ideas with extra interest."
                logger.debug(f"Agent {agent_id}: Applying 'positive' prompt modifier.")
        elif most_negative_agent and most_negative_score < -0.3:
            # Only have strong negative relationship
            modifier = f"Be cautious and formal in interactions with {most_negative_agent} (relationship: {most_negative_score:.1f}). Focus on facts rather than opinions in your responses."
            logger.debug(f"Agent {agent_id}: Applying 'negative' prompt modifier.")
        else:
            logger.debug(f"Agent {agent_id}: Applying 'neutral' prompt modifier (no agents with scores).")
    else:
        logger.debug(f"Agent {agent_id}: Applying 'neutral' prompt modifier (alone).")

    return {"prompt_modifier": modifier}

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
    prompt_parts.append(f"Your current mood is: {agent_state.mood}.")
    prompt_parts.append(f"Your current descriptive mood is: {agent_state.descriptive_mood}.")
    prompt_parts.append(f"You have taken {agent_state.step_counter} steps so far.")
    prompt_parts.append(f"Your current Influence Points (IP): {influence_points}.")
    prompt_parts.append(f"Your current Data Units (DU): {data_units}.")
    prompt_parts.append(f"Your current role: {raw_role_name}.")
    prompt_parts.append(f"Steps in current role: {steps_in_current_role}.")
    
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
    
    prompt_parts.append(f"\nTask: Based on all the context, generate your internal thought, decide if you want to send a message, and choose your primary action intent for this turn.")
    
    prompt_parts.append(f"\nMESSAGING OPTIONS:")
    prompt_parts.append(f"- Send a message to all agents (broadcast) by leaving message_recipient_id as null")
    prompt_parts.append(f"- Send a targeted message to a specific agent by setting message_recipient_id to their agent ID (as shown in 'Other Agents Present')")
    prompt_parts.append(f"- Choose not to send any message by setting message_content to null")
    
    prompt_parts.append(f"\nIMPORTANT ACTION CHOICES:")
    prompt_parts.append(f"1. 'idle' - No specific action, continue monitoring.")
    prompt_parts.append(f"2. 'continue_collaboration' - Standard contribution to ongoing discussion.")
    prompt_parts.append(f"3. 'propose_idea' - Suggest a formal idea to be added to the Knowledge Board for permanent reference (costs {PROPOSE_DETAILED_IDEA_DU_COST} DU and {IP_COST_TO_POST_IDEA} IP).")
    prompt_parts.append(f"4. 'ask_clarification' - Request more information about something unclear (may cost {DU_COST_REQUEST_DETAILED_CLARIFICATION} DU for detailed requests).")
    prompt_parts.append(f"5. 'perform_deep_analysis' - Perform a deep analysis (costs {DU_COST_DEEP_ANALYSIS} DU) - As an Analyzer, use this to signal you are conducting a thorough investigation of a proposal or situation. Your broadcast message should reflect your findings or critical questions.")
    prompt_parts.append(f"6. 'create_project' - Create a new project (costs {config.IP_COST_CREATE_PROJECT} IP and {config.DU_COST_CREATE_PROJECT} DU). Specify the project name in the project_name_to_create field and an optional description in project_description_for_creation.")
    prompt_parts.append(f"7. 'join_project' - Join an existing project (costs {config.IP_COST_JOIN_PROJECT} IP and {config.DU_COST_JOIN_PROJECT} DU). Specify the project ID in the project_id_to_join_or_leave field.")
    prompt_parts.append(f"8. 'leave_project' - Leave your current project. You can specify the project ID in the project_id_to_join_or_leave field, or leave it empty to leave your current project.")
    
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
    Node that takes the output from the LLM and updates agent state.
    Returns the updated state dictionary.
    
    Note: This is the main "actions" processor. If the agent requested to perform
    an action via one of the structured intents, this is where that would be processed.
    """
    # Import role constants
    from src.agents.core.roles import ROLE_ANALYZER
    
    structured_output = state['structured_output']
    agent_id = state['agent_id']
    sim_step = state['simulation_step']
    
    # Get current agent state
    agent_state = state['state']
    current_role = agent_state.role
    
    # Process passive Data Units generation based on role
    du_generated = ROLE_DU_GENERATION.get(current_role, 0)
    agent_state.du += du_generated
    logger.debug(f"Agent {agent_id} (Role: {current_role}) passively generated {du_generated} DU. New DU: {agent_state.du}.")
    
    # Process intents
    # If we added more action types, they'd be routed here
    action_intent = structured_output.action_intent
    message_content = structured_output.message_content if structured_output else None
    thought = structured_output.thought if structured_output else ""
    
    # Check for constructive references to board entries (simplified approach)
    if message_content or thought:
        text_to_check = f"{thought} {message_content if message_content else ''}"
        reference_keywords = [
            "building on", "extending", "refining idea from", "referencing", 
            "regarding step", "based on the idea", "inspired by", "following up on"
        ]
        
        has_reference = False
        for keyword in reference_keywords:
            if keyword.lower() in text_to_check.lower():
                has_reference = True
                break
        
        if has_reference:
            # Award DU bonus for constructively referencing board entries
            agent_state.du += DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE
            logger.info(f"Agent {agent_id} earned {DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE} DU for constructively referencing a board entry. New DU: {agent_state.du}")
    
    # Check for successful analysis (for Analyzer role)
    if current_role == ROLE_ANALYZER and (message_content or thought):
        text_to_check = f"{thought} {message_content if message_content else ''}"
        analysis_keywords = [
            "potential flaw", "risk identified", "improvement could be", 
            "vulnerability found", "issue detected", "concern about", 
            "critical analysis", "weak point", "limitation", "problematic assumption"
        ]
        
        has_analysis = False
        for keyword in analysis_keywords:
            if keyword.lower() in text_to_check.lower():
                has_analysis = True
                break
        
        # Check if the message sentiment is not negative (constructive criticism)
        message_sentiment = 'neutral'
        if message_content:
            message_sentiment = analyze_sentiment(message_content) or 'neutral'
        
        if has_analysis and message_sentiment != 'negative':
            # Award DU for successful analysis
            agent_state.du += DU_AWARD_SUCCESSFUL_ANALYSIS
            logger.info(f"Agent {agent_id} (Analyzer) earned {DU_AWARD_SUCCESSFUL_ANALYSIS} DU for successful analysis. New DU: {agent_state.du}")
    
    if action_intent == 'propose_idea':
        # This intent creates a new entry on the Knowledge Board
        # Typically this would be a new proposal, solution, or important insight
        if message_content:
            # First check if agent has enough DU to post an idea
            du_cost = PROPOSE_DETAILED_IDEA_DU_COST
            
            if agent_state.du >= du_cost:
                # Have enough DU to post the idea - deduct DU cost
                agent_state.du -= du_cost
                logger.info(f"Agent {agent_id} spent {du_cost} DU to post a detailed idea. Remaining DU: {agent_state.du}")
                
                # Now check for IP cost
                ip_cost = IP_COST_TO_POST_IDEA
                
                if agent_state.ip >= ip_cost:
                    # Have enough IP to post the idea
                    agent_state.ip -= ip_cost
                    logger.info(f"Agent {agent_id} spent {ip_cost} IP to post an idea. Remaining IP: {agent_state.ip}")
                    
                    # Add to knowledge board
                    knowledge_board = state.get('knowledge_board')
                    if knowledge_board:
                        knowledge_board.add_entry(message_content, agent_id, state['simulation_step'])
                    
                    # Award IP for successful proposal
                    award_ip = IP_AWARD_FOR_PROPOSAL
                    agent_state.ip += award_ip
                    logger.info(f"Agent {agent_id} earned {award_ip} IP for proposing an idea. New IP: {agent_state.ip}")
                else:
                    # Not enough IP to post the idea
                    logger.warning(f"Agent {agent_id} attempted to post an idea but had insufficient IP ({agent_state.ip} IP) for the cost of {ip_cost} IP. Idea not posted.")
                    
                    # Refund the DU since the idea wasn't posted
                    agent_state.du += du_cost
                    logger.info(f"Agent {agent_id} was refunded {du_cost} DU since the idea wasn't posted due to insufficient IP.")
            else:
                # Not enough DU to post the idea
                logger.warning(f"Agent {agent_id} attempted to post an idea but had insufficient DU ({agent_state.du} DU) for the cost of {du_cost} DU. Idea not posted.")
    
    # Process mood and emotional state based on perception analysis
    # This uses the sentiment score calculated earlier
    sentiment_score = state.get('turn_sentiment_score', 0)
    
    # Process the thought from the structured output
    if thought:
        logger.debug(f"  Processing thought from structured output: '{thought}'")
        # Store the thought in memory
        memory_entry = {"step": sim_step, "type": "thought", "content": thought}
        agent_state.short_term_memory.append(memory_entry)
    
    # Update mood (very simple for now)
    # Simple: mood is slightly shifted by sentiment, but mostly stays the same for stability
    # Mood changes by 20% of the sentiment score
    logger.debug(f"  Using overall sentiment score {sentiment_score} to update mood.")
    new_mood_value = agent_state.mood_decay_rate * 0.8 + sentiment_score * 0.2
    new_mood_value = max(-1.0, min(1.0, new_mood_value))  # Keep within [-1, 1]
    
    # Only change mood descriptor if it's a meaningful change
    mood_level = get_mood_level(new_mood_value)
    if agent_state.mood != mood_level:
        logger.debug(f"Agent {agent_id} mood changed from '{agent_state.mood}' to '{mood_level}'.")
    
    descriptive_mood = get_descriptive_mood(new_mood_value)
    
    # Update mood and add to history
    agent_state.mood = mood_level
    agent_state.mood_history.append((sim_step, mood_level))
    
    # Process requested role change if any
    requested_role = structured_output.requested_role_change
    
    if requested_role and requested_role in VALID_ROLES:
        if requested_role != agent_state.role:
            # Check if the cooldown period has passed
            if agent_state.steps_in_current_role >= agent_state.role_change_cooldown:
                # Check if the agent has enough IP
                if agent_state.ip >= agent_state.role_change_ip_cost:
                    # Deduct IP cost
                    agent_state.ip -= agent_state.role_change_ip_cost
                    
                    # Update role
                    old_role = agent_state.role
                    agent_state.role = requested_role
                    agent_state.steps_in_current_role = 0
                    agent_state.role_history.append((sim_step, requested_role))
                    
                    logger.info(f"Agent {agent_id} changed role from {old_role} to {requested_role}. Spent {agent_state.role_change_ip_cost} IP. Remaining IP: {agent_state.ip}")
                else:
                    logger.warning(f"Agent {agent_id} requested role change to {requested_role} but had insufficient IP (needed {agent_state.role_change_ip_cost}, had {agent_state.ip}).")
            else:
                logger.warning(f"Agent {agent_id} requested role change to {requested_role} but cooldown period not satisfied (needs {agent_state.role_change_cooldown} steps, current: {agent_state.steps_in_current_role}).")
    
    # Increment steps in current role
    agent_state.steps_in_current_role += 1
    
    # Update track of actions taken
    agent_state.actions_taken_count += 1
    agent_state.last_action_step = sim_step
    
    # If a message was sent, update message tracking
    if message_content:
        agent_state.messages_sent_count += 1
        agent_state.last_message_step = sim_step
    
    # Update history fields for various state parameters
    agent_state.ip_history.append((sim_step, agent_state.ip))
    agent_state.du_history.append((sim_step, agent_state.du))
    agent_state.relationship_history.append((sim_step, agent_state.relationships.copy()))
    
    # Return the updated state
    return {
        "state": agent_state,
        "action_intent": structured_output.action_intent,
        "message_content": structured_output.message_content,
        "message_recipient_id": structured_output.message_recipient_id
    }

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

def finalize_message_agent_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Makes the final decision on whether or not to send the message based on
    agent's mood, updates memory history with the decision, and prepares the return value
    to inform the simulation whether a message was sent or not.
    """
    agent_id = state['agent_id']
    sim_step = state.get('simulation_step', 0)
    current_mood = state.get('updated_state', {}).get('mood', 'neutral')
    descriptive_mood = state.get('updated_state', {}).get('descriptive_mood', 'neutral')
    structured_output = state.get('structured_output')
    message_content = structured_output.message_content if structured_output else None
    message_recipient_id = structured_output.message_recipient_id if structured_output else None
    action_intent = structured_output.action_intent if structured_output else 'idle'
  
    # Enhanced logic: Only send message if mood is appropriate AND there's a message
    should_send_message = (current_mood != 'unhappy' and 
                          descriptive_mood not in ['negative', 'very_negative'] and 
                          message_content is not None)
    
    # Create a clone to avoid altering the original dictionary
    final_agent_state = state.get('updated_state', state.get('current_state', {})).copy()
    
    # Only log about messaging if there was a message generated
    if message_content:
        if should_send_message:
            logger.info(f"Agent {agent_id} (mood: {current_mood}, descriptive: {descriptive_mood}) decides to send message: \"{message_content}\"")
            
            # Update memory with sent message
            memory_list = final_agent_state.get('memory_history', [])
            memory_deque = deque(memory_list, maxlen=5)
            
            # Add message sent entry to memory
            message_type = "private" if message_recipient_id else "broadcast"
            memory_deque.append((sim_step, f'message_sent_{message_type}', f"Me: \"{message_content}\""))
            final_agent_state['memory_history'] = list(memory_deque)
            
            # Store the last message in agent state
            final_agent_state['last_message'] = message_content
            
            # Setup return values for successful message
            return_values = {
                'message_content': message_content,
                'message_recipient_id': message_recipient_id,
                'action_intent': action_intent,
                'updated_state': final_agent_state
            }
        else:
            # Message suppressed due to mood
            logger.info(f"Agent {agent_id} (mood: {current_mood}, descriptive: {descriptive_mood}) suppresses message: \"{message_content}\"")
            return_values = {
                'message_content': None,  # Suppressed
                'message_recipient_id': None,
                'action_intent': 'idle',  # Force idle when suppressing
                'updated_state': final_agent_state
            }
    else:
        # No message was generated
        logger.debug(f"Agent {agent_id} (descriptive mood: {descriptive_mood}) has no message to send.")
        return_values = {
            'message_content': None,
            'message_recipient_id': None,
            'action_intent': action_intent,  # Maintain original intent
            'updated_state': final_agent_state
        }
    
    logger.debug(f"FINALIZE_RETURN :: Agent {agent_id}: Returning final state with updated_state included")
    return return_values

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
    if not other_agents_info:
        return "  You are currently alone."
    
    lines = []
    lines.append("  Other agents you can interact with (use their ID when sending targeted messages):")
    for agent_info in other_agents_info:
        other_id = agent_info.get('agent_id', 'unknown')
        other_name = agent_info.get('name', other_id[:8])
        other_mood = agent_info.get('mood', 'unknown')
        other_descriptive_mood = agent_info.get('descriptive_mood', 'unknown')
        relationship_score = relationships.get(other_id, 0.0)
        # Include the full agent_id for clear targeting
        lines.append(f"  - {other_name} (Agent ID for targeting: '{other_id}', Mood: {other_mood}, Relationship: {relationship_score:.1f})")
    
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
        
        # Set the agent's current project affiliation in both fields to keep them in sync
        agent_persistent_state.current_project_id = project_id
        agent_persistent_state.current_project_affiliation = project_id
        
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
    
    # Attempt to join the project
    success = simulation.join_project(project_id, agent_id)
    
    if success:
        # Project joined successfully, deduct costs
        agent_persistent_state.ip -= config.IP_COST_JOIN_PROJECT
        agent_persistent_state.du -= config.DU_COST_JOIN_PROJECT
        
        # Set the agent's current project affiliation in both fields to keep them in sync
        agent_persistent_state.current_project_id = project_id
        agent_persistent_state.current_project_affiliation = project_id
        
        # Update project history
        agent_persistent_state.project_history.append((state.get('simulation_step', 0), project_id))
        
        # Get project details for logging
        project_name = simulation.projects.get(project_id, {}).get('name', 'Unknown Project')
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
    else:
        # For unknown intents, go to update_state
        return "update_state"

# --- Graph Definition ---

def create_basic_agent_graph():
    """
    Builds the agent turn graph with intent routing.
    Flow: Analyze Sentiment -> Prepare Prompt -> Generate Output -> Route Intent -> [Handle Intent] -> Update State -> Finalize Message -> END
    """
    workflow = StateGraph(AgentTurnState)

    # Add the nodes
    workflow.add_node("analyze_sentiment", analyze_perception_sentiment_node)
    workflow.add_node("prepare_relationship_prompt", prepare_relationship_prompt_node)
    workflow.add_node("generate_action_output", generate_thought_and_message_node)
    workflow.add_node("handle_propose_idea", handle_propose_idea_node) # Specific intent handler
    workflow.add_node("handle_continue_collaboration", handle_continue_collaboration_node) # Specific intent handler
    workflow.add_node("handle_idle", handle_idle_node) # Specific intent handler
    workflow.add_node("handle_ask_clarification", handle_ask_clarification_node) # Specific intent handler
    workflow.add_node("handle_deep_analysis", handle_deep_analysis_node) # Specific intent handler
    workflow.add_node("handle_create_project", handle_create_project_node) # Specific intent handler
    workflow.add_node("handle_join_project", handle_join_project_node) # Specific intent handler
    workflow.add_node("handle_leave_project", handle_leave_project_node) # Specific intent handler
    workflow.add_node("update_state", update_state_node) # Unified state update
    workflow.add_node("finalize_message", finalize_message_agent_node) # Final decision on message sending

    # Define edges
    workflow.set_entry_point("analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "prepare_relationship_prompt")
    workflow.add_edge("prepare_relationship_prompt", "generate_action_output")
    
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
    
    # Final state update and message decision
    workflow.add_edge("update_state", "finalize_message")
    workflow.add_edge("finalize_message", END)

    return workflow.compile()

# Compile the graph
basic_agent_graph_compiled = create_basic_agent_graph() 