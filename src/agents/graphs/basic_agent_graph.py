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

# List of valid roles
VALID_ROLES = [ROLE_FACILITATOR, ROLE_INNOVATOR, ROLE_ANALYZER]

# Define the Pydantic model for structured LLM output
class AgentActionOutput(BaseModel):
    """Defines the expected structured output from the LLM."""
    thought: str = Field(..., description="The agent's internal thought or reasoning for the turn.")
    message_content: Optional[str] = Field(None, description="The message to send to other agents, or None if choosing not to send a message.")
    message_recipient_id: Optional[str] = Field(None, description="The ID of the agent this message is directed to. None means broadcast to all agents.")
    action_intent: Literal["idle", "continue_collaboration", "propose_idea", "ask_clarification"] = Field(
        default="idle", # Default intent
        description="The agent's primary intent for this turn."
    )
    requested_role_change: Optional[str] = Field(None, description="Optional: If you wish to request a change to a different role, specify the role name here (e.g., 'Innovator', 'Analyzer', 'Facilitator'). Otherwise, leave as null.")

# Define the state the graph will operate on during a single agent turn
class AgentTurnState(TypedDict):
    """Represents the state passed into and modified by the agent's graph turn."""
    agent_id: str
    current_state: Dict[str, Any] # The agent's full state dictionary
    simulation_step: int          # The current step number from the simulation
    previous_thought: str | None  # The thought from the *last* turn
    environment_perception: Dict[str, Any] # Perception data from the environment
    perceived_messages: List[Dict[str, Any]] # Messages perceived from last step (broadcasts and targeted)
    memory_history_list: List[Tuple[int, str, str]] # Field for history list
    turn_sentiment_score: int     # Field for aggregated sentiment score
    prompt_modifier: str          # Field for relationship-based prompt adjustments
    structured_output: Optional[AgentActionOutput] # Holds the parsed LLM output object
    agent_goal: str               # The agent's goal for the simulation
    updated_state: Dict[str, Any] # Output field: The updated state after the turn
    vector_store_manager: Optional[Any] # For persisting memories to vector store
    rag_summary: str              # Summarized memories from vector store
    knowledge_board_content: List[str]  # Current entries on the knowledge board
    knowledge_board: Optional[Any] # The knowledge board instance for posting entries
    scenario_description: str     # Description of the simulation scenario
    current_role: str             # The agent's current role in the simulation
    influence_points: int         # The agent's current Influence Points
    steps_in_current_role: int    # Steps taken in the current role
    data_units: int               # The agent's current Data Units

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
    current_agent_state = state['current_state']
    relationships = current_agent_state.get('relationships', {})
    
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
    current_agent_state = state['current_state']
    relationships = current_agent_state.get('relationships', {})
    
    # Get the raw current role string
    raw_role_name = state.get('current_role', ROLE_ANALYZER)  # Default to Analyzer if not found
    
    # Get role description and role-specific guidance
    role_description = ROLE_DESCRIPTIONS.get(raw_role_name, f"{raw_role_name}: Contribute effectively based on your role.")
    role_specific_guidance = ROLE_PROMPT_SNIPPETS.get(raw_role_name, "Consider how your role might influence your perspective and contributions.")
    
    # Get steps in current role and influence points
    steps_in_current_role = current_agent_state.get('steps_in_current_role', 0)
    influence_points = current_agent_state.get('influence_points', 0)
    
    # Get data units
    data_units = current_agent_state.get('data_units', 0)
    
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
    prompt_parts.append(f"Your current mood is: {current_agent_state.get('mood', 'neutral')}.")
    prompt_parts.append(f"Your current descriptive mood is: {current_agent_state.get('descriptive_mood', 'neutral')}.")
    prompt_parts.append(f"You have taken {current_agent_state.get('step_counter', 0)} steps so far.")
    prompt_parts.append(f"Your current Influence Points (IP): {influence_points}.")
    prompt_parts.append(f"Your current Data Units (DU): {data_units}.")
    prompt_parts.append(f"Your current role: {raw_role_name}.")
    prompt_parts.append(f"Steps in current role: {steps_in_current_role}.")
    
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
    prompt_parts.append(f"4. 'ask_clarification' - Request more information about something unclear.")
    
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
    structured_output = state['structured_output']
    agent_id = state['agent_id']
    
    # Get current state
    current_agent_state = state['current_state'].copy()
    current_role = current_agent_state.get('current_role', 'Default Contributor')
    
    # Process passive Data Units generation based on role
    current_du = current_agent_state.get('data_units', 0)
    du_generated = ROLE_DU_GENERATION.get(current_role, 0)
    new_du = current_du + du_generated
    current_agent_state['data_units'] = new_du
    logger.debug(f"Agent {agent_id} (Role: {current_role}) passively generated {du_generated} DU. New DU: {new_du}.")
    
    # Process intents
    # If we added more action types, they'd be routed here
    action_intent = structured_output.action_intent
    
    if action_intent == 'propose_idea':
        # This intent creates a new entry on the Knowledge Board
        # Typically this would be a new proposal, solution, or important insight
        message_content = structured_output.message_content
        if message_content:
            # First check if agent has enough DU to post an idea
            current_du = current_agent_state.get('data_units', 0)
            du_cost = PROPOSE_DETAILED_IDEA_DU_COST
            
            if current_du >= du_cost:
                # Have enough DU to post the idea - deduct DU cost
                new_du = current_du - du_cost
                current_agent_state['data_units'] = new_du
                logger.info(f"Agent {agent_id} spent {du_cost} DU to post a detailed idea. Remaining DU: {new_du}")
                
                # Now check for IP cost
                current_ip = current_agent_state.get('influence_points', 0)
                ip_cost = IP_COST_TO_POST_IDEA
                
                if current_ip >= ip_cost:
                    # Have enough IP to post the idea
                    new_ip = current_ip - ip_cost
                    current_agent_state['influence_points'] = new_ip
                    logger.info(f"Agent {agent_id} spent {ip_cost} IP to post an idea. Remaining IP: {new_ip}")
                    
                    # Add to knowledge board
                    knowledge_board = state.get('knowledge_board')
                    if knowledge_board:
                        knowledge_board.add_entry(agent_id, state['simulation_step'], message_content)
                    
                    # Award IP for successful proposal
                    award_ip = IP_AWARD_FOR_PROPOSAL
                    final_ip = new_ip + award_ip
                    current_agent_state['influence_points'] = final_ip
                    logger.info(f"Agent {agent_id} earned {award_ip} IP for proposing an idea. New IP: {final_ip}")
                    
                    # Update the last_proposed_idea field
                    current_agent_state['last_proposed_idea'] = message_content
                else:
                    # Not enough IP to post the idea
                    logger.warning(f"Agent {agent_id} attempted to post an idea but had insufficient IP ({current_ip} IP) for the cost of {ip_cost} IP. Idea not posted.")
                    # Still update last_proposed_idea even though it wasn't posted
                    current_agent_state['last_proposed_idea'] = message_content
                    
                    # Refund the DU since the idea wasn't posted
                    current_agent_state['data_units'] = current_du
                    logger.info(f"Agent {agent_id} was refunded {du_cost} DU since the idea wasn't posted due to insufficient IP.")
            else:
                # Not enough DU to post the idea
                logger.warning(f"Agent {agent_id} attempted to post an idea but had insufficient DU ({current_du} DU) for the cost of {du_cost} DU. Idea not posted.")
                # Still update last_proposed_idea even though it wasn't posted
                current_agent_state['last_proposed_idea'] = message_content
    
    elif action_intent == 'ask_clarification':
        # This intent asks a clarification question
        # This would typically be a request for more information about something
        question_content = structured_output.message_content
        if question_content:
            current_agent_state['last_clarification_question'] = question_content
    
    # Process mood and emotional state based on perception analysis
    # This uses the sentiment score calculated earlier
    sentiment_score = state.get('turn_sentiment_score', 0)
    
    # Process the thought from the structured output
    thought = structured_output.thought
    if thought:
        logger.debug(f"  Processing thought from structured output: '{thought}'")
        # Store the thought in memory
        sim_step = state['simulation_step']
        memory_history = current_agent_state.get('memory_history', [])
        memory_history.append((sim_step, 'thought', thought))
        current_agent_state['memory_history'] = memory_history
        current_agent_state['last_thought'] = thought
    
    # Update mood (very simple for now)
    current_mood_value = current_agent_state.get('mood_value', 0.0)
    current_mood = current_agent_state.get('mood', 'neutral')
    
    # Simple: mood is slightly shifted by sentiment, but mostly stays the same for stability
    # Mood changes by 20% of the sentiment score
    logger.debug(f"  Using overall sentiment score {sentiment_score} to update mood.")
    new_mood_value = current_mood_value * 0.8 + sentiment_score * 0.2
    new_mood_value = max(-1.0, min(1.0, new_mood_value))  # Keep within [-1, 1]
    
    # Only change mood descriptor if it's a meaningful change
    mood_level = get_mood_level(new_mood_value)
    if current_mood != mood_level:
        logger.debug(f"Agent {agent_id} mood changed from '{current_mood}' to '{mood_level}'.")
    else:
        logger.debug(f"Agent {agent_id} mood remains '{current_mood}'.")
    
    descriptive_mood = get_descriptive_mood(new_mood_value)
    current_descriptive_mood = current_agent_state.get('descriptive_mood', 'neutral')
    if current_descriptive_mood != descriptive_mood:
        logger.debug(f"Agent {agent_id} descriptive mood changed from '{current_descriptive_mood}' to '{descriptive_mood}'.")
    else:
        logger.debug(f"Agent {agent_id} descriptive mood remains '{descriptive_mood}'.")
    
    # Update state with new values
    current_agent_state['mood_value'] = new_mood_value
    current_agent_state['mood'] = mood_level
    current_agent_state['descriptive_mood'] = descriptive_mood
    
    # Update memory with the just-added thought (for better logs)
    logger.debug(f"  Updated memory history (pre-message decision): {current_agent_state.get('memory_history', [])}")
    
    # Process requested role change if any
    requested_role = structured_output.requested_role_change
    from src.agents.core.roles import ROLE_FACILITATOR, ROLE_INNOVATOR, ROLE_ANALYZER
    VALID_ROLES = [ROLE_FACILITATOR, ROLE_INNOVATOR, ROLE_ANALYZER]
    
    # Constants for role change
    ROLE_CHANGE_IP_COST = 5
    ROLE_CHANGE_COOLDOWN = 3
    
    if requested_role:
        current_role = current_agent_state.get('current_role', 'N/A')
        current_ip = current_agent_state.get('influence_points', 0)
        steps_in_role = current_agent_state.get('steps_in_current_role', 0)
        
        logger.info(f"Agent {agent_id} requested role change from {current_role} to {requested_role}.")
        
        # Check if the role is valid and different from current role
        if requested_role not in VALID_ROLES:
            logger.warning(f"Agent {agent_id} requested role change to '{requested_role}', which is not a valid role. Valid roles are: {VALID_ROLES}")
            # No action taken
        elif requested_role == current_role:
            logger.warning(f"Agent {agent_id} requested role change to '{requested_role}', which is their current role. No change needed.")
            # No action taken
        else:
            # Now check if they meet requirements
            can_change_role = True
            reason = ""
            
            if current_ip < ROLE_CHANGE_IP_COST:
                can_change_role = False
                reason = f"insufficient IP (needs {ROLE_CHANGE_IP_COST}, has {current_ip})"
            elif steps_in_role < ROLE_CHANGE_COOLDOWN:
                can_change_role = False
                reason = f"hasn't spent enough steps in current role (needs {ROLE_CHANGE_COOLDOWN}, has {steps_in_role})"
            
            if can_change_role:
                # Apply the change
                logger.info(f"Agent {agent_id} role change from {current_role} to {requested_role} approved!")
                
                # Deduct IP cost
                new_ip = current_ip - ROLE_CHANGE_IP_COST
                current_agent_state['influence_points'] = new_ip
                logger.info(f"Agent {agent_id} spent {ROLE_CHANGE_IP_COST} IP for role change. New IP: {new_ip}")
                
                # Update role
                current_agent_state['current_role'] = requested_role
                current_agent_state['steps_in_current_role'] = 0  # Reset counter for new role
                logger.info(f"Agent {agent_id} is now a {requested_role}.")
            else:
                logger.warning(f"Agent {agent_id} role change to {requested_role} rejected: {reason}")
    
    # Always increment counter of steps in current role
    current_agent_state['steps_in_current_role'] = current_agent_state.get('steps_in_current_role', 0) + 1
    
    # Always increment step counter
    current_agent_state['step_counter'] = current_agent_state.get('step_counter', 0) + 1
    
    return {
        'updated_state': current_agent_state,  # The actual agent state that will persist
        'structured_output': structured_output,   # The parsed output from the LLM
        'updated_agent_state': current_agent_state  # (Redundant, both keys used by different parts of code)
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
                'updated_agent_state': final_agent_state
            }
        else:
            # Message suppressed due to mood
            logger.info(f"Agent {agent_id} (mood: {current_mood}, descriptive: {descriptive_mood}) suppresses message: \"{message_content}\"")
            return_values = {
                'message_content': None,  # Suppressed
                'message_recipient_id': None,
                'action_intent': 'idle',  # Force idle when suppressing
                'updated_agent_state': final_agent_state
            }
    else:
        # No message was generated
        logger.debug(f"Agent {agent_id} (descriptive mood: {descriptive_mood}) has no message to send.")
        return_values = {
            'message_content': None,
            'message_recipient_id': None,
            'action_intent': action_intent,  # Maintain original intent
            'updated_agent_state': final_agent_state
        }
    
    logger.debug(f"FINALIZE_RETURN :: Agent {agent_id}: Returning final state with updated_agent_state included")
    return return_values

def route_relationship_context(state: AgentTurnState) -> str:
    """
    Routes to relationship prompt modifier generation based on whether the agent has any
    relationships established yet.
    """
    agent_id = state.get('agent_id')
    current_agent_state = state.get('current_state', {})
    relationships = current_agent_state.get('relationships', {})
    
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
    agent_persistent_state = state.get('current_state', {})
    current_ip = agent_persistent_state.get('influence_points', 0)
    current_du = agent_persistent_state.get('data_units', 0)
    
    # Log the current IP and DU amounts for debugging
    logging.debug(f"Agent {agent_id} current IP: {current_ip}, current DU: {current_du}, attempting to propose idea")
    
    if idea_content:
        # First check if agent has enough DU to post to the Knowledge Board
        if current_du >= PROPOSE_DETAILED_IDEA_DU_COST:
            # Deduct the DU cost for the detailed idea
            agent_persistent_state['data_units'] = current_du - PROPOSE_DETAILED_IDEA_DU_COST
            updated_du = agent_persistent_state['data_units']
            logging.info(f"Agent {agent_id} spent {PROPOSE_DETAILED_IDEA_DU_COST} DU to post a detailed idea. Remaining DU: {updated_du}.")
            
            # Now check if agent has enough IP to post to the Knowledge Board
            if current_ip >= IP_COST_TO_POST_IDEA:
                # Deduct the cost to post the idea
                agent_persistent_state['influence_points'] = current_ip - IP_COST_TO_POST_IDEA
                updated_ip = agent_persistent_state['influence_points']
                logging.info(f"Agent {agent_id} spent {IP_COST_TO_POST_IDEA} IP to post an idea. Remaining IP: {updated_ip}.")
                
                # Add the proposed idea to the Knowledge Board
                if knowledge_board:
                    entry = f"{idea_content}"
                    knowledge_board.add_entry(agent_id=agent_id, entry=entry, step=sim_step)
                    logging.info(f"KnowledgeBoard: Added entry from Agent {agent_id} at step {sim_step}")
                    award_ip = True
                
                # If the idea was successfully posted, award IP
                if award_ip:
                    # Award the IP for a successful proposal
                    agent_persistent_state['influence_points'] += IP_AWARD_FOR_PROPOSAL
                    final_ip = agent_persistent_state['influence_points']
                    logging.info(f"Agent {agent_id} earned {IP_AWARD_FOR_PROPOSAL} IP for proposing an idea. New IP: {final_ip}.")
            else:
                # Agent doesn't have enough IP to post to the Knowledge Board
                logging.warning(f"Agent {agent_id} attempted to post idea '{idea_content[:30]}...' but had insufficient IP ({current_ip} IP) for the cost of {IP_COST_TO_POST_IDEA} IP. Idea not posted.")
                
                # Refund the DU since the idea wasn't posted
                agent_persistent_state['data_units'] = current_du
                logging.info(f"Agent {agent_id} was refunded {PROPOSE_DETAILED_IDEA_DU_COST} DU since the idea wasn't posted due to insufficient IP.")
        else:
            # Agent doesn't have enough DU to post the detailed idea
            logging.warning(f"Agent {agent_id} attempted to post idea '{idea_content[:30]}...' but had insufficient DU ({current_du} DU) for the cost of {PROPOSE_DETAILED_IDEA_DU_COST} DU. Idea not posted.")
    
    # Store the proposed idea content and updated agent state in the return state
    ret_state['proposed_idea_content'] = idea_content
    ret_state['current_state'] = agent_persistent_state
    
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
    Currently a placeholder for future functionality.
    """
    agent_id = state.get('agent_id', 'UNKNOWN_HANDLER')
    logger.info(f"Agent {agent_id}: Executing 'ask_clarification' intent handler (currently placeholder).")
    # Potentially add logic to track what was asked about
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
            "update_state": "update_state" # Default fallback
        }
    )
    
    # All intent handlers go to update_state
    workflow.add_edge("handle_propose_idea", "update_state")
    workflow.add_edge("handle_continue_collaboration", "update_state")
    workflow.add_edge("handle_idle", "update_state")
    workflow.add_edge("handle_ask_clarification", "update_state")
    
    # Final state update and message decision
    workflow.add_edge("update_state", "finalize_message")
    workflow.add_edge("finalize_message", END)

    return workflow.compile()

# Compile the graph
basic_agent_graph_compiled = create_basic_agent_graph() 