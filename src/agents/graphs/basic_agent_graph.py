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
from src.infra.llm_client import generate_structured_output, analyze_sentiment, generate_text, summarize_memory_context # Updated import
from collections import deque
from pydantic import BaseModel, Field

# Use TYPE_CHECKING to avoid circular import issues
if TYPE_CHECKING:
    from src.sim.knowledge_board import KnowledgeBoard

logger = logging.getLogger(__name__)

# Decay factors for mood and relationships
MOOD_DECAY_FACTOR = 0.02  # Mood decays towards neutral by 2% each turn
RELATIONSHIP_DECAY_FACTOR = 0.01  # Relationships decay towards neutral by 1% each turn

# Define the Pydantic model for structured LLM output
class AgentActionOutput(BaseModel):
    """Defines the expected structured output from the LLM."""
    thought: str = Field(..., description="The agent's internal thought or reasoning for the turn.")
    broadcast: Optional[str] = Field(None, description="The message to broadcast to other agents, or None if choosing not to broadcast.")
    action_intent: Literal["idle", "continue_collaboration", "propose_idea", "ask_clarification"] = Field(
        default="idle", # Default intent
        description="The agent's primary intent for this turn."
    )

# Define the state the graph will operate on during a single agent turn
class AgentTurnState(TypedDict):
    """Represents the state passed into and modified by the agent's graph turn."""
    agent_id: str
    current_state: Dict[str, Any] # The agent's full state dictionary
    simulation_step: int          # The current step number from the simulation
    previous_thought: str | None  # The thought from the *last* turn
    environment_perception: Dict[str, Any] # Perception data from the environment
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

# --- Node Functions ---

def analyze_perception_sentiment_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Analyzes the sentiment of perceived broadcasts from the previous step.
    Calculates an aggregated sentiment score for the turn.
    """
    agent_id = state['agent_id']
    sim_step = state['simulation_step']
    perception = state.get('environment_perception', {})
    perceived_broadcasts = perception.get('broadcasts', [])
    logger.debug(f"Node 'analyze_perception_sentiment_node' executing for agent {agent_id} at step {sim_step}")

    total_sentiment_score = 0
    analyzed_count = 0

    if not perceived_broadcasts:
        logger.debug("  No broadcasts perceived, sentiment score remains 0.")
        return {"turn_sentiment_score": 0}

    for msg in perceived_broadcasts:
        sender_id = msg.get('sender_id', 'unknown')
        message = msg.get('message', None)

        # Optional: Skip analyzing own messages if they are included in perception
        # if sender_id == agent_id: continue

        if message:
            sentiment = analyze_sentiment(message) # Use the new utility function
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

def generate_thought_and_message_node(state: AgentTurnState) -> Dict[str, Optional[AgentActionOutput]]:
    """
    Node that calls the LLM to generate structured output (thought, broadcast, intent)
    based on context.
    """
    agent_id = state['agent_id']
    sim_step = state['simulation_step']
    prev_thought = state.get('previous_thought', None)
    perception = state.get('environment_perception', {})
    sentiment_score = state.get('turn_sentiment_score', 0)
    prompt_modifier = state.get('prompt_modifier', "")
    agent_goal = state.get('agent_goal', "Contribute to the simulation as effectively as possible.")
    rag_summary = state.get('rag_summary', "(No relevant past memories found via RAG)")
    knowledge_board_content = state.get('knowledge_board_content', [])
    scenario_description = state.get('scenario_description', "")
    
    # Get the current agent state
    current_agent_state = state['current_state']
    relationships = current_agent_state.get('relationships', {})
    
    logger.debug(f"Node 'generate_thought_and_message_node' executing for agent {agent_id} at step {sim_step}")
    logger.debug(f"  Overall sentiment score from perceived messages: {sentiment_score}")
    logger.debug(f"  Current Relationships: {relationships}")
    logger.debug(f"  Using relationship prompt modifier: '{prompt_modifier}'")
    logger.debug(f"  Agent goal: '{agent_goal}'")
    logger.debug(f"  Using memory summary: '{rag_summary}'")
    logger.debug(f"  Knowledge board has {len(knowledge_board_content)} entries")
    logger.debug(f"  Using simulation scenario: '{scenario_description}'")

    # Format other agents' information
    other_agents_info = perception.get('other_agents_state', [])
    formatted_other_agents = _format_other_agents(other_agents_info, relationships)
    
    # Format knowledge board content
    formatted_board = _format_knowledge_board(knowledge_board_content)
    
    # Format broadcasts
    broadcasts = perception.get('broadcasts', [])
    formatted_broadcasts = _format_broadcasts(broadcasts)
    
    # Build the prompt with the agent's persona, context, and task
    prompt = f"""You are Agent_{agent_id}, an AI agent in a simulation.
Current simulation step: {sim_step}.
Your current mood is: {current_agent_state.get('mood', 'neutral')}.
You have taken {current_agent_state.get('step_counter', 0)} steps so far.

Current Simulation Scenario:
{scenario_description}

Your primary goal for this simulation is: {agent_goal}
Keep this goal in mind when deciding your thoughts and actions.

Other Agents Present:
{formatted_other_agents}

Relevant Memory Context (Summarized):
{rag_summary}

Current Knowledge Board Content (Last 10 Entries):
{formatted_board}
  
Messages Broadcast by Others in Previous Step:
{formatted_broadcasts}
  The overall sentiment of messages you perceived last step was: {sentiment_score} (positive > 0, negative < 0, neutral = 0). 

Guidance based on relationships: {prompt_modifier}
                            
Task: Based on all the context, generate your internal thought, decide if you want to broadcast a message, and choose your primary action intent for this turn.
                            
IMPORTANT ACTION CHOICES:
1. 'idle' - No specific action, continue monitoring.
2. 'continue_collaboration' - Standard contribution to ongoing discussion.
3. 'propose_idea' - Suggest a formal idea to be added to the Knowledge Board for permanent reference.
4. 'ask_clarification' - Request more information about something unclear.

If you have a significant insight or proposal you'd like to be added to the shared Knowledge Board, use 'propose_idea'.

You MUST respond ONLY with a valid JSON object matching the specified schema.
Example for no broadcast:
{{
  "thought": "My internal reasoning...",
  "broadcast": null,
  "action_intent": "idle"
}}
Example with broadcast:
{{
  "thought": "My internal reasoning...",
  "broadcast": "My message to others.",
  "action_intent": "continue_collaboration"
}}
Example with proposal for Knowledge Board:
{{
  "thought": "I have a valuable idea to share...",
  "broadcast": "I propose we consider implementing X to solve Y...",
  "action_intent": "propose_idea"
}}
"""
    
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
                    f"Broadcast='{structured_output.broadcast}', "
                    f"Intent='{structured_output.action_intent}'")
    else:
        logger.warning(f"Agent {agent_id} failed to generate or parse structured output.")

    return {"structured_output": structured_output} # Return the object (or None)

def update_state_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Updates mood based on sentiment, updates relationships based on
    perceived broadcast sentiment, stores the latest thought, and updates memory history.
    This node prepares the state *before* the final decision on broadcasting.
    """
    agent_id = state['agent_id']
    sim_step = state['simulation_step']
    
    # --- Get Structured Output ---
    structured_output = state.get('structured_output')
    thought = structured_output.thought if structured_output else "[thought generation failed]"
    # Broadcast will be handled by the routing/finalization steps
    # --- End Get Structured Output ---
    
    perception = state.get('environment_perception', {})
    perceived_broadcasts = perception.get('broadcasts', [])
    sentiment_score = state.get('turn_sentiment_score', 0)

    logger.debug(f"Node 'update_state_node' executing for agent {agent_id}")
    logger.debug(f"  Using overall sentiment score {sentiment_score} to update mood.")
    logger.debug(f"  Processing thought from structured output: '{thought}'")

    current_agent_state = state['current_state'].copy()
    try:
        # Increment step counter
        current_count = current_agent_state.get('step_counter', 0)
        new_count = current_count + 1
        current_agent_state['step_counter'] = new_count

        # Update Mood Based on Overall Sentiment
        current_mood = current_agent_state.get('mood', 'neutral')
        new_mood = current_mood
        if sentiment_score > 0: 
            new_mood = 'happy'
        elif sentiment_score < 0: 
            new_mood = 'unhappy' # Add unhappy possibility
        else: 
            new_mood = 'neutral'
            
        if new_mood != current_mood:
            logger.info(f"Agent {agent_id} mood changed from '{current_mood}' to '{new_mood}' based on overall sentiment score {sentiment_score}.")
            current_agent_state['mood'] = new_mood
        else: 
            logger.debug(f"Agent {agent_id} mood remains '{current_mood}'.")

        # Update Relationships Based on Individual Message Sentiment
        relationships = current_agent_state.get('relationships', {}).copy()
        if perceived_broadcasts:
            logger.debug(f"  Updating relationships based on {len(perceived_broadcasts)} perceived broadcasts...")
            for msg in perceived_broadcasts:
                sender_id = msg.get('sender_id')
                message = msg.get('message')
                if sender_id and message and sender_id != agent_id:
                    msg_sentiment = analyze_sentiment(message)
                    current_relationship_score = relationships.get(sender_id, 0.0)
                    new_relationship_score = current_relationship_score
                    sentiment_delta = 0.0
                    if msg_sentiment == 'positive': 
                        sentiment_delta = 0.1
                    elif msg_sentiment == 'negative': 
                        sentiment_delta = -0.1
                        
                    if sentiment_delta != 0.0:
                        new_relationship_score += sentiment_delta
                        new_relationship_score = max(-1.0, min(1.0, new_relationship_score)) # Clamp
                        if new_relationship_score != current_relationship_score:
                             relationships[sender_id] = new_relationship_score
                             logger.info(f"Agent {agent_id} relationship with {sender_id} updated to {new_relationship_score:.2f} (sentiment: {msg_sentiment})")
            current_agent_state['relationships'] = relationships

        # Store latest thought (broadcast stored later based on decision)
        current_agent_state['last_thought'] = thought

        # Update Memory History (including perceived, thought, but NOT sent broadcast yet)
        memory_list = current_agent_state.get('memory_history', [])
        memory_deque = deque(memory_list, maxlen=5)
        if perceived_broadcasts:
            for msg in perceived_broadcasts:
                sender = msg.get('sender_id', 'unknown')
                message = msg.get('message', '')
                memory_deque.append((sim_step, 'broadcast_perceived', f"{sender}: \"{message}\""))
        if thought:
            memory_deque.append((sim_step, 'thought', thought))
        # DO NOT add broadcast_sent here

        current_agent_state['memory_history'] = list(memory_deque)
        logger.debug(f"  Updated memory history (pre-broadcast decision): {list(memory_deque)}")

    except Exception as e:
        logger.error(f"Error in update_state_node for agent {agent_id}: {e}", exc_info=True)
        return {"updated_state": state['current_state'], "structured_output": structured_output} # Return original on error

    # Return the updated state dictionary and pass along structured_output for the routing decision
    return {"updated_state": current_agent_state, "structured_output": structured_output}

# Add this routing function
def route_broadcast_decision(state: AgentTurnState) -> str:
    """
    Determines whether to broadcast based on mood and if a message exists
    in the structured output.
    """
    agent_id = state['agent_id']
    # Get the mood from the *updated* state dictionary
    current_mood = state.get('updated_state', {}).get('mood', 'neutral')
    structured_output = state.get('structured_output')
    potential_broadcast = structured_output.broadcast if structured_output else None

    # Simple logic: Only broadcast if mood is not 'unhappy' AND there's a message
    if current_mood != 'unhappy' and potential_broadcast:
        logger.debug(f"Agent {agent_id}: Mood is '{current_mood}', deciding to broadcast.")
        return "broadcast"
    else:
        reason = "unhappy mood" if current_mood == 'unhappy' else "no broadcast message"
        logger.debug(f"Agent {agent_id}: Deciding to skip broadcast due to {reason}.")
        return "skip"

# Add this final update node
def finalize_state_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Finalizes the agent state for the turn. If broadcasting, adds the
    sent broadcast from structured_output to memory and sets 'last_broadcast'.
    """
    agent_id = state['agent_id']
    sim_step = state['simulation_step']
    final_agent_state = state.get('updated_state', state['current_state']).copy()
    structured_output = state.get('structured_output')
    broadcast_to_send = structured_output.broadcast if structured_output else None

    memory_list = final_agent_state.get('memory_history', [])
    memory_deque = deque(memory_list, maxlen=5)

    # Implicitly check router decision: if broadcast_to_send exists AND mood wasn't unhappy (checked by router)
    if broadcast_to_send:
         final_agent_state['last_broadcast'] = broadcast_to_send
         memory_deque.append((sim_step, 'broadcast_sent', broadcast_to_send)) # Add sent broadcast to memory
         logger.debug(f"  Finalizing state: Added broadcast to memory and state.")
    else:
         final_agent_state['last_broadcast'] = None
         logger.debug(f"  Finalizing state: No broadcast sent/stored.")

    final_agent_state['memory_history'] = list(memory_deque)

    # Return only the final state dictionary
    return {"updated_state": final_agent_state}

# Add this new routing function
def route_relationship_context(state: AgentTurnState) -> str:
    """
    Determines the relationship context (e.g., friendly, neutral, wary)
    based on average relationship score with present agents.
    """
    agent_id = state['agent_id']
    perception = state.get('environment_perception', {})
    other_agents_info = perception.get('other_agents_state', [])
    relationships = state['current_state'].get('relationships', {})

    if not other_agents_info:
        return "neutral" # No one else present

    total_score = 0.0
    count = 0
    for agent_info in other_agents_info:
        other_id = agent_info.get('id')
        if other_id:
            total_score += relationships.get(other_id, 0.0) # Default to 0 if no score yet
            count += 1

    if count == 0:
        return "neutral"

    average_score = total_score / count
    logger.debug(f"Agent {agent_id}: Average relationship score with present agents: {average_score:.2f}")

    # Define thresholds for routing
    if average_score > 0.3:
        return "friendly"
    elif average_score < -0.3:
        return "wary" # Example threshold for negative relationships
    else:
        return "neutral"

# Add this new node function
def prepare_relationship_prompt_node(state: AgentTurnState) -> Dict[str, str]:
    """
    Generates a prompt modifier string based on the relationship context route.
    """
    agent_id = state['agent_id']
    perception = state.get('environment_perception', {})
    other_agents_info = perception.get('other_agents_state', [])
    relationships = state['current_state'].get('relationships', {})

    modifier = "" # Default empty modifier

    if other_agents_info:
        total_score = 0.0
        count = 0
        for agent_info in other_agents_info:
            other_id = agent_info.get('id')
            if other_id:
                total_score += relationships.get(other_id, 0.0)
                count += 1

        if count > 0:
            average_score = total_score / count
            if average_score > 0.3:
                modifier = "You generally have a friendly relationship with the agents present. Be warm and collaborative."
                logger.debug(f"Agent {agent_id}: Applying 'friendly' prompt modifier.")
            elif average_score < -0.3:
                modifier = "You generally have a wary or negative relationship with the agents present. Be cautious and reserved."
                logger.debug(f"Agent {agent_id}: Applying 'wary' prompt modifier.")
            else:
                modifier = "You have a neutral relationship with the agents present. Be professional and balanced."
                logger.debug(f"Agent {agent_id}: Applying 'neutral' prompt modifier.")
        else:
            logger.debug(f"Agent {agent_id}: Applying 'neutral' prompt modifier (no agents with scores).")
    else:
        logger.debug(f"Agent {agent_id}: Applying 'neutral' prompt modifier (alone).")

    return {"prompt_modifier": modifier}

def handle_propose_idea_node(state: AgentTurnState) -> Dict[str, Any]:
    """
    Handles the 'propose_idea' intent by extracting the proposed idea
    and adding it to the agent's state and to the Knowledge Board.
    """
    agent_id = state.get('agent_id')
    structured_output = state.get('structured_output')
    current_step = state.get('simulation_step')
    knowledge_board = state.get('knowledge_board')

    logger.debug(f"Node 'handle_propose_idea_node' executing for agent {agent_id}")

    if not structured_output:
        logger.warning(f"Agent {agent_id} has no structured output to process for propose_idea")
        return state

    # Extract the idea content from the structured output broadcast field
    idea_content = structured_output.broadcast
    
    # Store it in agent state for reference
    if idea_content:
        updated_state = dict(state)
        if 'current_state' not in updated_state:
            updated_state['current_state'] = {}
        
        if 'last_proposed_idea' not in updated_state['current_state']:
            updated_state['current_state']['last_proposed_idea'] = idea_content
        
        # Post to knowledge board if available
        if knowledge_board and current_step is not None:
            try:
                knowledge_board.add_entry(
                    entry=idea_content, 
                    agent_id=agent_id, 
                    step=current_step
                )
                logger.info(f"Agent {agent_id} posted idea to knowledge board: '{idea_content}'")
            except Exception as e:
                logger.error(f"Agent {agent_id} failed to post idea to knowledge board: {e}")
    else:
        logger.warning(f"Agent {agent_id} has no valid broadcast content for the proposed idea")
        updated_state = state

    return updated_state

# Add formatting helper functions
def _format_other_agents(other_agents_info, relationships):
    """Helper function to format other agents information for the prompt."""
    if not other_agents_info:
        return "  You are currently alone."
    
    lines = []
    for agent_info in other_agents_info:
        other_id = agent_info.get('id', 'unknown')
        other_name = agent_info.get('name', other_id)
        other_mood = agent_info.get('mood', 'unknown')
        relationship_score = relationships.get(other_id, 0.0)
        lines.append(f"  - {other_name} (Mood: {other_mood}, Relationship: {relationship_score:.1f})")
    
    return "\n".join(lines)

def _format_knowledge_board(board_entries):
    """Helper function to format knowledge board entries for the prompt."""
    if not board_entries:
        return "  (Board is currently empty)"
    
    lines = []
    for i, entry in enumerate(board_entries):
        lines.append(f"  - {entry}")
    
    return "\n".join(lines)

def _format_broadcasts(broadcasts):
    """Helper function to format perceived broadcasts for the prompt."""
    if not broadcasts:
        return "  No messages were broadcast in the previous step."
    
    lines = []
    for msg in broadcasts:
        sender = msg.get('sender_id', 'unknown')
        message = msg.get('message', '')
        lines.append(f"  - {sender} said: \"{message}\"")
    
    return "\n".join(lines)

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

# Update the route_action_intent function to handle all intents
def route_action_intent(state: AgentTurnState) -> str:
    """
    Determines the next node based on the agent's action_intent.
    Routes to the appropriate handler for each intent type.
    """
    structured_output: Optional[AgentActionOutput] = state.get('structured_output')
    intent = structured_output.action_intent if structured_output else "idle"
    agent_id = state.get('agent_id', 'UNKNOWN')

    logger.debug(f"Agent {agent_id}: Routing based on action_intent '{intent}'...")

    if intent == "propose_idea":
        return "handle_propose_idea"
    elif intent == "ask_clarification":
        return "handle_ask_clarification"
    elif intent == "continue_collaboration":
        return "handle_continue_collaboration"
    elif intent == "idle":
        return "handle_idle"
    else:
        # Fallback for unknown intents
        logger.warning(f"Agent {agent_id}: Unknown action_intent '{intent}'. Routing to update_state directly.")
        return "update_state"

# --- Graph Definition ---

def create_basic_agent_graph():
    """
    Builds the agent turn graph with intent routing.
    Flow: Analyze Sentiment -> Prepare Prompt -> Generate Output -> Route Intent -> [Handle Intent] -> Update State -> END
    """
    workflow = StateGraph(AgentTurnState)

    # Add the nodes
    workflow.add_node("analyze_sentiment", analyze_perception_sentiment_node)
    workflow.add_node("prepare_relationship_prompt", prepare_relationship_prompt_node)
    workflow.add_node("generate_action_output", generate_thought_and_message_node)
    workflow.add_node("handle_propose_idea", handle_propose_idea_node) # Specific intent handler
    workflow.add_node("handle_ask_clarification", handle_propose_idea_node) # Temporarily reuse propose_idea handler
    workflow.add_node("handle_continue_collaboration", handle_continue_collaboration_node) # New handler
    workflow.add_node("handle_idle", handle_idle_node) # New handler
    workflow.add_node("update_state", update_state_node) # Unified state update

    # Define edges
    workflow.set_entry_point("analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "prepare_relationship_prompt")
    workflow.add_edge("prepare_relationship_prompt", "generate_action_output")

    # Conditional edge based on Action Intent
    workflow.add_conditional_edges(
        "generate_action_output", # Source node after LLM output
        route_action_intent,      # Function checking structured_output.action_intent
        {
            "handle_propose_idea": "handle_propose_idea",
            "handle_ask_clarification": "handle_ask_clarification",
            "handle_continue_collaboration": "handle_continue_collaboration",
            "handle_idle": "handle_idle",
            "update_state": "update_state" # Direct fallback for unknown intents
        }
    )

    # Edges from intent handlers to the final update node
    workflow.add_edge("handle_propose_idea", "update_state")
    workflow.add_edge("handle_ask_clarification", "update_state")
    workflow.add_edge("handle_continue_collaboration", "update_state")
    workflow.add_edge("handle_idle", "update_state")

    # Final update node goes to END
    workflow.add_edge("update_state", END)

    graph = workflow.compile()
    logger.info("Agent graph with comprehensive intent routing compiled successfully.")
    return graph

# Compile the graph
basic_agent_graph_compiled = create_basic_agent_graph() 