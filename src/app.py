# src/app.py - Modified for relationship dynamics verification
"""
Temporary verification file for relationship dynamics in the Culture.ai project.
Contains test cases for verifying the new relationship updates.
Run each test case separately and observe the results.
"""

import os
import sys
import logging
import argparse
import time
from typing import List, Dict, Any, Optional
import json

# Configure logging for verification
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set specific module loggers to higher levels for clarity
logging.getLogger('src.infra.llm_client').setLevel(logging.WARNING)
logging.getLogger('src.agents.core.agent_state').setLevel(logging.INFO)

# Import simulation components
from src.sim.simulation import Simulation
from src.sim.knowledge_board import KnowledgeBoard
from src.infra.llm_client import get_ollama_client
from src.agents.core.base_agent import Agent
from src.agents.memory.vector_store import ChromaVectorStoreManager

# Try to import discord bot if present and enabled
try:
    from src.interfaces.discord_bot import DiscordBot
except ImportError:
    logging.warning("Discord bot module not found, will run without Discord integration.")
    DiscordBot = None

# === TEMPORARY: Force agent-to-agent messages for DSPy relationship updater spot-check ===
FORCED_INTERACTION_SCENARIO = (
    "Agents Alice, Bob, and Carol must discuss and agree on a topic for their next group project. "
    "Alice should initiate by asking Bob for his ideas. Bob should respond to Alice. "
    "Carol should then comment on Bob's idea to Alice. All agents should address each other directly in their first message."
)

# Replace the default scenario for this test run
DEFAULT_SCENARIO = FORCED_INTERACTION_SCENARIO

# Test scenario for relationship verification
VERIFICATION_SCENARIO = "The team is collaboratively designing a specification for a communication protocol. Each agent should contribute ideas and feedback while being aware of their relationships with others."

# --- Dark Forest Hypothesis Scenario ---
DARK_FOREST_SCENARIO = (
    "In a vast galaxy filled with unknown civilizations, each agent must decide whether to broadcast their existence, remain hidden, or preemptively attack others. "
    "Revealing oneself may attract allies or deadly enemies. Hiding may ensure survival but limit opportunities. "
    "Agents have incomplete information about others' intentions and must weigh the risks of communication, cooperation, and aggression. "
    "The simulation explores the consequences of the 'dark forest' hypothesis: in a universe where any contact could be fatal, what strategies emerge?"
)

def create_base_simulation(
    scenario: str = VERIFICATION_SCENARIO,
    num_agents: int = 3,
    steps: int = 10,
    use_discord: bool = False,
    use_vector_store: bool = False,
    vector_store_dir: str = "./chroma_db"
) -> Simulation:
    """
    Creates a baseline simulation with the specified number of agents.
    
    Args:
        scenario: The scenario description
        num_agents: Number of agents in the simulation
        steps: Number of steps to run
        use_discord: Whether to use Discord for output
        use_vector_store: Whether to use vector store for memory
        vector_store_dir: Directory path for ChromaDB persistence (default: ./chroma_db)
    
    Returns:
        A configured Simulation instance
    """
    # Check Ollama availability
    ollama_client = get_ollama_client()
    if not ollama_client:
        logging.error("Failed to connect to Ollama. Please ensure Ollama is running.")
        sys.exit(1)
    
    # Initialize Discord bot if requested
    discord_bot = None
    if use_discord and DiscordBot:
        discord_bot = DiscordBot()
        if not discord_bot.is_ready:
            logging.warning("Discord bot not ready, will run without Discord integration.")
            discord_bot = None
    
    # Create the simulation
    kb = KnowledgeBoard()
    
    # Create agents first
    agents = []
    for i in range(1, num_agents + 1):
        agent_id = f"agent_{i}"
        agent_name = f"Agent_{i}"
        agent = Agent(agent_id=agent_id, name=agent_name)
        agents.append(agent)
    
    # Create the simulation with the agents
    sim = Simulation(
        agents=agents,
        vector_store_manager=None if not use_vector_store else ChromaVectorStoreManager(persist_directory=vector_store_dir),
        scenario=scenario,
        discord_bot=discord_bot
    )
    
    # Set the knowledge board
    sim.knowledge_board = kb
    
    # Set the number of steps
    sim.steps_to_run = steps
    
    return sim

def test_case_1_positive_targeted(use_discord=False):
    """
    Test Case 1: Positive Interaction (Targeted)
    
    Tests that a targeted positive message correctly updates the
    relationship scores for both parties with the targeted multiplier.
    """
    logging.info("STARTING TEST CASE 1: POSITIVE TARGETED INTERACTION")
    
    # Create simulation with 2 agents
    sim = create_base_simulation(num_agents=2, steps=5, use_discord=use_discord)
    
    # Directly manipulate the relationship scores to simulate a positive targeted message
    logging.info("DIRECTLY FORCING A POSITIVE RELATIONSHIP UPDATE (SIMULATING TARGETED MESSAGE)")
    sim.agents[0].state.relationships["agent_2"] = 0.0  # Start neutral
    
    # Verify the targeted_message_multiplier value
    targeted_multiplier = sim.agents[0].state.targeted_message_multiplier
    logging.info(f"TARGETED_MESSAGE_MULTIPLIER = {targeted_multiplier}")
    
    # Update relationship with positive delta and targeted=True
    positive_delta = 0.15  # Sentiment value equivalent to "positive"
    sim.agents[0].update_relationship("agent_2", positive_delta, is_targeted=True)
    
    # Run the simulation steps
    sim.run(5)
    
    # Print the final relationship states
    logging.info("TEST CASE 1 VERIFICATION: FINAL RELATIONSHIP STATES")
    for agent in sim.agents:
        logging.info(f"Agent {agent.agent_id} relationships: {agent.state.relationships}")
    
    logging.info("TEST CASE 1 COMPLETED")

def test_case_2_negative_targeted(use_discord=False):
    """
    Test Case 2: Negative Interaction (Targeted)
    
    Tests that a targeted negative message correctly updates the
    relationship scores for both parties with the targeted multiplier.
    """
    logging.info("STARTING TEST CASE 2: NEGATIVE TARGETED INTERACTION")
    
    # Create simulation with 2 agents
    sim = create_base_simulation(num_agents=2, steps=5, use_discord=use_discord)
    
    # Directly manipulate the relationship scores to simulate a negative targeted message
    logging.info("DIRECTLY FORCING A NEGATIVE RELATIONSHIP UPDATE (SIMULATING TARGETED MESSAGE)")
    sim.agents[0].state.relationships["agent_2"] = 0.0  # Start neutral
    
    # Verify the targeted_message_multiplier value
    targeted_multiplier = sim.agents[0].state.targeted_message_multiplier
    logging.info(f"TARGETED_MESSAGE_MULTIPLIER = {targeted_multiplier}")
    
    # Update relationship with negative delta and targeted=True
    negative_delta = -0.2  # Sentiment value equivalent to "negative"
    sim.agents[0].update_relationship("agent_2", negative_delta, is_targeted=True)
    
    # Run the simulation steps
    sim.run(5)
    
    # Print the final relationship states
    logging.info("TEST CASE 2 VERIFICATION: FINAL RELATIONSHIP STATES")
    for agent in sim.agents:
        logging.info(f"Agent {agent.agent_id} relationships: {agent.state.relationships}")
    
    logging.info("TEST CASE 2 COMPLETED")

def test_case_3_neutral_targeted(use_discord=False):
    """
    Test Case 3: Neutral Interaction (Targeted)
    
    Tests that a targeted neutral message correctly updates the
    relationship scores (or lack thereof) for both parties.
    """
    logging.info("STARTING TEST CASE 3: NEUTRAL TARGETED INTERACTION")
    
    # Create simulation with 2 agents
    sim = create_base_simulation(num_agents=2, steps=5, use_discord=use_discord)
    
    # Start with a non-neutral relationship to better observe the effect of neutral message
    logging.info("INITIALIZING NON-NEUTRAL RELATIONSHIP AND SENDING NEUTRAL MESSAGE")
    sim.agents[0].state.relationships["agent_2"] = 0.3  # Start positive
    
    # Log the initial relationship
    logging.info("INITIAL RELATIONSHIP STATE:")
    logging.info(f"Agent agent_1 relationships: {sim.agents[0].state.relationships}")
    from src.infra.config import get_relationship_label
    initial_label = get_relationship_label(sim.agents[0].state.relationships.get("agent_2", 0.0))
    logging.info(f"Initial relationship label: {initial_label}")
    
    # Use the proper method from base_agent with numeric delta (0.0 = neutral)
    logging.info(f"Agent agent_1 sending neutral targeted message to agent_2")
    sim.agents[0].update_relationship("agent_2", 0.0, True)
    
    # Log the updated relationship
    logging.info("RELATIONSHIP STATES AFTER UPDATE:")
    logging.info(f"Agent agent_1 relationships: {sim.agents[0].state.relationships}")
    updated_label = get_relationship_label(sim.agents[0].state.relationships.get("agent_2", 0.0))
    logging.info(f"Updated relationship label: {updated_label}")
    
    # Run the simulation for a few steps
    sim.run(5)
    
    # Log final relationship states
    logging.info("TEST CASE 3 VERIFICATION: FINAL RELATIONSHIP STATES")
    logging.info(f"Agent {sim.agents[0].agent_id} relationships: {sim.agents[0].state.relationships}")
    logging.info(f"Agent {sim.agents[1].agent_id} relationships: {sim.agents[1].state.relationships}")
    
    logging.info("TEST CASE 3 COMPLETED")

def test_case_4_broadcast(use_discord=False):
    """
    Test Case 4: Broadcast Interaction
    
    Tests that broadcast messages have a lower impact on relationship scores
    compared to targeted messages.
    """
    logging.info("STARTING TEST CASE 4: BROADCAST INTERACTION")
    
    # Create simulation with 3 agents to test broadcast vs targeted
    sim = create_base_simulation(num_agents=3, steps=5, use_discord=use_discord)
    
    # Set initial relationships to neutral
    sim.agents[0].state.relationships["agent_2"] = 0.0
    sim.agents[0].state.relationships["agent_3"] = 0.0
    
    # Log initial relationship states
    logging.info("INITIAL RELATIONSHIP STATES:")
    logging.info(f"Agent agent_1 -> agent_2: {sim.agents[0].state.relationships.get('agent_2', 0.0)}")
    logging.info(f"Agent agent_1 -> agent_3: {sim.agents[0].state.relationships.get('agent_3', 0.0)}")
    
    # First test: targeted positive update to agent_2
    logging.info("PERFORMING TARGETED POSITIVE UPDATE TO AGENT_2")
    sim.agents[0].update_relationship("agent_2", 1.0, True)  # Targeted with delta=1.0
    
    # Second test: broadcast positive update (will affect agent_3)
    logging.info("PERFORMING BROADCAST POSITIVE UPDATE (AFFECTING AGENT_3)")
    sim.agents[0].update_relationship("agent_3", 1.0, False)  # Broadcast with delta=1.0
    
    # Log the updated relationships
    logging.info("RELATIONSHIP STATES AFTER UPDATES:")
    from src.infra.config import get_relationship_label
    
    agent2_score = sim.agents[0].state.relationships.get("agent_2", 0.0)
    agent3_score = sim.agents[0].state.relationships.get("agent_3", 0.0)
    
    logging.info(f"Agent agent_1 -> agent_2 (Targeted): {agent2_score} ({get_relationship_label(agent2_score)})")
    logging.info(f"Agent agent_1 -> agent_3 (Broadcast): {agent3_score} ({get_relationship_label(agent3_score)})")
    
    # Calculate ratio to verify targeted multiplier
    if agent3_score > 0:
        ratio = agent2_score / agent3_score
        logging.info(f"Targeted/Broadcast ratio: {ratio:.2f} (Expected: {sim.agents[0].state.targeted_message_multiplier:.2f})")
    
    # Run the simulation for a few steps
    sim.run(5)
    
    # Log final relationship states
    logging.info("TEST CASE 4 VERIFICATION: FINAL RELATIONSHIP STATES")
    logging.info(f"Agent {sim.agents[0].agent_id} relationships: {sim.agents[0].state.relationships}")
    
    logging.info("TEST CASE 4 COMPLETED")

def test_case_5_decay(use_discord=False):
    """
    Test Case 5: Relationship Decay
    
    Tests that relationships naturally decay toward neutral over time
    when there are no interactions.
    """
    logging.info("STARTING TEST CASE 5: RELATIONSHIP DECAY")
    
    # Create simulation with 2 agents
    sim = create_base_simulation(num_agents=2, steps=10, use_discord=use_discord)
    
    # Set initial non-neutral relationship scores
    sim.agents[0].state.relationships["agent_2"] = 0.6  # Positive relationship
    sim.agents[1].state.relationships["agent_1"] = -0.6  # Negative relationship
    
    # Configure both agents to be idle
    idle_goal = "Your primary goal is to observe the conversation without participating. Remain idle for the entire simulation, sending no messages."
    sim.agents[0].state.agent_goal = idle_goal
    sim.agents[1].state.agent_goal = idle_goal
    
    # Log initial relationship states
    logging.info("TEST CASE 5 VERIFICATION: INITIAL RELATIONSHIP STATES")
    for agent in sim.agents:
        logging.info(f"Agent {agent.agent_id} relationships: {agent.state.relationships}")
    
    # Run the simulation for more steps to observe decay
    sim.run(10)
    
    # Log final relationship states
    logging.info("TEST CASE 5 VERIFICATION: FINAL RELATIONSHIP STATES")
    for agent in sim.agents:
        logging.info(f"Agent {agent.agent_id} relationships: {agent.state.relationships}")
    
    logging.info("TEST CASE 5 COMPLETED")

def test_case_6_influence(use_discord=False):
    """
    Test Case 6: Relationship Influence on Behavior
    
    Tests that relationships influence agent behavior through the
    prompting system, qualitatively observing interactions.
    """
    logging.info("STARTING TEST CASE 6: RELATIONSHIP INFLUENCE ON BEHAVIOR")
    
    # Create simulation with more agents and steps
    sim = create_base_simulation(num_agents=4, steps=15, use_discord=use_discord)
    
    # Set up strong initial relationships (some positive, some negative)
    sim.agents[0].state.relationships["agent_2"] = 0.8  # Agent_1 strongly likes Agent_2
    sim.agents[0].state.relationships["agent_3"] = -0.8  # Agent_1 strongly dislikes Agent_3
    sim.agents[0].state.relationships["agent_4"] = 0.0  # Agent_1 is neutral toward Agent_4
    
    sim.agents[1].state.relationships["agent_1"] = 0.8  # Agent_2 strongly likes Agent_1 (reciprocal)
    sim.agents[2].state.relationships["agent_1"] = -0.6  # Agent_3 dislikes Agent_1 (asymmetric)
    
    # Configure with a goal that allows varied interactions
    collab_goal = "Your goal is to actively collaborate on the protocol design. Interact with other agents based on your relationships with them, prioritizing those you have positive relationships with and being cautious with those you have negative relationships with."
    for agent in sim.agents:
        agent.state.agent_goal = collab_goal
    
    # Log initial relationship states
    logging.info("TEST CASE 6 VERIFICATION: INITIAL RELATIONSHIP STATES")
    for agent in sim.agents:
        logging.info(f"Agent {agent.agent_id} relationships: {agent.state.relationships}")
    
    # Run the simulation for more steps to observe behavioral influence
    sim.run(15)
    
    # Log final relationship states 
    logging.info("TEST CASE 6 VERIFICATION: FINAL RELATIONSHIP STATES")
    for agent in sim.agents:
        logging.info(f"Agent {agent.agent_id} relationships: {agent.state.relationships}")
    
    logging.info("TEST CASE 6 COMPLETED")

def test_case_1_forced_direct_message(use_discord=False):
    """
    Test Case 1 (Forced): Positive Interaction (Targeted)
    
    Directly forces agent_1 to send a positive targeted message to agent_2
    to test the relationship dynamics.
    """
    logging.info("STARTING TEST CASE 1 (FORCED): POSITIVE TARGETED INTERACTION")
    
    # Create simulation with 2 agents
    sim = create_base_simulation(num_agents=2, steps=1, use_discord=use_discord)
    
    # Log initial relationship states (should be empty or zero)
    logging.info("INITIAL RELATIONSHIP STATES:")
    logging.info(f"Agent {sim.agents[0].agent_id} relationships: {sim.agents[0].state.relationships}")
    logging.info(f"Agent {sim.agents[1].agent_id} relationships: {sim.agents[1].state.relationships}")
    
    # Create the message content with strong positive sentiment
    message_content = "I really appreciate your contributions to our project. Your insights are extremely valuable and I'm grateful for your collaboration. You've done an outstanding job!"
    
    # Manually modify relationship scores directly
    logging.info("DIRECTLY MODIFYING RELATIONSHIP SCORES...")
    
    # First, update agent_1's relationship with agent_2 (outgoing sentiment)
    sim.agents[0].state.relationships["agent_2"] = 0.5  # Direct positive relationship
    logging.info(f"Set agent_1->agent_2 relationship to 0.5")
    
    # Second, update agent_2's relationship with agent_1 (incoming sentiment)
    sim.agents[1].state.relationships["agent_1"] = 0.3  # Direct positive relationship, slightly lower
    logging.info(f"Set agent_2->agent_1 relationship to 0.3")
    
    # Log final relationship states
    logging.info("FINAL RELATIONSHIP STATES AFTER DIRECT UPDATES:")
    logging.info(f"Agent {sim.agents[0].agent_id} relationships: {sim.agents[0].state.relationships}")
    logging.info(f"Agent {sim.agents[1].agent_id} relationships: {sim.agents[1].state.relationships}")
    
    # Add relationship history entries to track relationships over time
    sim.agents[0].state.update_relationship_history(1, sim.agents[0].state.relationships.copy())
    sim.agents[1].state.update_relationship_history(1, sim.agents[1].state.relationships.copy())
    
    # Verification
    from src.infra.config import get_relationship_label
    agent1_to_agent2_label = get_relationship_label(sim.agents[0].state.relationships["agent_2"])
    agent2_to_agent1_label = get_relationship_label(sim.agents[1].state.relationships["agent_1"])
    
    logging.info(f"Relationship Labels:")
    logging.info(f"Agent {sim.agents[0].agent_id} -> {sim.agents[1].agent_id}: {agent1_to_agent2_label}")
    logging.info(f"Agent {sim.agents[1].agent_id} -> {sim.agents[0].agent_id}: {agent2_to_agent1_label}")
    
    logging.info("TEST CASE 1 (FORCED) COMPLETED")

def test_case_decay_verification(use_discord=False):
    """
    Relationship Decay Verification Test
    
    Sets initial relationship scores and runs the simulation for multiple steps
    to verify relationship decay works correctly.
    """
    logging.info("STARTING RELATIONSHIP DECAY VERIFICATION TEST")
    
    # Create simulation with 2 agents and more steps to observe decay
    sim = create_base_simulation(num_agents=2, steps=6, use_discord=use_discord)
    
    # Set initial non-neutral relationship scores
    sim.agents[0].state.relationships["agent_2"] = 0.8  # Strong positive relationship
    sim.agents[1].state.relationships["agent_1"] = -0.6  # Negative relationship
    
    # Configure agents with idle goals to avoid relationship updates from messages
    idle_goal = "Remain idle and don't send any messages. Just observe the conversation."
    sim.agents[0].state.agent_goal = idle_goal
    sim.agents[1].state.agent_goal = idle_goal
    
    # Override relationship decay rate to a higher value for testing (normally 0.01)
    decay_rate = 0.1  # 10% decay per step for demonstration
    sim.agents[0].state.relationship_decay_rate = decay_rate
    sim.agents[1].state.relationship_decay_rate = decay_rate
    
    # Log initial relationship states
    logging.info("INITIAL RELATIONSHIP STATES:")
    logging.info(f"Agent {sim.agents[0].agent_id} relationships: {sim.agents[0].state.relationships}")
    logging.info(f"Agent {sim.agents[1].agent_id} relationships: {sim.agents[1].state.relationships}")
    
    # Prediction of expected values after decay
    predicted_values = []
    
    # Calculate expected values for agent_1's relationship with agent_2 after each step
    agent1_score = 0.8
    for step in range(1, 7):
        decay_amount = agent1_score * decay_rate
        agent1_score -= decay_amount
        predicted_values.append((step, agent1_score))
    
    logging.info(f"PREDICTED DECAY VALUES FOR agent_1->agent_2:")
    for step, value in predicted_values:
        logging.info(f"  Step {step}: {value:.4f}")
    
    # Run the simulation to observe decay
    sim.run(6)
    
    # Log final relationship states and history
    logging.info("FINAL RELATIONSHIP STATES AFTER DECAY:")
    logging.info(f"Agent {sim.agents[0].agent_id} relationships: {sim.agents[0].state.relationships}")
    logging.info(f"Agent {sim.agents[1].agent_id} relationships: {sim.agents[1].state.relationships}")
    
    # Print relationship history to see progression of decay
    logging.info("RELATIONSHIP HISTORY (DECAY PROGRESSION):")
    for agent_idx, agent in enumerate(sim.agents):
        logging.info(f"Agent {agent.agent_id} relationship history:")
        history = agent.state.relationship_history
        for step, relationships in history:
            logging.info(f"  Step {step}: {relationships}")
    
    logging.info("RELATIONSHIP DECAY TEST COMPLETED")

def test_case_4_broadcast_vs_targeted(use_discord=False):
    """
    Test Case 4: Broadcast vs. Targeted Interaction
    
    Tests that the targeted message multiplier correctly differentiates
    between broadcast and targeted messages.
    """
    logging.info("STARTING TEST CASE 4: BROADCAST VS TARGETED INTERACTION")
    
    # Create simulation with 3 agents
    sim = create_base_simulation(num_agents=3, steps=5, use_discord=use_discord)
    
    # Directly manipulate relationship scores to ensure known starting points
    sim.agents[0].state.relationships["agent_2"] = 0.9  # Start with high positive relationship
    sim.agents[0].state.relationships["agent_3"] = 0.3  # Start with low positive relationship
    
    # Verify the targeted_message_multiplier value
    targeted_multiplier = sim.agents[0].state.targeted_message_multiplier
    logging.info(f"TARGETED_MESSAGE_MULTIPLIER = {targeted_multiplier}")
    
    # First, send a broadcast positive message (affects all agents)
    logging.info("SENDING BROADCAST MESSAGE - should update all relationships without multiplier")
    initial_relation_agent2 = sim.agents[0].state.relationships["agent_2"]
    initial_relation_agent3 = sim.agents[0].state.relationships["agent_3"]
    
    # Send broadcast message with positive sentiment (delta = 0.15)
    sim.agents[0].update_relationship("agent_2", 0.15, is_targeted=False)  # Broadcast, not targeted
    sim.agents[0].update_relationship("agent_3", 0.15, is_targeted=False)  # Broadcast, not targeted
    
    # Get updated relationship scores after broadcast
    broadcast_relation_agent2 = sim.agents[0].state.relationships["agent_2"]
    broadcast_relation_agent3 = sim.agents[0].state.relationships["agent_3"]
    
    # Log the effect of the broadcast message
    logging.info(f"BROADCAST effect on agent_2: {initial_relation_agent2:.4f} -> {broadcast_relation_agent2:.4f} (change: {broadcast_relation_agent2 - initial_relation_agent2:.4f})")
    logging.info(f"BROADCAST effect on agent_3: {initial_relation_agent3:.4f} -> {broadcast_relation_agent3:.4f} (change: {broadcast_relation_agent3 - initial_relation_agent3:.4f})")
    
    # Now, send a targeted message to agent_2
    logging.info(f"SENDING TARGETED MESSAGE - should apply {targeted_multiplier}x multiplier")
    
    # Reset relationship scores for fair comparison
    sim.agents[0].state.relationships["agent_2"] = initial_relation_agent2
    sim.agents[0].state.relationships["agent_3"] = initial_relation_agent3
    
    # Send targeted message with same sentiment (delta = 0.15)
    sim.agents[0].update_relationship("agent_2", 0.15, is_targeted=True)  # Targeted!
    
    # Get the updated scores
    targeted_relation_agent2 = sim.agents[0].state.relationships["agent_2"]
    
    # Log the effect of the targeted message
    logging.info(f"TARGETED effect on agent_2: {initial_relation_agent2:.4f} -> {targeted_relation_agent2:.4f} (change: {targeted_relation_agent2 - initial_relation_agent2:.4f})")
    
    # Calculate and display the actual observed multiplier
    broadcast_change = broadcast_relation_agent2 - initial_relation_agent2
    targeted_change = targeted_relation_agent2 - initial_relation_agent2
    
    if broadcast_change != 0:
        observed_multiplier = targeted_change / broadcast_change
        logging.info(f"OBSERVED MULTIPLIER: {observed_multiplier:.2f}x (Expected: {targeted_multiplier:.2f}x)")
        
        # Verify that the targeted multiplier is working correctly
        if abs(observed_multiplier - targeted_multiplier) < 0.1:
            logging.info("✅ VERIFICATION PASSED: Targeted message multiplier is working correctly")
        else:
            logging.info("❌ VERIFICATION FAILED: Targeted message multiplier not applying correctly")
    else:
        logging.info("Unable to calculate multiplier - broadcast change was 0")
    
    # Run the simulation to observe relationship decay and influence on behavior
    sim.run(5)
    
    # Print final relationship states
    logging.info("TEST CASE 4 VERIFICATION: FINAL RELATIONSHIP STATES")
    logging.info(f"Agent {sim.agents[0].agent_id} relationships: {sim.agents[0].state.relationships}")
    
    logging.info("TEST CASE 4 COMPLETED")

def test_case_10_targeted_multiplier_comprehensive(use_discord=False):
    """
    Test Case 10: Comprehensive Targeted Message Multiplier Test
    
    Tests that the targeted message multiplier correctly applies to both
    positive and negative interactions with the expected 3.0x magnitude.
    """
    logging.info("STARTING TEST CASE 10: COMPREHENSIVE TARGETED MULTIPLIER TEST")
    
    # Create simulation with 3 agents
    sim = create_base_simulation(num_agents=3, steps=3, use_discord=use_discord)
    
    # Reset all relationships to neutral (0.0)
    sim.agents[0].state.relationships = {}
    
    # Get the configured multiplier
    targeted_multiplier = sim.agents[0].state.targeted_message_multiplier
    logging.info(f"TARGETED_MESSAGE_MULTIPLIER = {targeted_multiplier}")
    
    # Test 1: Positive relationship updates
    logging.info("TEST 1: POSITIVE RELATIONSHIP UPDATES")
    delta = 0.20  # Positive delta
    
    # Broadcast message (no multiplier)
    logging.info("Sending broadcast positive message (delta = 0.20)")
    initial_score = sim.agents[0].state.relationships.get("agent_2", 0.0)
    sim.agents[0].update_relationship("agent_2", delta, is_targeted=False)
    broadcast_score = sim.agents[0].state.relationships["agent_2"]
    broadcast_change = broadcast_score - initial_score
    logging.info(f"Broadcast positive change: {initial_score:.4f} → {broadcast_score:.4f} (change: {broadcast_change:.4f})")
    
    # Reset and test targeted message (with multiplier)
    sim.agents[0].state.relationships["agent_2"] = initial_score
    logging.info(f"Sending targeted positive message (delta = 0.20, multiplier = {targeted_multiplier})")
    sim.agents[0].update_relationship("agent_2", delta, is_targeted=True)
    targeted_score = sim.agents[0].state.relationships["agent_2"]
    targeted_change = targeted_score - initial_score
    logging.info(f"Targeted positive change: {initial_score:.4f} → {targeted_score:.4f} (change: {targeted_change:.4f})")
    
    # Verify multiplier effect
    if broadcast_change != 0:
        positive_ratio = targeted_change / broadcast_change
        logging.info(f"Positive ratio: {positive_ratio:.2f}x (Expected: {targeted_multiplier:.2f}x)")
        
        if abs(positive_ratio - targeted_multiplier) < 0.1:
            logging.info("✅ POSITIVE TEST PASSED: Targeted multiplier is working correctly")
        else:
            logging.info("❌ POSITIVE TEST FAILED: Targeted multiplier not applying correctly")
    
    # Test 2: Negative relationship updates
    logging.info("\nTEST 2: NEGATIVE RELATIONSHIP UPDATES")
    delta = -0.25  # Negative delta
    
    # Reset relationships for negative test
    sim.agents[0].state.relationships = {}
    initial_score = sim.agents[0].state.relationships.get("agent_3", 0.0)
    
    # Broadcast message (no multiplier)
    logging.info("Sending broadcast negative message (delta = -0.25)")
    sim.agents[0].update_relationship("agent_3", delta, is_targeted=False)
    broadcast_score = sim.agents[0].state.relationships["agent_3"]
    broadcast_change = broadcast_score - initial_score
    logging.info(f"Broadcast negative change: {initial_score:.4f} → {broadcast_score:.4f} (change: {broadcast_change:.4f})")
    
    # Reset and test targeted message (with multiplier)
    sim.agents[0].state.relationships["agent_3"] = initial_score
    logging.info(f"Sending targeted negative message (delta = -0.25, multiplier = {targeted_multiplier})")
    sim.agents[0].update_relationship("agent_3", delta, is_targeted=True)
    targeted_score = sim.agents[0].state.relationships["agent_3"]
    targeted_change = targeted_score - initial_score
    logging.info(f"Targeted negative change: {initial_score:.4f} → {targeted_score:.4f} (change: {targeted_change:.4f})")
    
    # Verify multiplier effect (note: both changes are negative, so we compare absolute values)
    if broadcast_change != 0:
        negative_ratio = targeted_change / broadcast_change
        logging.info(f"Negative ratio: {negative_ratio:.2f}x (Expected: {targeted_multiplier:.2f}x)")
        
        if abs(negative_ratio - targeted_multiplier) < 0.1:
            logging.info("✅ NEGATIVE TEST PASSED: Targeted multiplier is working correctly")
        else:
            logging.info("❌ NEGATIVE TEST FAILED: Targeted multiplier not applying correctly")
            
    logging.info("TEST CASE 10 COMPLETED")

def test_case_11_dark_forest(use_discord=False):
    """
    Test Case 11: Dark Forest Hypothesis Scenario
    Agents must choose between hiding, broadcasting, or attacking, with incomplete information and existential risk.
    """
    logging.info("STARTING TEST CASE 11: DARK FOREST HYPOTHESIS SCENARIO")
    sim = create_base_simulation(
        scenario=DARK_FOREST_SCENARIO,
        num_agents=4,
        steps=8,
        use_discord=use_discord
    )
    # Optionally, you could customize agent goals or state here for more realism
    sim.run(8)
    logging.info("TEST CASE 11 COMPLETED: See logs for agent strategies and outcomes.")

def main():
    """
    Main entry point for the application.
    """
    import argparse
    from src.infra.logging_config import setup_logging
    # --- DSPy LM configuration ---
    from src.infra.dspy_ollama_integration import configure_dspy_with_ollama
    lm = configure_dspy_with_ollama(model_name="mistral:latest", temperature=0.1)
    import dspy
    import logging
    logging.getLogger("dspy_ollama").info(f"[DSPy] LM configured at app startup: {dspy.settings.lm}")
    # --- end DSPy LM configuration ---
    # Setup logging with our custom configuration
    root_logger, llm_perf_logger = setup_logging(log_dir="logs")
    
    parser = argparse.ArgumentParser(description='Run the Culture.ai simulation with different test cases')
    parser.add_argument('test_case', type=int, choices=list(range(1,12)), 
                        help='Test case to run (1-11, where 11 is the dark forest scenario)')
    parser.add_argument('--discord', action='store_true', help='Enable Discord integration')
    parser.add_argument('--log', choices=['debug', 'info', 'warning'], default='info',
                        help='Set logging level')
    
    args = parser.parse_args()
    
    # Set logging level for root logger (won't affect llm_perf_logger which has separate handlers)
    if args.log == 'debug':
        root_logger.setLevel(logging.DEBUG)
    elif args.log == 'info':
        root_logger.setLevel(logging.INFO)
    elif args.log == 'warning':
        root_logger.setLevel(logging.WARNING)
    
    # Log that LLM performance monitoring is active
    llm_perf_logger.info("LLM performance monitoring initialized")
    
    use_discord = args.discord
    test_case = args.test_case
    
    # Run the selected test case
    if test_case == 1:
        test_case_1_positive_targeted(use_discord)
    elif test_case == 2:
        test_case_2_negative_targeted(use_discord)
    elif test_case == 3:
        test_case_3_neutral_targeted(use_discord)
    elif test_case == 4:
        test_case_4_broadcast(use_discord)
    elif test_case == 5:
        test_case_5_decay(use_discord)
    elif test_case == 6:
        test_case_6_influence(use_discord)
    elif test_case == 7:
        test_case_1_forced_direct_message(use_discord)
    elif test_case == 8:
        test_case_decay_verification(use_discord)
    elif test_case == 9:
        test_case_4_broadcast_vs_targeted(use_discord)
    elif test_case == 10:
        test_case_10_targeted_multiplier_comprehensive(use_discord)
    elif test_case == 11:
        test_case_11_dark_forest(use_discord)
    else:
        logging.error(f"Invalid test case: {test_case}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Unhandled exception in main application: {e}", exc_info=True)
        import sys
        sys.exit(1) 