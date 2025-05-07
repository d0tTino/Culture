"""
Defines the Simulation class responsible for managing agents and the simulation loop.
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, TYPE_CHECKING, Optional

# Use TYPE_CHECKING to avoid circular import issues if Agent needs Simulation later
if TYPE_CHECKING:
    from src.agents.core.base_agent import Agent
    from src.infra.memory.vector_store import ChromaVectorStoreManager
    from src.interfaces.discord_bot import SimulationDiscordBot

from src.sim.knowledge_board import KnowledgeBoard

# Configure the logger for this module
logger = logging.getLogger(__name__)

class Simulation:
    """
    Manages the simulation environment, agents, and time steps.
    """
    def __init__(self, agents: List['Agent'], vector_store_manager: Optional['ChromaVectorStoreManager'] = None, 
                scenario: str = "", discord_bot: Optional['SimulationDiscordBot'] = None):
        """
        Initializes the Simulation instance.

        Args:
            agents (List[Agent]): A list of Agent instances participating
                                  in the simulation.
            vector_store_manager (Optional[ChromaVectorStoreManager]): Manager for 
                                  vector-based agent memory storage and retrieval.
            scenario (str): Description of the simulation scenario that provides context for agent interactions.
            discord_bot (Optional[SimulationDiscordBot]): Discord bot for sending simulation updates to Discord.
        """
        self.agents: List['Agent'] = agents
        self.current_step: int = 0
        # Add other simulation-wide state if needed (e.g., environment properties)
        # self.environment_state = {}
        
        # --- Store the simulation scenario ---
        self.scenario = scenario
        if scenario:
            logger.info(f"Simulation initialized with scenario: {scenario}")
        else:
            logger.warning("Simulation initialized without a scenario description.")
        
        # --- NEW: Initialize Knowledge Board ---
        self.knowledge_board = KnowledgeBoard()
        logger.info("Simulation initialized with Knowledge Board.")
        
        # --- Store the vector store manager ---
        self.vector_store_manager = vector_store_manager
        if vector_store_manager:
            logger.info("Simulation initialized with vector store manager for memory persistence.")
        else:
            logger.warning("Simulation initialized without vector store manager. Memory will not be persisted.")
        
        # --- Store Discord bot ---
        self.discord_bot = discord_bot
        if discord_bot:
            logger.info("Simulation initialized with Discord bot for sending updates.")
        else:
            logger.info("Simulation initialized without Discord bot. No Discord updates will be sent.")
        
        # --- Store broadcasts from the previous step ---
        self.last_step_broadcasts: List[Dict[str, Any]] = []
        logger.info("Initialized storage for last step's broadcasts.")
        # --- End NEW ---

        if not self.agents:
            logger.warning("Simulation initialized with zero agents.")
        else:
            logger.info(f"Simulation initialized with {len(self.agents)} agents:")
            for agent in self.agents:
                logger.info(f"  - {agent.get_id()}")

    async def send_discord_update(self, message: str):
        """
        Send an update to Discord if the discord_bot is available.
        
        Args:
            message (str): The message to send to Discord
        """
        if self.discord_bot:
            # Use asyncio.create_task to avoid blocking the simulation
            asyncio.create_task(self.discord_bot.send_simulation_update(message))
        
    def run_step(self, max_turns: int = 1) -> int:
        """
        Runs the simulation for a specified number of steps.

        Args:
            max_turns (int): Maximum number of turns to run.

        Returns:
            int: The actual number of turns executed.
        """
        steps_executed = 0

        for _ in range(max_turns):
            self.current_step += 1
            current_step = self.current_step
            logger.info(f"\n--- Starting Simulation Step {current_step} ---")
            
            # Send step start update to Discord
            if self.discord_bot:
                asyncio.create_task(self.discord_bot.send_simulation_update(
                    f"📊 **Starting Simulation Step {current_step}**"
                ))

            # Get state from all agents for shared perceptions
            other_agents_state = []
            for agent in self.agents:
                # Basic state - ID, mood, step counter only for now
                agent_public_state = {
                    "agent_id": agent.agent_id,
                    "mood": agent.state.get('mood', 'neutral'),
                    "descriptive_mood": agent.state.get('descriptive_mood', 'neutral'),
                    "step_counter": agent.state.get('step_counter', 0)
                }
                other_agents_state.append(agent_public_state)

            # --- Access Knowledge Board State ---
            board_state = []
            if self.knowledge_board:
                # Get the board state (list of entries) for passing to agents
                board_state = self.knowledge_board.get_state()
                logger.debug(f"Retrieved knowledge board state: {len(board_state)} entries")
            # --- End Access Knowledge Board ---

            # --- Collect Messages from Previous Step ---
            # At the start of the step, the last_step_messages list contains
            # all messages sent during the previous step
            last_step_messages = getattr(self, 'last_step_messages', [])
            # --- End Collection ---

            # --- Process Each Agent's Turn ---
            # A list to store all messages generated in this step (for next step)
            this_step_messages = []
            
            for agent in self.agents:
                agent_id = agent.agent_id
                logger.info(f"Running turn for Agent {agent_id} at step {current_step}...")
                
                # Check and report agent's current state before turn
                current_role = agent.state.get('current_role', 'Unknown')
                current_ip = agent.state.get('influence_points', 0)
                current_du = agent.state.get('data_units', 0)
                current_mood = agent.state.get('descriptive_mood', 'neutral')
                
                # Prepare the perception dictionary
                perception_data = {
                    # Add other perception data here if needed in the future
                    "other_agents_state": other_agents_state, # List of dicts with richer info
                    # --- ADD perceived messages ---
                    # Filter messages: only broadcasts (recipient_id=None) OR 
                    # messages specifically targeted to this agent
                    "perceived_messages": [
                        msg for msg in last_step_messages
                        if msg.get('recipient_id') is None or msg.get('recipient_id') == agent_id
                    ],
                    # --- Add Knowledge Board content ---
                    "knowledge_board_content": board_state, # Pass the current board state
                    # --- Add simulation scenario ---
                    "scenario_description": self.scenario # Pass the simulation scenario
                    # --- End ADD ---
                }

                # Run the agent's turn with perception data
                agent_output = agent.run_turn(
                    simulation_step=current_step,
                    environment_perception=perception_data,
                    vector_store_manager=self.vector_store_manager,
                    knowledge_board=self.knowledge_board
                )
                
                # --- Process Agent Output ---
                # Add any broadcasts/messages to the collection for next step
                message_content = agent_output.get('message_content')
                message_recipient_id = agent_output.get('message_recipient_id')
                action_intent = agent_output.get('action_intent', 'idle')
                
                if message_content:
                    message_type = "broadcast" if message_recipient_id is None else f"targeted to {message_recipient_id}"
                    logger.info(f"Agent {agent_id} sent a message ({message_type}): '{message_content}'")
                    
                    # Store the message for the next step's perception
                    this_step_messages.append({
                        'sender_id': agent_id,
                        'content': message_content,
                        'recipient_id': message_recipient_id
                    })
                    
                    # Send message update to Discord
                    if self.discord_bot and message_recipient_id is None:  # Only send broadcast messages
                        asyncio.create_task(self.discord_bot.send_simulation_update(
                            f"💬 **Agent {agent_id}** ({current_role}): \"{message_content}\""
                        ))
                
                # Check for role changes
                new_role = agent.state.get('current_role')
                if new_role != current_role:
                    logger.info(f"Agent {agent_id} changed role from {current_role} to {new_role}")
                    
                    # Send role change update to Discord
                    if self.discord_bot:
                        asyncio.create_task(self.discord_bot.send_simulation_update(
                            f"🔄 **Role Change**: Agent {agent_id} changed from **{current_role}** to **{new_role}**"
                        ))
                
                # Check for significant IP changes
                new_ip = agent.state.get('influence_points', 0)
                ip_change = new_ip - current_ip
                if abs(ip_change) >= 5:  # Only report significant IP changes
                    ip_change_str = f"+{ip_change}" if ip_change > 0 else f"{ip_change}"
                    logger.info(f"Agent {agent_id} had significant IP change: {ip_change_str} (now {new_ip})")
                    
                    # Send IP change update to Discord
                    if self.discord_bot:
                        emoji = "📈" if ip_change > 0 else "📉"
                        asyncio.create_task(self.discord_bot.send_simulation_update(
                            f"{emoji} **IP Change**: Agent {agent_id} ({new_role}): {ip_change_str} IP (now {new_ip} IP)"
                        ))
                
                # Check for significant DU changes
                new_du = agent.state.get('data_units', 0)
                du_change = new_du - current_du
                if abs(du_change) >= 3:  # Only report significant DU changes
                    du_change_str = f"+{du_change}" if du_change > 0 else f"{du_change}"
                    logger.info(f"Agent {agent_id} had significant DU change: {du_change_str} (now {new_du})")
                    
                    # Send DU change update to Discord
                    if self.discord_bot:
                        emoji = "⬆️" if du_change > 0 else "⬇️"
                        asyncio.create_task(self.discord_bot.send_simulation_update(
                            f"{emoji} **DU Change**: Agent {agent_id} ({new_role}): {du_change_str} DU (now {new_du} DU)"
                        ))
                
                # Check for Knowledge Board posts
                if action_intent == 'propose_idea' and message_content:
                    # Send Knowledge Board update to Discord
                    if self.discord_bot:
                        # Truncate long messages
                        content_preview = message_content[:150] + ("..." if len(message_content) > 150 else "")
                        asyncio.create_task(self.discord_bot.send_simulation_update(
                            f"📝 **Knowledge Board**: Agent {agent_id} ({new_role}) posted:\n> {content_preview}"
                        ))
                # --- End Process Agent Output ---

            # --- End of Step Processing ---
            # Save the messages for the next step's perception
            self.last_step_messages = this_step_messages
            logger.info(f"Collected {len(this_step_messages)} messages for next step's perception")
            logger.info(f"--- Completed Simulation Step {current_step} ---\n")
            
            # Send step end update to Discord
            if self.discord_bot:
                # Create a summary message with agent states
                agent_summaries = []
                for agent in self.agents:
                    agent_id = agent.agent_id
                    role = agent.state.get('current_role', 'Unknown')
                    ip = agent.state.get('influence_points', 0)
                    du = agent.state.get('data_units', 0)
                    mood = agent.state.get('descriptive_mood', 'neutral')
                    agent_summaries.append(f"• **Agent {agent_id}** ({role}): {ip} IP, {du} DU, Mood: {mood}")
                
                summary = "\n".join(agent_summaries)
                asyncio.create_task(self.discord_bot.send_simulation_update(
                    f"✅ **Completed Simulation Step {current_step}**\n\n**Agent Status:**\n{summary}"
                ))
            
            steps_executed += 1
            # --- End of Step Processing ---

        return steps_executed


    def run(self, num_steps: int):
        """
        Runs the simulation for a specified number of steps.

        Args:
            num_steps (int): The total number of steps to simulate.
        """
        logger.info(f"Starting simulation run for {num_steps} steps.")
        if num_steps <= 0:
            logger.warning("Number of steps must be positive.")
            return

        # Send simulation start update to Discord
        if self.discord_bot:
            asyncio.create_task(self.discord_bot.send_simulation_update(
                f"🚀 **Starting Culture Simulation Run**\n"
                f"Steps: {num_steps}\n"
                f"Agents: {len(self.agents)}\n"
                f"Scenario: {self.scenario[:150]}..."
            ))

        for step in range(num_steps):
            self.run_step()
            time.sleep(0.2)  # Small delay between steps for clearer logging
            # Add condition checks here to stop early if needed
            # (e.g., all agents inactive, goal achieved)

        # Send simulation end update to Discord
        if self.discord_bot:
            asyncio.create_task(self.discord_bot.send_simulation_update(
                f"🏁 **Simulation Run Completed**\n"
                f"Completed {self.current_step} steps."
            ))

        logger.info(f"Simulation run finished after {self.current_step} steps.")


    # --- Optional helper methods for future use ---
    # def get_environment_view(self, agent: 'Agent'):
    #     """Provides the agent with its perception of the environment."""
    #     # To be implemented: return relevant state based on agent position, sensors etc.
    #     return {"global_time": self.current_step}

    # def execute_action(self, agent: 'Agent', action: str):
    #     """Handles the execution of an agent's chosen action."""
    #     # To be implemented: update agent state, environment state based on action
    #     logger.info(f"Agent {agent.get_id()} performs action: {action}")

    # def update_environment(self):
    #      """Updates the global environment state after agent actions."""
    #      # To be implemented
    #      pass 