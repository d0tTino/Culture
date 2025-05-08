"""
Defines the Simulation class responsible for managing agents and the simulation loop.
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, TYPE_CHECKING, Optional
import discord
from src.infra import config  # Import to access MAX_PROJECT_MEMBERS

# Use TYPE_CHECKING to avoid circular import issues if Agent needs Simulation later
if TYPE_CHECKING:
    from src.agents.core.base_agent import Agent
    from src.infra.memory.vector_store import ChromaVectorStoreManager
    from src.interfaces.discord_bot import SimulationDiscordBot

from src.agents.core.agent_state import AgentState
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
        
        # --- NEW: Initialize Project Tracking ---
        self.projects = {}  # Structure: {project_id: {name, creator_id, members}}
        logger.info("Simulation initialized with project tracking system.")
        
        # --- NEW: Initialize Collective Metrics ---
        self.collective_ip: float = 0.0
        self.collective_du: float = 0.0
        logger.info("Simulation initialized with collective IP/DU tracking.")
        
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
            
            # Initialize collective metrics based on starting agent states
            self._update_collective_metrics()
            logger.info(f"Initial collective metrics - IP: {self.collective_ip:.1f}, DU: {self.collective_du:.1f}")

    # Add method to update collective metrics
    def _update_collective_metrics(self):
        """
        Updates the collective IP and DU metrics by summing across all agents.
        """
        total_ip = 0.0
        total_du = 0.0
        
        for agent in self.agents:
            agent_state = agent.state
            total_ip += agent_state.ip
            total_du += agent_state.du
        
        self.collective_ip = total_ip
        self.collective_du = total_du

    async def send_discord_update(self, message: Optional[str] = None, embed: Optional[discord.Embed] = None):
        """
        Send an update to Discord if the discord_bot is available.
        
        Args:
            message (Optional[str]): The text message to send to Discord
            embed (Optional[discord.Embed]): The embed object to send to Discord
        """
        if self.discord_bot:
            # Use asyncio.create_task to avoid blocking the simulation
            asyncio.create_task(self.discord_bot.send_simulation_update(content=message, embed=embed))
        
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
                step_start_embed = self.discord_bot.create_step_start_embed(current_step)
                asyncio.create_task(self.discord_bot.send_simulation_update(embed=step_start_embed))

            # Get state from all agents for shared perceptions
            other_agents_state = []
            for agent in self.agents:
                # Get key information from the agent's state for perception
                agent_state = agent.state
                agent_public_state = {
                    "agent_id": agent.agent_id,
                    "name": agent_state.name,
                    "role": agent_state.role,
                    "mood": agent_state.mood,
                    "descriptive_mood": getattr(agent_state, 'descriptive_mood', 'neutral'),
                    "step_counter": getattr(agent_state, 'actions_taken_count', 0),
                    "current_project_id": agent_state.current_project_id
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
                agent_state = agent.state
                logger.info(f"Running turn for Agent {agent_id} at step {current_step}...")
                
                # Check and report agent's current state before turn
                current_role = agent_state.role
                current_ip = agent_state.ip
                current_du = agent_state.du
                current_mood = agent_state.mood
                
                # Update agent's perception of collective metrics
                agent_state.update_collective_metrics(self.collective_ip, self.collective_du)
                
                # Ensure agent has access to llm_client for memory operations
                if hasattr(self, 'llm_client') and self.llm_client:
                    agent_state.llm_client = self.llm_client
                
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
                    "scenario_description": self.scenario, # Pass the simulation scenario
                    # --- Add available projects ---
                    "available_projects": self.get_project_details(), # Pass all projects for perception
                    # --- Add simulation reference for project actions ---
                    "simulation": self, # Pass the simulation instance for project operations
                    # --- Add collective metrics ---
                    "collective_ip": self.collective_ip,
                    "collective_du": self.collective_du
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
                        'step': current_step,
                        'sender_id': agent_id,
                        'recipient_id': message_recipient_id,
                        'content': message_content,
                        'action_intent': action_intent
                    })
                    
                    # Send message to Discord if integration enabled
                    if self.discord_bot:
                        message_embed = self.discord_bot.create_agent_message_embed(
                            agent_id=agent_id, 
                            message_content=message_content,
                            recipient_id=message_recipient_id,
                            action_intent=action_intent,
                            agent_role=current_role,
                            mood=current_mood,
                            step=current_step
                        )
                        asyncio.create_task(self.discord_bot.send_simulation_update(embed=message_embed))
                
                # Process action intent if any
                if action_intent and action_intent != 'idle':
                    logger.info(f"Agent {agent_id} performed action intent: {action_intent}")
                    
                    # Send intent action to Discord if integration enabled
                    if self.discord_bot and not message_content:  # Only if no message was sent
                        action_embed = self.discord_bot.create_agent_action_embed(
                            agent_id=agent_id, 
                            action_intent=action_intent,
                            agent_role=current_role,
                            mood=current_mood,
                            step=current_step
                        )
                        asyncio.create_task(self.discord_bot.send_simulation_update(embed=action_embed))
                
                # Update agent counters and report
                logger.info(f"Agent {agent_id} completed turn {current_step}:")
                logger.info(f"  - Role: {current_role} (steps in role: {agent_state.steps_in_current_role})")
                logger.info(f"  - Mood: {current_mood}")
                logger.info(f"  - IP: {agent_state.ip:.1f} (from {current_ip:.1f})")
                logger.info(f"  - DU: {agent_state.du:.1f} (from {current_du:.1f})")
                
            # Store the messages from this step for next step's perception
            self.last_step_messages = this_step_messages
            
            # Update collective metrics after all agents have taken their turns
            self._update_collective_metrics()
            logger.info(f"Step {current_step} collective metrics - IP: {self.collective_ip:.1f}, DU: {self.collective_du:.1f}")
            
            # Send step end update to Discord
            if self.discord_bot:
                step_end_embed = self.discord_bot.create_step_end_embed(
                    step=current_step, 
                    collective_ip=self.collective_ip,
                    collective_du=self.collective_du
                )
                asyncio.create_task(self.discord_bot.send_simulation_update(embed=step_end_embed))
            
            steps_executed += 1
        
        return steps_executed

    def run(self, num_steps: int):
        """
        Runs the simulation for a specified number of steps.

        Args:
            num_steps (int): Number of steps to run.
        """
        logger.info(f"Starting simulation run for {num_steps} steps")
        
        total_steps_executed = 0
        start_time = time.time()
        
        for step in range(num_steps):
            steps = self.run_step(1)
            total_steps_executed += steps
            
            # Optional: Add a small delay between steps
            time.sleep(0.1)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Simulation completed {total_steps_executed} steps in {elapsed_time:.2f} seconds")

    def create_project(self, project_name: str, creator_agent_id: str, project_description: str = None) -> Optional[str]:
        """
        Creates a new project with the given name, initiated by the specified agent.
        
        Args:
            project_name (str): The name of the new project
            creator_agent_id (str): The ID of the agent creating the project
            project_description (str, optional): A description of the project's purpose
            
        Returns:
            Optional[str]: The ID of the newly created project, or None if creation failed
        """
        if not project_name or not creator_agent_id:
            logger.warning("Cannot create project: missing name or creator ID")
            return None
            
        # Create a unique project ID
        project_id = f"proj_{len(self.projects) + 1}_{int(time.time())}"
        
        # Check if a project with this name already exists
        for existing_id, existing_proj in self.projects.items():
            if existing_proj.get('name') == project_name:
                logger.warning(f"Cannot create project: a project named '{project_name}' already exists (ID: {existing_id})")
                return None
                
        # Find the creator agent
        creator_agent = None
        for agent in self.agents:
            if agent.agent_id == creator_agent_id:
                creator_agent = agent
                break
                
        if not creator_agent:
            logger.warning(f"Cannot create project: creator agent '{creator_agent_id}' not found")
            return None
            
        # Create the project
        self.projects[project_id] = {
            'id': project_id,
            'name': project_name,
            'description': project_description or f"Project created by {creator_agent_id}",
            'creator_id': creator_agent_id,
            'created_step': self.current_step,
            'members': [creator_agent_id],  # Creator is automatically a member
            'status': 'active'
        }
        
        logger.info(f"Project '{project_name}' (ID: {project_id}) created by Agent {creator_agent_id}")
        
        # Add to Knowledge Board
        if self.knowledge_board:
            project_info = f"New Project Created: {project_name}\nID: {project_id}\nCreator: {creator_agent_id}"
            if project_description:
                project_info += f"\nDescription: {project_description}"
            self.knowledge_board.add_entry(project_info, creator_agent_id, self.current_step)
        
        # Send Discord notification if bot is available
        if self.discord_bot:
            embed = self.discord_bot.create_project_embed(
                action="create", 
                project_name=project_name, 
                project_id=project_id, 
                agent_id=creator_agent_id,
                step=self.current_step
            )
            asyncio.create_task(self.discord_bot.send_simulation_update(embed=embed))
        
        return project_id

    def join_project(self, project_id: str, agent_id: str) -> bool:
        """
        Adds an agent to an existing project as a member.
        
        Args:
            project_id (str): The ID of the project to join
            agent_id (str): The ID of the agent joining the project
            
        Returns:
            bool: True if successfully joined, False otherwise
        """
        from src.infra import config  # Import to access MAX_PROJECT_MEMBERS

        # Ensure the project exists
        if project_id not in self.projects:
            logger.warning(f"Cannot join project: project ID '{project_id}' does not exist")
            return False
            
        project = self.projects[project_id]
        
        # Ensure the project is active
        if project.get('status') != 'active':
            logger.warning(f"Cannot join project: project '{project['name']}' is not active (status: {project.get('status')})")
            return False
            
        # Check if agent is already a member
        if agent_id in project['members']:
            logger.info(f"Agent {agent_id} is already a member of project '{project['name']}'")
            return True
            
        # Check if the project has reached the maximum number of members
        if len(project['members']) >= config.MAX_PROJECT_MEMBERS:
            logger.warning(f"Agent {agent_id} cannot join project '{project['name']}': maximum member limit ({config.MAX_PROJECT_MEMBERS}) reached")
            return False
            
        # Find the agent
        agent_obj = None
        for agent in self.agents:
            if agent.agent_id == agent_id:
                agent_obj = agent
                break
                
        if not agent_obj:
            logger.warning(f"Cannot join project: agent '{agent_id}' not found")
            return False
            
        # Add the agent to the project
        project['members'].append(agent_id)
        
        logger.info(f"Agent {agent_id} joined project '{project['name']}' (ID: {project_id})")
        
        # Record the event in the Knowledge Board
        if self.knowledge_board:
            join_info = f"Agent {agent_id} joined Project: {project['name']} (ID: {project_id})"
            self.knowledge_board.add_entry(join_info, agent_id, self.current_step)
            
        # Send Discord notification if bot is available
        if self.discord_bot:
            embed = self.discord_bot.create_project_embed(
                action="join", 
                project_name=project['name'], 
                project_id=project_id, 
                agent_id=agent_id,
                step=self.current_step
            )
            asyncio.create_task(self.discord_bot.send_simulation_update(embed=embed))
        
        return True

    def leave_project(self, project_id: str, agent_id: str) -> bool:
        """
        Removes an agent from a project they are currently a member of.
        
        Args:
            project_id (str): The ID of the project to leave
            agent_id (str): The ID of the agent leaving the project
            
        Returns:
            bool: True if successfully left, False otherwise
        """
        # Ensure the project exists
        if project_id not in self.projects:
            logger.warning(f"Cannot leave project: project ID '{project_id}' does not exist")
            return False
            
        project = self.projects[project_id]
        
        # Check if agent is a member
        if agent_id not in project['members']:
            logger.warning(f"Agent {agent_id} is not a member of project '{project['name']}'")
            return False
            
        # Find the agent
        agent_obj = None
        for agent in self.agents:
            if agent.agent_id == agent_id:
                agent_obj = agent
                break
                
        if not agent_obj:
            logger.warning(f"Cannot leave project: agent '{agent_id}' not found")
            return False
            
        # Remove the agent from the project
        project['members'].remove(agent_id)
        
        logger.info(f"Agent {agent_id} left project '{project['name']}' (ID: {project_id})")
        
        # Record the event in the Knowledge Board
        if self.knowledge_board:
            leave_info = f"Agent {agent_id} left Project: {project['name']} (ID: {project_id})"
            self.knowledge_board.add_entry(leave_info, agent_id, self.current_step)
            
        # Send Discord notification if bot is available
        if self.discord_bot:
            embed = self.discord_bot.create_project_embed(
                action="leave", 
                project_name=project['name'], 
                project_id=project_id, 
                agent_id=agent_id,
                step=self.current_step
            )
            asyncio.create_task(self.discord_bot.send_simulation_update(embed=embed))
        
        return True

    def get_project_details(self) -> dict:
        """
        Returns a dictionary containing details of all projects for agent perception.
        
        Returns:
            dict: Dictionary mapping project IDs to project details
        """
        return self.projects.copy()

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