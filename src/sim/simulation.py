#!/usr/bin/env python
import argparse
import asyncio
import logging
import random
import threading
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

from pydantic import ValidationError
from typing_extensions import Self

from src.agents.core.agent_controller import AgentController
from src.agents.core.agent_state import AgentActionIntent
from src.agents.memory.memory_service import MemoryService
from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaDBException
from src.governance import evaluate_policy
from src.infra import config  # Import to access MAX_PROJECT_MEMBERS
from src.infra.event_log import log_event
from src.infra.ledger import ledger
from src.infra.llm_client import get_llm_client
from src.infra.logging_config import setup_logging
from src.infra.snapshot import (
    compute_trace_hash,
    load_snapshot,
    save_snapshot,
    upload_snapshot,
)
from src.interfaces.dashboard_backend import (
    SimulationEvent,
    emit_event,
)
from src.shared.typing import SimulationMessage
from src.sim.event_kernel import EventKernel
from src.sim.graph_knowledge_board import GraphKnowledgeBoard
from src.sim.knowledge_board import KnowledgeBoard
from src.sim.quests import generate_quest
from src.sim.resource_manager import get_resource_manager
from src.sim.version_vector import VersionVector
from src.sim.world_map import WorldMap

from .event_bus import get_event_bus

# Use TYPE_CHECKING to avoid circular import issues if Agent needs Simulation later
if TYPE_CHECKING:
    from src.agents.core.base_agent import Agent
    from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
    from src.agents.memory.vector_store import ChromaVectorStoreManager
    from src.interfaces.discord_bot import SimulationDiscordBot

# Configure the logger for this module
logger = logging.getLogger(__name__)


class Simulation:
    """
    Manages the simulation environment, agents, and time steps.

    Attributes:
        steps_to_run (int): Number of steps the simulation should run (set externally).
    """

    def __init__(
        self: Self,
        agents: list["Agent"],
        memory_service: MemoryService | None = None,
        vector_store_manager: Optional["ChromaVectorStoreManager"] = None,
        semantic_manager: Optional["SemanticMemoryManager"] = None,
        scenario: str = "",
        beats: list[str] | None = None,
        discord_bot: Optional["SimulationDiscordBot"] = None,
    ) -> None:
        """
        Initializes the Simulation instance.

        Args:
            agents (list[Agent]): A list of Agent instances participating
                in the simulation.
            memory_service (MemoryService | None): Unified memory service. If
                ``None``, one will be created from ``vector_store_manager`` and
                ``semantic_manager``.
            vector_store_manager (Optional[ChromaVectorStoreManager]): Manager for
                vector-based agent memory storage and retrieval.
            scenario (str): Description of the simulation scenario that provides
                context for agent interactions.
            beats (list[str] | None): Optional beat names that segment the
                scenario into phases for evaluation.
            discord_bot (Optional[SimulationDiscordBot]): Discord bot for sending
                simulation updates to Discord.
        """
        # Reload configuration to pick up any environment overrides set in tests
        config.load_config(validate_required=False)

        self.agents: list[Agent] = agents
        self.current_step: int = 0
        self.current_agent_index: int = 0
        self.last_completed_agent_index: int | None = None
        self.steps_to_run: int = 0  # Number of steps to run, set externally
        self.total_turns_executed = 0
        self.resource_manager = get_resource_manager()
        self.simulation_complete = False
        self.event_kernel = EventKernel()
        self.vector = VersionVector()
        self.paused: bool = False
        self.speed: float = 1.0
        self.agent_initial_token_budget = int(config.get_config("AGENT_TOKEN_BUDGET"))
        self.muted_agents: set[str] = set()
        self.beats: list[str] = beats or []
        self._beat_index = 0
        self._beat_interval = 0
        # Add other simulation-wide state if needed (e.g., environment properties)
        # self.environment_state = {}

        # Background task for processing incoming events
        self._event_listener_task: asyncio.Task[None] | None = None
        self._event_task: asyncio.Task[Any] | None = None
        self._stop_listener_task: asyncio.Task[None] | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._event_loop_thread: threading.Thread | None = None

        # Lock for concurrent access to message queues
        self._msg_lock = asyncio.Lock()

        # --- Store the simulation scenario ---
        self.scenario = scenario
        if scenario:
            logger.info(f"Simulation initialized with scenario: {scenario}")
        else:
            logger.warning("Simulation initialized without a scenario description.")

        # --- NEW: Initialize Knowledge Board ---
        self.knowledge_board: GraphKnowledgeBoard | KnowledgeBoard
        if config.KNOWLEDGE_BOARD_BACKEND == "graph":
            self.knowledge_board = GraphKnowledgeBoard()
            logger.info("Simulation initialized with Graph Knowledge Board.")
        else:
            self.knowledge_board = KnowledgeBoard()
            logger.info("Simulation initialized with Knowledge Board.")

        # Initialize world map and place agents
        self.world_map = WorldMap()
        self._init_tasks: list[asyncio.Task[Any]] = []
        for idx, ag in enumerate(self.agents):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.world_map.add_agent(ag.agent_id, x=idx, y=0))
            else:
                task = loop.create_task(self.world_map.add_agent(ag.agent_id, x=idx, y=0))
                self._init_tasks.append(task)
        logger.info("Simulation initialized with world map.")

        # --- NEW: Initialize Project Tracking ---
        self.projects: dict[
            str, dict[str, Any]
        ] = {}  # Structure: {project_id: {name, creator_id, members}}

        logger.info("Simulation initialized with project tracking system.")

        # --- NEW: Initialize Collective Metrics ---
        self.collective_ip: float = 0.0
        self.collective_du: float = 0.0
        logger.info("Simulation initialized with collective IP/DU tracking.")

        # --- NEW: Evaluation hooks and metrics ---
        self.evaluation_hooks: list[Callable[[Self, list[Any]], dict[str, Any] | None]] = []
        self.metrics: list[dict[str, Any]] = []
        self.add_evaluation_hook(self._collect_metrics)

        # --- Initialize memory service ---
        if memory_service is None:
            memory_service = MemoryService(vector_store_manager, semantic_manager)
        self.memory_service = memory_service
        self.vector_store_manager = memory_service.vector_store
        self.semantic_manager = memory_service.semantic_manager
        if self.vector_store_manager:
            logger.info("Simulation initialized with vector store manager for memory persistence.")
        else:
            logger.warning(
                "Simulation initialized without vector store manager. "
                "Memory will not be persisted."
            )

        self._last_semantic_job_step = 0
        self._last_memory_prune_step = 0
        self._last_consolidation_step = 0
        self._last_quest_step = 0
        self._last_trace_hash = ""

        # --- Store Discord bot ---
        self.discord_bot = discord_bot
        if discord_bot:
            logger.info("Simulation initialized with Discord bot for sending updates.")
        else:
            logger.info(
                "Simulation initialized without Discord bot. No Discord updates will be sent."
            )

        # --- Store broadcasts from the previous step ---
        self.last_step_messages: list[SimulationMessage] = []
        logger.info("Initialized storage for last step's messages.")
        # --- End NEW ---

        # Lock to synchronize access to message buffers

        self.pending_messages_for_next_round: list[SimulationMessage] = []
        # Messages available for agents to perceive in the current round.
        self.messages_to_perceive_this_round: list[
            SimulationMessage
        ] = []  # THIS WILL BE THE ACCUMULATOR FOR THE CURRENT ROUND

        self.track_collective_metrics: bool = True

        if not self.agents:
            logger.warning("Simulation initialized with zero agents.")
        else:
            logger.info(f"Simulation initialized with {len(self.agents)} agents:")
            for agent in self.agents:
                logger.info(f"  - {agent.get_id()}")
                self.event_kernel.set_budget(agent.get_id(), self.agent_initial_token_budget)
                self.resource_manager.set_du_budget(agent.get_id(), agent.state.du)

            # Initialize collective metrics based on starting agent states
            self._update_collective_metrics()
            logger.info(
                f"Initial collective metrics - IP: {self.collective_ip:.1f}, "
                f"DU: {self.collective_du:.1f}"
            )
            # Prime the event scheduler with the first agent turn
            first_agent = self.agents[0]
            self.event_kernel.schedule_immediate_nowait(
                self._create_agent_event(0),
                agent_id=first_agent.get_id(),
            )

        # --- Message Handling ---
        # Messages generated by agents in the current round, to be perceived in the next.
        # self.pending_messages_for_next_round: list[dict[str, Any]] = [] # Already initialized above
        # Messages available for agents to perceive in the current round.
        # self.messages_to_perceive_this_round: list[dict[str, Any]] = [] # Already initialized above

        # Background task for forwarding external events (e.g., Discord messages)
        self._event_task = None
        self._event_loop = None
        self._event_loop_thread = None

        # Automatically start listening for external events when an event loop
        # is already running. This allows tests to enqueue events before calling
        # :meth:`run_step`.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            self._event_listener_task = loop.create_task(self._event_listener_loop())
            self._event_task = loop.create_task(
                self.event_kernel.forward_external_events(self._handle_human_command)
            )
        else:
            asyncio.set_event_loop(asyncio.new_event_loop())

    # Add method to update collective metrics
    def _update_collective_metrics(self: Self) -> None:
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

        if self.track_collective_metrics:
            current_collective_ip = sum(agent.state.ip for agent in self.agents)
            current_collective_du = sum(agent.state.du for agent in self.agents)
            for agent_instance in self.agents:
                AgentController(agent_instance.state).update_collective_metrics(
                    current_collective_ip, current_collective_du
                )

        # current_round = (self.current_step -1) // len(self.agents) # Not clearly used, commenting out

    async def _handle_human_command(self: Self, text: str) -> None:
        """Handle a human-issued command or prompt."""
        if text.startswith("/kb ") and self.knowledge_board:
            entry = text[4:].strip()
            if entry:
                async with self.knowledge_board.lock:
                    self.knowledge_board.add_entry(
                        entry,
                        "human",
                        self.current_step,
                        self.vector.to_dict(),
                    )
            return

        if not self.agents:
            return

        broadcast = False
        if text.startswith("/broadcast "):
            broadcast = True
            text = text[len("/broadcast ") :].strip()

        agent_id = None
        if self.discord_bot and self.discord_bot.last_agent_id:
            agent_id = self.discord_bot.last_agent_id
        target = next((a for a in self.agents if a.agent_id == agent_id), None)
        if target is None:
            target = self.agents[self.current_agent_index]
        state = target.state
        if broadcast:
            ip_cost = float(
                config.get_config("IP_COST_BROADCAST_MESSAGE")
                or config.get_config("IP_COST_SEND_DIRECT_MESSAGE")
                or 0.0
            )
            du_cost = float(
                config.get_config("DU_COST_BROADCAST_ACTION")
                or config.get_config("DU_COST_PER_ACTION")
                or 0.0
            )
        else:
            ip_cost = float(config.get_config("IP_COST_SEND_DIRECT_MESSAGE") or 0.0)
            du_cost = float(config.get_config("DU_COST_PER_ACTION") or 0.0)

        if state.ip < ip_cost or state.du < du_cost:
            logger.info(
                "Rejecting human %s for %s: insufficient resources",
                "broadcast" if broadcast else "command",
                target.agent_id,
            )
            if self.discord_bot:
                await self.discord_bot.send_simulation_update(
                    "Insufficient IP/DU",
                    agent_id=target.agent_id,
                    target_channel_id=self.discord_bot.last_channel_id,
                )
            return

        state.ip -= ip_cost
        state.du -= du_cost
        try:
            await ledger.spend(
                target.agent_id,
                ip=ip_cost,
                du=du_cost,
                reason="human_broadcast" if broadcast else "human_dm",
            )
        except Exception:  # pragma: no cover - optional
            logger.debug("Ledger spend failed", exc_info=True)

        msgs = []
        if broadcast:
            for ag in self.agents:
                msgs.append(
                    {
                        "step": self.current_step,
                        "sender_id": "human",
                        "recipient_id": ag.agent_id,
                        "content": text,
                        "action_intent": AgentActionIntent.SEND_DIRECT_MESSAGE.value,
                        "sentiment_score": None,
                    }
                )
        else:
            msgs.append(
                {
                    "step": self.current_step,
                    "sender_id": "human",
                    "recipient_id": target.agent_id,
                    "content": text,
                    "action_intent": AgentActionIntent.SEND_DIRECT_MESSAGE.value,
                    "sentiment_score": None,
                }
            )
        async with self._msg_lock:
            self.pending_messages_for_next_round.extend(msgs)
            self.messages_to_perceive_this_round.extend(msgs)

        if self.discord_bot and self.discord_bot.last_channel_id is not None:
            chan = self.discord_bot.last_channel_id
            aid = target.agent_id
            self.discord_bot.channel_map[aid] = chan
            self.discord_bot.channel_to_agent[chan] = aid

    async def handle_control_command(self: Self, cmd: dict[str, Any]) -> None:
        """Process a control command sent via the event queue."""
        action = cmd.get("command")
        if action == "pause":
            self.paused = True
        elif action == "resume":
            self.paused = False
        elif action == "start":
            self.paused = False
        elif action == "stop":
            self.simulation_complete = True
            await self.stop_event_listener()
        elif action == "spawn":
            agent_id = cmd.get("agent_id")
            if agent_id:
                try:
                    from src.agents.core.base_agent import Agent

                    new_agent = Agent(agent_id=str(agent_id), name=str(agent_id))
                    await self.spawn_agent(new_agent)
                except Exception:
                    logger.error("Failed to spawn agent %s", agent_id, exc_info=True)
        elif action == "set_speed":
            try:
                self.speed = float(cmd.get("value", 1))
            except (TypeError, ValueError):
                pass
        elif action == "post_kb":
            text = cmd.get("text")
            author = cmd.get("author", "human")
            if text and self.knowledge_board:
                async with self.knowledge_board.lock:
                    self.knowledge_board.add_entry(
                        text,
                        str(author),
                        self.current_step,
                        self.vector.to_dict(),
                    )
                if self.discord_bot:
                    embed = self.discord_bot.create_knowledge_board_embed(
                        str(author),
                        text,
                        self.current_step,
                    )
                    task = asyncio.create_task(
                        self.discord_bot.send_simulation_update(
                            embed=embed,
                            agent_id=str(author),
                        )
                    )
                    _ = task

    async def mute_agent(self: Self, agent_id: str) -> None:
        from .resources import mute_agent as _mute_agent

        await _mute_agent(self, agent_id)

    async def reset_memory(self: Self, agent_id: str) -> None:
        from .resources import reset_memory as _reset_memory

        await _reset_memory(self, agent_id)

    async def apply_penalty(self: Self, agent_id: str, ip: float, du: float) -> None:
        from .resources import apply_penalty as _apply_penalty

        await _apply_penalty(self, agent_id, ip, du)

    async def handle_moderation_command(self: Self, cmd: dict[str, Any]) -> None:
        """Process moderation actions like muting or penalties."""
        action = cmd.get("command")
        agent_id = cmd.get("agent_id")
        if action == "mute" and agent_id:
            await self.event_kernel.schedule_immediate(
                lambda aid=str(agent_id): self.mute_agent(aid),
                vector=self.vector,
            )
        elif action == "reset_memory" and agent_id:
            await self.event_kernel.schedule_immediate(
                lambda aid=str(agent_id): self.reset_memory(aid),
                vector=self.vector,
            )
        elif action == "penalty" and agent_id:
            ip = float(cmd.get("ip", 0))
            du = float(cmd.get("du", 0))
            await self.event_kernel.schedule_immediate(
                lambda aid=str(agent_id), ip=ip, du=du: self.apply_penalty(aid, ip, du),
                vector=self.vector,
            )

    async def spawn_agent(
        self: Self,
        agent: "Agent",
        *,
        inheritance: float = 0.0,
        parent: "Agent | None" = None,
        mutation_rate: float | None = None,
    ) -> None:
        """Add a new agent to the simulation, inheriting genes with mutation."""
        if mutation_rate is None:
            mutation_rate = float(config.get_config("GENE_MUTATION_RATE") or 0.0)

        agent.state.ip += inheritance

        if parent is not None:
            agent.state.parent_id = parent.agent_id
            genes = parent.state.genes.copy()
            for k, v in genes.items():
                if random.random() < mutation_rate:
                    genes[k] = min(max(v + random.uniform(-0.1, 0.1), 0.0), 1.0)
            genes.update(agent.state.genes)
            agent.state.genes = genes
            try:
                ledger.record_genealogy(parent.agent_id, agent.agent_id)
            except Exception:
                pass
            if self.knowledge_board:
                async with self.knowledge_board.lock:
                    self.knowledge_board.add_entry(
                        f"Agent {parent.agent_id} spawned child {agent.agent_id}",
                        parent.agent_id,
                        self.current_step,
                        self.vector.to_dict(),
                    )
        else:
            if hasattr(agent.state, "mutate_genes"):
                agent.state.mutate_genes(mutation_rate)

        self.agents.append(agent)
        await self.world_map.add_agent(agent.agent_id, x=len(self.agents) - 1, y=0)
        self._update_collective_metrics()

    async def retire_agent(self: Self, agent: "Agent") -> None:
        """Retire an agent and compute inheritance."""
        state = agent.state
        state.is_alive = False
        state.inheritance = state.ip + state.du
        state.ip = 0.0
        state.du = 0.0
        if self.knowledge_board:
            async with self.knowledge_board.lock:
                self.knowledge_board.add_entry(
                    f"Agent {agent.agent_id} retired",
                    agent.agent_id,
                    self.current_step,
                    self.vector.to_dict(),
                )
        agent.update_state(state)

    def get_other_agents_public_state(self: Self, current_agent_id: str) -> list[dict[str, Any]]:
        """
        Returns a list of public state information for all agents other than the current one.

        Args:
            current_agent_id (str): The ID of the agent whose perspective this is from.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, each representing another agent's public state.
        """
        other_agents_info = []
        for agent in self.agents:
            if agent.agent_id != current_agent_id:
                # Ensure agent.state is the Pydantic model AgentState
                if hasattr(agent, "state") and hasattr(agent.state, "model_dump"):
                    public_info = {
                        "agent_id": agent.agent_id,
                        "name": agent.state.name,
                        "role": agent.state.current_role,
                        "mood": agent.state.mood_value,
                        "current_project_id": agent.state.current_project_id,
                        # Add other relevant public fields, avoid sensitive internal state
                    }
                    other_agents_info.append(public_info)
                else:
                    # Fallback or log warning if state structure is not as expected
                    logger.warning(
                        f"Agent {agent.agent_id} state is not a Pydantic model or lacks model_dump, skipping public state."
                    )
        return other_agents_info

    async def send_discord_update(
        self: Self,
        message: Optional[str] = None,
        embed: Optional[object] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """
        Send an update to Discord if the discord_bot is available.

        Args:
            message (Optional[str]): The text message to send to Discord
            embed (Optional[object]): The embed object to send to Discord
        """
        if self.discord_bot:
            # Use asyncio.create_task to avoid blocking the simulation
            task = asyncio.create_task(
                self.discord_bot.send_simulation_update(
                    content=message, embed=embed, agent_id=agent_id
                )
            )
            _ = task

    async def _run_agent_turn(self: Self, agent_index: int) -> None:
        """Execute a single agent turn and schedule the next."""
        if not self.agents:
            logger.warning("No agents in simulation to run.")
            return

        # Increment step and select agent
        self.current_step += 1
        agent = self.agents[agent_index]
        agent_id = agent.agent_id
        if agent_id in self.muted_agents:
            self.current_agent_index = (agent_index + 1) % len(self.agents)
            return
        self.vector.increment(agent_id)
        current_agent_state = agent.state

        if hasattr(current_agent_state, "age"):
            current_agent_state.age += 1
            max_age = int(config.get_config("MAX_AGENT_AGE"))
            if current_agent_state.age >= max_age:
                await self.retire_agent(agent)
                self.current_agent_index = (agent_index + 1) % len(self.agents)
                return

        if not getattr(current_agent_state, "is_alive", True):
            self.current_agent_index = (agent_index + 1) % len(self.agents)
            return

        perception_data: dict[str, Any] = {}
        has_role_change = False
        turn_counter_this_run_step = 0
        next_agent_index = (agent_index + 1) % len(self.agents)

        if agent_index == 0:
            agent_to_run_index = self.current_agent_index
            agent = self.agents[agent_to_run_index]
            agent_id = agent.agent_id
            current_agent_state = agent.state

            if not getattr(current_agent_state, "is_alive", True):
                self.current_agent_index = (agent_to_run_index + 1) % len(self.agents)
                return

            # At the start of a new round (first agent), clear messages_to_perceive_this_round
            # and populate it from what was pending for the next round.
            if agent_to_run_index == 0:
                self.messages_to_perceive_this_round = list(self.pending_messages_for_next_round)
                self.pending_messages_for_next_round = []  # Clear pending for the new round accumulation

                debug_len = len(self.messages_to_perceive_this_round)
                logger.debug(
                    f"Turn {self.current_step} (Agent {agent_id}, Index 0): Initialized messages_to_perceive_this_round "
                    f"with {debug_len} messages from pending_messages_for_next_round."
                )

        ip_start = current_agent_state.ip
        du_start = current_agent_state.du

        # Provide perceived messages and knowledge board content
        async with self._msg_lock:
            perception_data["perceived_messages"] = list(self.messages_to_perceive_this_round)
        if self.knowledge_board:
            perception_data["knowledge_board_content"] = (
                self.knowledge_board.get_recent_entries_for_prompt()
            )

        agent_output = await agent.run_turn(
            simulation_step=self.current_step,
            environment_perception=perception_data,
            memory_service=self.memory_service,
            vector_store_manager=self.vector_store_manager,
            knowledge_board=self.knowledge_board,
        )

        self.resource_manager.cap_tick(
            ip_start=ip_start, du_start=du_start, obj=current_agent_state
        )

        this_agent_turn_generated_messages: list[SimulationMessage] = []

        message_content = agent_output.get("message_content")
        message_recipient_id = agent_output.get("message_recipient_id")
        action_intent_str = agent_output.get("action_intent", "idle")
        map_action = agent_output.get("map_action")

        allowed = await evaluate_policy(action_intent_str)
        if not allowed:
            action_intent_str = AgentActionIntent.IDLE.value
            message_content = None
            message_recipient_id = None
            map_action = None

        if message_content:
            msg_data = cast(
                SimulationMessage,
                {
                    "step": self.current_step,
                    "sender_id": agent_id,
                    "recipient_id": message_recipient_id,
                    "content": message_content,
                    "action_intent": action_intent_str,
                    "sentiment_score": None,
                },
            )
            this_agent_turn_generated_messages.append(msg_data)
            current_agent_state.messages_sent_count += 1
            current_agent_state.last_message_step = self.current_step

            # If the message is a proposal to the Knowledge Board, record it
            if (
                self.knowledge_board
                and action_intent_str == AgentActionIntent.PROPOSE_IDEA.value
                and message_recipient_id is None
            ):
                async with self.knowledge_board.lock:
                    self.knowledge_board.add_entry(
                        message_content,
                        agent_id,
                        self.current_step,
                        self.vector.to_dict(),
                    )

        self.agents[agent_index] = agent
        self.agents[agent_index].update_state(current_agent_state)

        # Add messages generated by this agent to:
        # 1. pending_messages_for_next_round (for the *next* full round of all agents)
        # 2. messages_to_perceive_this_round (so subsequent agents in *this current* round can see them)
        async with self._msg_lock:
            self.pending_messages_for_next_round.extend(this_agent_turn_generated_messages)
            self.messages_to_perceive_this_round.extend(this_agent_turn_generated_messages)

        if hasattr(current_agent_state, "short_term_memory"):
            stm = current_agent_state.short_term_memory
            if stm:
                last_mem = stm[-1]
                if (
                    isinstance(last_mem, dict)
                    and last_mem.get("type") == "role_change"
                    and last_mem.get("step") == self.current_step
                ):
                    has_role_change = True

        if isinstance(map_action, dict):
            from .world_map_actions import process_map_action

            await process_map_action(
                self,
                agent_index,
                agent_id,
                current_agent_state,
                map_action,
            )

        async with self._msg_lock:
            pending_len = len(self.pending_messages_for_next_round)
            perceive_len = len(self.messages_to_perceive_this_round)
        logger.debug(
            f"SIM_DEBUG: After Agent {agent_id}'s turn in Global Turn {self.current_step}: "
            f"pending_messages_for_next_round now has {pending_len} messages. "
            f"messages_to_perceive_this_round now has {perceive_len} messages."
        )
        if has_role_change:
            next_agent_index = agent_index

        self.current_agent_index = next_agent_index
        self.last_completed_agent_index = agent_index

        self._update_collective_metrics()

        logger.info(f"Agent {agent_id} completed Global Turn {self.current_step}:")
        logger.info(f"  - IP: {current_agent_state.ip:.1f} (from {ip_start})")
        logger.info(f"  - DU: {current_agent_state.du:.1f} (from {du_start})")

        event = log_event(
            {
                "type": "agent_action",
                "agent_id": agent_id,
                "step": self.current_step,
                "action_intent": action_intent_str,
                "ip": current_agent_state.ip,
                "du": current_agent_state.du,
            }
        )
        if event is None:
            event = {
                "type": "agent_action",
                "agent_id": agent_id,
                "step": self.current_step,
                "action_intent": action_intent_str,
                "ip": current_agent_state.ip,
                "du": current_agent_state.du,
            }
            event["trace_hash"] = compute_trace_hash(event)
        trace_hash = event["trace_hash"]
        await emit_event(SimulationEvent(type="agent_action", data=event))

        if self.current_step % int(config.SNAPSHOT_INTERVAL_STEPS) == 0:
            snapshot = {
                "step": self.current_step,
                "collective_ip": self.collective_ip,
                "collective_du": self.collective_du,
                "knowledge_board": self.knowledge_board.to_dict(),
                "world_map": self.world_map.to_dict(),
                "agents": [
                    {
                        "agent_id": ag.agent_id,
                        "ip": ag.state.ip,
                        "du": ag.state.du,
                        "mood": ag.state.mood_level,
                    }
                    for ag in self.agents
                ],
                "trace_hash": self._last_trace_hash,
            }
            snapshot_no_vector = {
                **{k: v for k, v in snapshot.items() if k != "trace_hash"},
                "knowledge_board": {
                    k: v for k, v in snapshot["knowledge_board"].items() if k != "vector"
                },
                "world_map": {k: v for k, v in snapshot["world_map"].items() if k != "vector"},
            }
            snapshot["trace_hash"] = compute_trace_hash(snapshot_no_vector)
            self._last_trace_hash = snapshot["trace_hash"]
            save_snapshot(self.current_step, snapshot)
            upload_snapshot(self.current_step)
            snapshot_event = log_event({"type": "snapshot", **snapshot})
            await emit_event(SimulationEvent(type="snapshot", data=snapshot_event))

        # Advance to the next agent for the next turn
        self.current_agent_index = next_agent_index
        self.total_turns_executed += 1
        turn_counter_this_run_step += 1

        if (
            self.memory_service.vector_store
            and config.MEMORY_STORE_PRUNE_INTERVAL_STEPS > 0
            and self.current_step - self._last_memory_prune_step
            >= config.MEMORY_STORE_PRUNE_INTERVAL_STEPS
        ):
            await self.event_kernel.schedule_immediate(
                self._prune_memory_event,
                vector=self.vector,
            )
            self._last_memory_prune_step = self.current_step

        if (
            self.memory_service.vector_store
            and config.MEMORY_STORE_PRUNE_INTERVAL_STEPS > 0
            and self.current_step % len(self.agents) == 0
            and self.current_step > self._last_consolidation_step
        ):
            start_step = self.current_step - len(self.agents) + 1
            await self.event_kernel.schedule_immediate(
                lambda start=start_step: self._consolidate_memory_event(start),
                vector=self.vector,
            )
        self._last_consolidation_step = self.current_step

        semantic_interval = int(
            config.get_config_value_with_override(
                "SEMANTIC_MEMORY_CONSOLIDATION_INTERVAL_STEPS",
                config.SEMANTIC_MEMORY_CONSOLIDATION_INTERVAL_STEPS,
            )
        )
        if (
            self.semantic_manager
            and semantic_interval > 0
            and self.current_step - self._last_semantic_job_step >= semantic_interval
        ):
            for ag in self.agents:
                try:
                    await self.memory_service.run_semantic_job(ag.agent_id)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Failed semantic nightly job: %s", exc)
            self._last_semantic_job_step = self.current_step

        quest_interval = int(
            config.get_config_value_with_override(
                "QUEST_GENERATION_INTERVAL_STEPS",
                config.QUEST_GENERATION_INTERVAL_STEPS,
            )
        )
        if quest_interval > 0 and self.current_step - self._last_quest_step >= quest_interval:
            await self.event_kernel.schedule_immediate(
                self._generate_quest_event,
                vector=self.vector,
            )
            self._last_quest_step = self.current_step

        if (
            self.beats
            and self._beat_interval > 0
            and self._beat_index < len(self.beats)
            and self.current_step >= (self._beat_index + 1) * self._beat_interval
        ):
            metrics: dict[str, Any] = {}
            for hook in self.evaluation_hooks:
                try:
                    metrics.update(hook(self, []) or {})
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Evaluation hook failed")
            metrics["beat"] = self.beats[self._beat_index]
            self.metrics.append({"step": self.current_step, **metrics})
            eval_event = log_event({"type": "evaluation", "step": self.current_step, **metrics})
            if eval_event is None:
                eval_event = {
                    "type": "evaluation",
                    "step": self.current_step,
                    **metrics,
                }
                eval_event["trace_hash"] = compute_trace_hash(eval_event)
            await emit_event(SimulationEvent(type="evaluation", data=eval_event))
            self._beat_index += 1

        self.vector.increment(self.agents[next_agent_index].get_id())
        await self.event_kernel.schedule_in(
            1,
            self._create_agent_event(next_agent_index),
            agent_id=self.agents[next_agent_index].get_id(),
            vector=self.vector,
        )

    def _create_agent_event(self: Self, agent_index: int) -> Callable[[], Awaitable[None]]:
        return lambda: self._run_agent_turn(agent_index)

    async def _prune_memory_event(self: Self) -> None:
        if not self.memory_service.vector_store:
            return
        try:
            self.memory_service.prune_expired(int(config.MEMORY_STORE_TTL_SECONDS))
            # Additionally prune memories based on MUS thresholds
            self.memory_service.prune_mus()
            event = {
                "type": "memory_prune",
                "step": self.current_step,
            }
            await self.event_kernel.emit_environment_event(event)
        except (ChromaDBException, ValidationError, OSError) as exc:
            logger.error("Failed to prune memory store: %s", exc)

    async def _consolidate_memory_event(self: Self, start_step: int) -> None:
        if not self.memory_service.vector_store:
            return
        for ag in self.agents:
            try:
                await self.memory_service.aconsolidate_daily_memories(
                    ag.agent_id,
                    start_step,
                    self.current_step,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed nightly consolidation: %s", exc)
        event = {
            "type": "memory_consolidation",
            "step": self.current_step,
            "start": start_step,
        }
        await self.event_kernel.emit_environment_event(event)

    async def _generate_quest_event(self: Self) -> None:
        quest = await generate_quest("Create a new quest for the agents")
        if quest is None:
            return
        event = {
            "type": "quest_generated",
            "step": self.current_step,
            "quest": quest.model_dump(),
        }
        await self.event_kernel.emit_environment_event(event)

    async def _event_listener_loop(self: Self) -> None:
        """Continuously process events from the shared queue."""
        bus = get_event_bus()
        queue = bus.subscribe()
        try:
            while True:
                evt: SimulationEvent | None = await queue.get()
                if evt is None:
                    break
                await self._handle_incoming_event(evt)
        except asyncio.CancelledError:  # pragma: no cover - task cancelled
            pass
        finally:
            bus.unsubscribe(queue)

    async def _handle_incoming_event(self: Self, evt: SimulationEvent) -> None:
        """Route a ``SimulationEvent`` to agents as a message."""
        if not evt.data:
            return
        if evt.type == "control":
            await self.handle_control_command(evt.data)
            return
        if evt.type == "moderation":
            await self.handle_moderation_command(evt.data)
            return
        sender = str(evt.data.get("author", "external"))
        if sender in self.muted_agents:
            return
        content = str(evt.data.get("content", ""))
        recipient = evt.data.get("recipient_id") if evt.type == "direct_message" else None
        msg: SimulationMessage = {
            "step": self.current_step,
            "sender_id": sender,
            "recipient_id": recipient,
            "content": content,
            "action_intent": None,
            "sentiment_score": None,
        }
        async with self._msg_lock:
            self.pending_messages_for_next_round.append(msg)
            self.messages_to_perceive_this_round.append(msg)

    async def start_event_listener(self: Self) -> None:
        """Start background processing of ``event_queue`` events."""
        if self._event_listener_task is None or self._event_listener_task.done():
            self._event_listener_task = asyncio.create_task(self._event_listener_loop())
        if self._event_task is None or self._event_task.done():
            self._event_task = asyncio.create_task(
                self.event_kernel.forward_external_events(self._handle_human_command)
            )

    async def stop_event_listener(self: Self) -> None:
        """Stop the background event listener task."""
        if self._event_listener_task:
            self._event_listener_task.cancel()
            try:
                await self._event_listener_task
            except asyncio.CancelledError:  # pragma: no cover - expected
                pass
            self._event_listener_task = None
        if self._event_task:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:  # pragma: no cover - expected
                pass
            self._event_task = None

    def add_evaluation_hook(
        self: Self, hook: Callable[[Self, list[Any]], dict[str, Any] | None]
    ) -> None:
        """Register a callback to collect metrics after each step."""
        self.evaluation_hooks.append(hook)

    def _collect_metrics(self: Self, _events: list[Any]) -> dict[str, Any]:
        """Default metrics: coalition count and average sentiment."""
        coalition_count = sum(
            1 for proj in self.projects.values() if len(proj.get("members", [])) > 1
        )
        avg_sentiment = (
            sum(agent.state.mood_level for agent in self.agents) / len(self.agents)
            if self.agents
            else 0.0
        )
        return {"coalitions": coalition_count, "sentiment": avg_sentiment}

    async def run_step(self: Self, max_turns: int = 1) -> int:
        """Dispatch up to ``max_turns`` events via the kernel."""
        if not self.agents:
            logger.warning("No agents in simulation to run.")
            return 0

        await self.start_event_listener()

        if self.event_kernel.empty():
            self.vector.increment(self.agents[self.current_agent_index].get_id())
            self.event_kernel.schedule_immediate_nowait(
                self._create_agent_event(self.current_agent_index),
                agent_id=self.agents[self.current_agent_index].get_id(),
                vector=self.vector,
            )

        events = await self.event_kernel.step(max_turns)

        metrics: dict[str, Any] = {}
        for hook in self.evaluation_hooks:
            try:
                result = hook(self, events) or {}
                metrics.update(result)
            except Exception:
                logger.exception("Evaluation hook failed")
        if metrics:
            self.metrics.append({"step": self.current_step, **metrics})
            eval_event = log_event({"type": "evaluation", "step": self.current_step, **metrics})
            if eval_event is None:
                eval_event = {
                    "type": "evaluation",
                    "step": self.current_step,
                    **metrics,
                }
                eval_event["trace_hash"] = compute_trace_hash(eval_event)
            await emit_event(SimulationEvent(type="evaluation", data=eval_event))

        return len(events)

    async def async_run(self: Self, num_steps: int) -> None:
        """
        Runs the simulation for a specified number of steps asynchronously.

        Args:
            num_steps (int): Number of steps to run.
        """
        logger.info(f"Starting simulation run for {num_steps} steps (async)")
        self.steps_to_run = num_steps
        if self.beats and self._beat_interval == 0:
            self._beat_interval = max(1, num_steps // len(self.beats))
        start_time = time.time()
        total_steps_executed = 0
        try:
            total_steps_executed = await self.run_step(num_steps)
        finally:
            elapsed_time = time.time() - start_time
            logger.info(
                "Simulation completed "
                f"{total_steps_executed} steps in {elapsed_time:.2f} seconds (async)"
            )
            await self.stop_event_listener()
            self.close()

    def apply_event(self: Self, event: dict[str, Any]) -> None:
        """Apply an event from the Redpanda log to the simulation."""
        rng_state = event.get("rng_state")
        if rng_state is not None:
            from src.infra.checkpoint import restore_rng_state

            restore_rng_state(rng_state)
        expected_hash = event.get("trace_hash")
        if expected_hash is not None:
            actual_hash = compute_trace_hash({k: v for k, v in event.items() if k != "trace_hash"})
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Trace hash mismatch for event at step {event.get('step')}:"
                    f" expected {expected_hash}, computed {actual_hash}"
                )
        if event.get("type") == "agent_action":
            aid = event.get("agent_id")
            for agent in self.agents:
                if agent.agent_id == aid:
                    if "ip" in event:
                        agent.state.ip = float(event["ip"])
                    if "du" in event:
                        agent.state.du = float(event["du"])
                    break
            step = event.get("step")
            if isinstance(step, int) and step > self.current_step:
                self.current_step = step
        elif event.get("type") == "environment_change":
            env = event.get("env", {})
            if isinstance(env, dict):
                from src.infra.checkpoint import restore_environment

                restore_environment(env)
        elif event.get("type") == "tick":
            step = event.get("step")
            if isinstance(step, int) and step > self.current_step:
                self.current_step = step
            if "ip" in event:
                self.collective_ip = float(event["ip"])
            if "du" in event:
                self.collective_du = float(event["du"])
        elif event.get("type") == "snapshot":
            step = event.get("step")
            if isinstance(step, int):
                snapshot = load_snapshot(step)
                if snapshot.get("trace_hash") != expected_hash:
                    raise ValueError(
                        f"Snapshot hash mismatch at step {step}:"
                        f" event {expected_hash} != file {snapshot.get('trace_hash')}"
                    )
                self._last_trace_hash = snapshot.get("trace_hash", "")

    @classmethod
    def from_snapshot(cls: type[Self], snapshot: dict[str, Any]) -> Self:
        """Create a ``Simulation`` instance from a snapshot dictionary."""
        agents_data = snapshot.get("agents", [])
        agents = [Agent(agent_id=a.get("agent_id", str(i))) for i, a in enumerate(agents_data)]
        sim = cls(agents=agents, scenario="")
        sim.current_step = int(snapshot.get("step", 0))
        sim.collective_ip = float(snapshot.get("collective_ip", 0.0))
        sim.collective_du = float(snapshot.get("collective_du", 0.0))
        sim._last_trace_hash = snapshot.get("trace_hash", "")

        for a_data in agents_data:
            aid = a_data.get("agent_id")
            for ag in sim.agents:
                if ag.agent_id == aid:
                    ag.state.ip = float(a_data.get("ip", 0.0))
                    ag.state.du = float(a_data.get("du", 0.0))
                    ag.state.mood_level = float(a_data.get("mood", 0.0))
                    break

        kb = snapshot.get("knowledge_board", {})
        for entry in kb.get("entries", []):
            sim.knowledge_board.add_entry(
                entry.get("content_full", ""),
                entry.get("agent_id", "unknown"),
                int(entry.get("step", 0)),
                kb.get("vector"),
            )
        if isinstance(kb.get("vector"), dict):
            sim.knowledge_board.vector.clock.update(kb["vector"])

        wm = snapshot.get("world_map", {})
        if wm:
            sim.world_map.width = int(wm.get("width", sim.world_map.width))
            sim.world_map.height = int(wm.get("height", sim.world_map.height))
            agents_pos = {k: tuple(v) for k, v in wm.get("agents", {}).items()}
            sim.world_map.agent_positions = agents_pos
            if isinstance(wm.get("vector"), dict):
                sim.world_map.vector.clock.update(wm["vector"])
        return sim

    @classmethod
    def replay_from_snapshot(
        cls: type[Self],
        snapshot_path: str | Path,
        *,
        start_step: int | None = None,
        end_step: int | None = None,
    ) -> Self:
        """Load a snapshot and replay events from the event log."""
        snap = load_snapshot(snapshot_path)
        sim = cls.from_snapshot(snap)
        from src.infra import event_log

        after_step = sim.current_step
        if start_step is not None:
            after_step = max(after_step, start_step - 1)

        for event in event_log.stream_events(after_step=after_step):
            step = int(event.get("step", 0))
            if start_step is not None and step < start_step:
                continue
            if end_step is not None and step > end_step:
                break
            sim.apply_event(event)
        return sim

    async def run_turns_concurrent(self: Self, agents: list["Agent"]) -> list[dict[str, Any]]:
        """Run a batch of agent turns concurrently.

        Each agent executes ``run_turn`` simultaneously using :func:`asyncio.gather`.
        This helper is useful for stress testing large simulations where sequential
        execution would be too slow.

        Args:
            agents: The agents whose turns should be executed.

        Returns:
            A list of dictionaries returned by each agent's ``run_turn``.
        """

        start_step = self.current_step + 1
        tasks = [
            agent.run_turn(
                simulation_step=start_step + idx,
                environment_perception={},
                memory_service=self.memory_service,
                vector_store_manager=self.vector_store_manager,
                knowledge_board=self.knowledge_board,
            )
            for idx, agent in enumerate(agents)
        ]

        results = await asyncio.gather(*tasks)

        self.current_step += len(agents)
        self.total_turns_executed += len(agents)
        self.current_agent_index = (self.current_agent_index + len(agents)) % len(self.agents)

        return list(results)

    def close(self: Self) -> None:
        """Release resources held by the simulation."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.stop_event_listener())
        else:
            if loop.is_running():
                task = loop.create_task(self.stop_event_listener())
                task.add_done_callback(lambda t: None)

            else:
                loop.run_until_complete(self.stop_event_listener())
        if hasattr(self.knowledge_board, "close"):
            try:
                close_fn = getattr(self.knowledge_board, "close")
                if callable(close_fn):
                    close_fn()
            except (OSError, RuntimeError) as exc:  # pragma: no cover - defensive
                logger.exception("Failed to close knowledge board: %s", exc)
        if hasattr(self.memory_service, "close"):
            try:
                self.memory_service.close()
            except (OSError, RuntimeError) as exc:  # pragma: no cover - defensive
                logger.exception("Failed to close vector store manager: %s", exc)
        if self._event_loop:
            if self._event_loop.is_running():
                self._event_loop.call_soon_threadsafe(self._event_loop.stop)
            if self._event_loop_thread:
                self._event_loop_thread.join(timeout=0.1)
            self._event_loop = None
            self._event_loop_thread = None
        try:
            get_event_bus().shutdown()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to shutdown event bus")

    def create_project(
        self: Self,
        project_name: str,
        creator_agent_id: str,
        project_description: Optional[str] = None,
    ) -> Optional[str]:
        from .simulation_projects import create_project as _create_project

        return _create_project(self, project_name, creator_agent_id, project_description)

    def join_project(self: Self, project_id: str, agent_id: str) -> bool:
        from .simulation_projects import join_project as _join_project

        return _join_project(self, project_id, agent_id)

    def leave_project(self: Self, project_id: str, agent_id: str) -> bool:
        from .simulation_projects import leave_project as _leave_project

        return _leave_project(self, project_id, agent_id)

    def get_project_details(self: Self) -> dict[str, dict[str, Any]]:
        from .simulation_projects import get_project_details as _get_project_details

        return _get_project_details(self)

    async def propose_law(
        self: Self, proposer_id: str, text: str, vote_weights: dict[str, int] | None = None
    ) -> bool:
        """Allow an agent to propose a law and trigger a weighted vote."""
        from src.governance.service import governance

        proposer = next((a for a in self.agents if a.agent_id == proposer_id), None)
        if proposer is None:
            return False

        if self.knowledge_board:
            async with self.knowledge_board.lock:
                self.knowledge_board.add_law_proposal(
                    text,
                    proposer_id,
                    self.current_step,
                    self.vector.to_dict(),
                )

        result = await governance.propose_law(
            proposer,
            text,
            self.agents,
            vote_weights,
        )
        approved = bool(result.get("approved"))
        if approved and self.knowledge_board:
            async with self.knowledge_board.lock:
                self.knowledge_board.add_entry(
                    f"Law approved: {text}",
                    proposer_id,
                    self.current_step,
                    self.vector.to_dict(),
                )

        return approved

    async def forward_proposal(self: Self, proposer_id: str, text: str) -> bool:
        """Forward a proposal to :func:`propose_law`."""
        return await self.propose_law(proposer_id, text)

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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Culture.ai simulation.")
    parser.add_argument(
        "--steps", type=int, default=5, help="Number of steps to run the simulation for."
    )
    parser.add_argument(
        "--agents", type=int, default=3, help="Number of agents to create for the simulation."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="Collaborative problem-solving session",
        help="Scenario description for the simulation.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for running the simulation.
    """
    args = parse_args()

    # Configure logging
    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.verbosity))

    try:
        get_llm_client()
    except Exception as exc:
        logging.error("SIMULATION: Failed to initialize LLM client: %s", exc)
        return

    # Test DSPy modules
    logging.info("SIMULATION: Attempting to import DSPy role_thought_generator as a test...")
    try:
        logging.info("SIMULATION: Successfully imported DSPy role_thought_generator!")
    except Exception as e:
        logging.error(f"SIMULATION: Failed to import DSPy role_thought_generator: {e}")

    # Test DSPy action intent selector
    logging.info("SIMULATION: Attempting to import DSPy action_intent_selector as a test...")
    try:
        from src.agents.dspy_programs.action_intent_selector import get_optimized_action_selector

        action_selector = get_optimized_action_selector()
        logging.info(
            "SIMULATION: Successfully imported and initialized DSPy action_intent_selector!"
        )

        # Run a quick test
        test_example = {
            "agent_role": "Facilitator",
            "current_situation": "Starting a new simulation.",
            "agent_goal": "Help the group collaborate effectively.",
            "available_actions": [
                "propose_idea",
                "ask_clarification",
                "continue_collaboration",
                "idle",
            ],
        }

        try:
            prediction = action_selector(**test_example)
            logging.info(
                "SIMULATION: Action selector test successful! Selected action: "
                f"{prediction.chosen_action_intent}"
            )
        except Exception as e:
            logging.error(f"SIMULATION: Action selector test call failed: {e}")
    except Exception as e:
        logging.error(f"SIMULATION: Failed to import/initialize DSPy action_intent_selector: {e}")
        import traceback

        logging.error(f"SIMULATION: {traceback.format_exc()}")

    # Create agents for the simulation
    from src.agents.core.base_agent import Agent

    agents = []
    for i in range(args.agents):
        agent = Agent(agent_id=f"agent_{i + 1}")
        agents.append(agent)

    # Create the simulation with the specified agents
    sim = Simulation(agents=agents, scenario=args.scenario)

    # Run the simulation
    import asyncio

    asyncio.run(sim.async_run(num_steps=args.steps))


# Call the main function when the script is run directly
if __name__ == "__main__":
    main()
