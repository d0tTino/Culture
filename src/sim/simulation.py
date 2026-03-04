#!/usr/bin/env python
import argparse
import asyncio
import copy
import logging
import random
import statistics
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
from opentelemetry import trace
from pydantic import ValidationError
from typing_extensions import Self

from src.agents.core.agent_controller import AgentController
from src.agents.core.agent_state import (
    AgentActionIntent,
    AgentLifecycleState,
)
from src.agents.core.personality_engine import ExperienceSignal, PersonalityEngine
from src.agents.memory.memory_service import MemoryService
from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaDBException
from src.governance import evaluate_policy
from src.governance.decision_kernel import PolicyDecisionService
from src.governance.rules_engine import governance_rules_engine
from src.governance.service import governance
from src.infra import config  # Import to access MAX_PROJECT_MEMBERS
from src.infra.event_log import log_event
from src.infra.ledger import ledger
from src.infra.llm_client import get_llm_client
from src.infra.logging_config import setup_logging
from src.infra.snapshot import save_snapshot, upload_snapshot
from src.interfaces.command_bus import CommandBus
from src.interfaces.dashboard_backend import (
    SimulationEvent,
    emit_event,
)
from src.interfaces.discord_event_listener import DiscordSimulationEventListener
from src.interfaces.domain_command_adapters import command_from_payload
from src.interfaces.interaction_commands import InteractionContext, InteractionService
from src.interfaces.metrics import (
    ACTIVE_AGENT_COUNT,
    STEP_PHASE_LATENCY_MS,
    STEP_PHASE_QUEUE_DEPTH,
)
from src.shared.telemetry import trace_agent_action
from src.shared.typing import SimulationMessage
from src.sim.command_service import SimulationCommandService
from src.sim.commands.dispatcher import SimulationCommandDispatcher
from src.sim.contracts.lifecycle import EVENT_STEP_LIFECYCLE_CONTRACTS
from src.sim.control_service import SimulationControlService
from src.sim.engine import SimulationEngine
from src.sim.engines.persistence_engine import PersistenceEngine
from src.sim.environment import EnvironmentState, EnvironmentSystem
from src.sim.event_kernel import EventKernel
from src.sim.graph_knowledge_board import GraphKnowledgeBoard
from src.sim.knowledge_board import KnowledgeBoard
from src.sim.knowledge_board_protocol import (
    KnowledgeBoardProtocol,
    UnsupportedKnowledgeBoardCapabilityError,
    as_semantic_query_store,
)
from src.sim.knowledge_board_service import KnowledgeBoardService
from src.sim.knowledge_entry import KnowledgeEntryType
from src.sim.lifecycle_service import LifecycleService
from src.sim.persistence.snapshot_migrations import (
    CURRENT_SNAPSHOT_SCHEMA_VERSION,
    migrate_snapshot,
)
from src.sim.persistence.snapshot_service import SnapshotPersistenceService
from src.sim.persistence.trace_hash_service import TraceHashService
from src.sim.quests import generate_quest
from src.sim.resource_manager import get_resource_manager
from src.sim.runtime import ExternalEventIngestionService
from src.sim.scheduler_protocol import SchedulerProtocol
from src.sim.version_vector import VersionVector
from src.sim.world_context import WorldContextProjection
from src.sim.world_map import WorldMap

from .event_bus import get_event_bus

tracer = trace.get_tracer(__name__)

# Use TYPE_CHECKING to avoid circular import issues if Agent needs Simulation later
if TYPE_CHECKING:
    from src.agents.core.base_agent import Agent
    from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
    from src.agents.memory.vector_store import ChromaVectorStoreManager
    from src.interfaces.discord_bot import SimulationDiscordBot

# Configure the logger for this module
logger = logging.getLogger(__name__)


# Backward-compatible alias retained for existing callers/tests.
EVENT_STEP_LIFECYCLE_MUST_NOT_CHANGE = EVENT_STEP_LIFECYCLE_CONTRACTS

# Backward-compatible module exports for snapshot monkeypatching in tests.
_ = (save_snapshot, upload_snapshot)


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
        seed: int | None = None,
        evaluation_hook_names: Sequence[str] | None = None,
        evaluation_targets: Mapping[str, Any] | None = None,
        success_metrics: Mapping[str, Any] | None = None,
        scheduler: SchedulerProtocol | None = None,
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
            seed (int | None): Seed for Python and NumPy random number generators.
            evaluation_hook_names (Sequence[str] | None): Symbolic evaluation
                hook names to register for metric collection.
            evaluation_targets (Mapping[str, Any] | None): Scenario-provided
                evaluation target thresholds for compliance checks.
            success_metrics (Mapping[str, Any] | None): High-level success metric
                definitions describing desired narrative outcomes.
        """
        # Reload configuration to pick up any environment overrides set in tests
        config.load_config(validate_required=False)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            logger.info("Random generators seeded with %s", seed)
        self.seed = seed

        self.agents: list[Agent] = agents
        ACTIVE_AGENT_COUNT.set(len(self.agents))
        self.current_step: int = 0
        self.current_agent_index: int = 0
        self.world_ticks_per_day: int = max(1, int(config.get_config("WORLD_TICKS_PER_DAY") or 24))
        self.turns_per_world_tick: int = max(
            1,
            int(config.get_config("WORLD_TICK_TURN_QUANTUM") or len(self.agents) or 1),
        )
        season_len = int(config.get_config("WORLD_SEASON_LENGTH_DAYS") or 0)
        self.world_season_length_days: int | None = season_len if season_len > 0 else None
        self.environment_state = EnvironmentState(
            world_season=0 if self.world_season_length_days else None
        )
        self.environment_system = EnvironmentSystem(
            state=self.environment_state,
            world_ticks_per_day=self.world_ticks_per_day,
            turns_per_world_tick=self.turns_per_world_tick,
            world_season_length_days=self.world_season_length_days,
            weather_shift_interval_ticks=int(config.get_config("WORLD_WEATHER_SHIFT_TICKS") or 6),
            council_window_days=int(config.get_config("WORLD_COUNCIL_WINDOW_DAYS") or 7),
            council_window_start_hour=int(
                config.get_config("WORLD_COUNCIL_WINDOW_START_HOUR") or 9
            ),
            council_window_duration_hours=int(
                config.get_config("WORLD_COUNCIL_WINDOW_DURATION_HOURS") or 3
            ),
            world_time_broadcast_cadence_ticks=int(
                config.get_config("WORLD_TIME_BROADCAST_CADENCE_TICKS") or self.world_ticks_per_day
            ),
        )
        self._last_consolidation_day = -1
        self.last_completed_agent_index: int | None = None
        self.steps_to_run: int = 0  # Number of steps to run, set externally
        self.total_turns_executed = 0
        self.resource_manager = get_resource_manager()
        self.personality_engine = PersonalityEngine()
        self.lifecycle_service = LifecycleService()
        self.simulation_complete = False
        self.event_kernel: SchedulerProtocol = scheduler or EventKernel()
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
        self._last_kb_time = 0.0
        self._last_relay_times: dict[str, float] = {}
        self._kb_cooldown = float(config.get_config("DISCORD_KB_RATE_LIMIT_SECONDS") or 1.0)
        self._relay_cooldown = float(
            config.get_config("DISCORD_MESSAGE_RATE_LIMIT_SECONDS") or 1.0
        )

        # --- Store the simulation scenario ---
        self.scenario = scenario
        if scenario:
            logger.info(f"Simulation initialized with scenario: {scenario}")
        else:
            logger.warning("Simulation initialized without a scenario description.")

        # --- NEW: Initialize Knowledge Board ---
        self.knowledge_board: KnowledgeBoardProtocol
        if config.KNOWLEDGE_BOARD_BACKEND == "graph":
            self.knowledge_board = GraphKnowledgeBoard()
            logger.info("Simulation initialized with Graph Knowledge Board.")
        else:
            self.knowledge_board = KnowledgeBoard()
            logger.info("Simulation initialized with Knowledge Board.")

        governance.attach_knowledge_board(
            self.knowledge_board,
            step_provider=lambda: int(self.current_step),
        )

        self.knowledge_board_service = KnowledgeBoardService(
            self.knowledge_board,
            step_provider=lambda: int(self.current_step),
            vector_provider=lambda: self.vector.to_dict(),
        )

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
        self.evaluation_hook_names: list[str] = []
        self.evaluation_targets: dict[str, Any] = (
            copy.deepcopy(evaluation_targets) if evaluation_targets else {}
        )
        self.success_metrics: dict[str, Any] = (
            copy.deepcopy(success_metrics) if success_metrics else {}
        )
        self.evaluation_hooks: list[Callable[[Self, list[Any]], dict[str, Any] | None]] = []
        self.metrics: list[dict[str, Any]] = []
        self.add_evaluation_hook(self._collect_metrics)
        if evaluation_hook_names:
            self.register_named_evaluation_hooks(list(evaluation_hook_names))

        # --- Initialize memory service ---
        if memory_service is None:
            try:
                memory_service = MemoryService(vector_store_manager, semantic_manager)
            except Exception:  # pragma: no cover - fallback for constrained environments
                class _OfflineTokenizer:
                    def encode(self, text: str) -> list[int]:
                        return [ord(ch) for ch in text]

                memory_service = MemoryService(
                    vector_store_manager,
                    semantic_manager,
                    tokenizer=_OfflineTokenizer(),
                )
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
        self._last_quest_step = 0
        self._last_trace_hash = ""

        # --- Store Discord bot ---
        self.discord_bot = discord_bot
        self._discord_listener: DiscordSimulationEventListener | None = None
        if discord_bot:
            logger.info("Simulation initialized with Discord bot for sending updates.")
            self._discord_listener = DiscordSimulationEventListener(discord_bot)
            self._discord_listener.start()
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
        self.command_service = SimulationCommandService(self)
        self.command_dispatcher = SimulationCommandDispatcher(self.command_service)
        self.interaction_service = InteractionService(self)
        self.command_bus = CommandBus(self.interaction_service)
        self.decision_service = PolicyDecisionService()
        self.control_service = SimulationControlService(self)
        self.engine = SimulationEngine(self)
        self.external_event_ingestion = ExternalEventIngestionService(self)

        # Automatically start listening for external events when an event loop
        # is already running. This allows tests to enqueue events before calling
        # :meth:`run_step`.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            self._event_listener_task = loop.create_task(
                self.external_event_ingestion._event_listener_loop()
            )
            self._event_task = loop.create_task(
                self.event_kernel.forward_external_events(self._handle_human_command_from_bus)
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
        ACTIVE_AGENT_COUNT.set(len(self.agents))

        if self.track_collective_metrics:
            current_collective_ip = sum(agent.state.ip for agent in self.agents)
            current_collective_du = sum(agent.state.du for agent in self.agents)
            for agent_instance in self.agents:
                AgentController(agent_instance.state).update_collective_metrics(
                    current_collective_ip, current_collective_du
                )

        # current_round = (self.current_step -1) // len(self.agents) # Not clearly used, commenting out

    def _assert_trait_update_invariants(
        self,
        state: Any,
        trait_records: Sequence[dict[str, float | str | int]],
        *,
        max_step: float,
    ) -> None:
        trait_values = state.traits.model_dump()
        for trait, value in trait_values.items():
            if not 0.0 <= float(value) <= 1.0:
                raise AssertionError(f"Trait '{trait}' out of bounds: {value}")

        for record in trait_records:
            delta = float(record.get("delta", 0.0))
            if abs(delta) > (max_step + 1e-9):
                raise AssertionError(f"Trait drift exceeded per-step max: {record}")
            if not record.get("cause") or not record.get("source"):
                raise AssertionError(f"Trait audit record missing cause/source: {record}")

    async def _handle_human_command_from_bus(
        self: Self, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Compatibility adapter for external event forwarding callbacks."""

        try:
            await self._handle_human_command(text, metadata)
        except TypeError:
            await self._handle_human_command(text)  # type: ignore[misc]

    async def _handle_human_command(
        self: Self, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Handle a human-issued command or prompt via the unified interaction service."""
        payload = dict(metadata or {})
        raw_channel_id = (
            payload.get("channel_id")
            or payload.get("source_channel_id")
            or payload.get("target_channel_id")
        )
        context = InteractionContext(
            sender_id=str(payload.get("sender_id", "human")),
            channel_id=str(raw_channel_id) if raw_channel_id is not None else None,
            source=str(payload.get("source", "simulation")),
            permissions=(
                set(payload.get("permissions", []))
                if isinstance(payload.get("permissions"), list | set | tuple)
                else set()
            ),
            metadata={k: v for k, v in payload.items() if k not in {"permissions"}},
        )
        command_type = payload.get("command_type")
        if not command_type and bool(payload.get("broadcast")):
            command_type = "broadcast"
        command = command_from_payload(
            {"command_type": command_type or "human_message", "content": text, **payload},
            context=context,
        )
        result = await self.command_dispatcher.dispatch(command, context=context)
        if result.status != "ok" and self.discord_bot:
            channel_id = context.channel_id
            target_channel_id = (
                int(channel_id)
                if channel_id is not None and channel_id.isdigit()
                else self.discord_bot.last_channel_id
            )
            await self.discord_bot.send_simulation_update(
                result.user_message,
                agent_id=context.sender_id,
                target_channel_id=target_channel_id,
            )

    async def handle_control_command(self: Self, cmd: Mapping[str, Any]) -> dict[str, Any] | None:
        """Process a control command sent via the event queue."""
        return await self.control_service.handle_control_command(cmd)

    async def mute_agent(self: Self, agent_id: str, *, emit_event: bool = True) -> None:
        from .resources import mute_agent as _mute_agent

        event = _mute_agent(self, agent_id)
        if emit_event:
            await self.event_kernel.emit_environment_event(event)

    async def unmute_agent(self: Self, agent_id: str, *, emit_event: bool = True) -> None:
        from .resources import unmute_agent as _unmute_agent

        event = _unmute_agent(self, agent_id)
        if emit_event:
            await self.event_kernel.emit_environment_event(event)

    async def reset_memory(self: Self, agent_id: str, *, emit_event: bool = True) -> None:
        from .resources import reset_memory as _reset_memory

        event = _reset_memory(self, agent_id)
        if emit_event and event:
            await self.event_kernel.emit_environment_event(event)

    async def apply_penalty(
        self: Self, agent_id: str, ip: float, du: float, *, emit_event: bool = True
    ) -> None:
        from .resources import apply_penalty as _apply_penalty

        event = _apply_penalty(self, agent_id, ip, du)
        if emit_event and event:
            await self.event_kernel.emit_environment_event(event)

    async def handle_moderation_command(self: Self, cmd: dict[str, Any]) -> None:
        """Process moderation actions like muting or penalties."""
        from src.interfaces.interaction_schema import (
            InteractionContext,
            parse_interaction_envelope,
        )

        envelope = parse_interaction_envelope(
            {
                "intent": "moderation",
                "action": cmd.get("command"),
                "agent_id": cmd.get("agent_id"),
                "metadata": cmd,
            }
        )
        decision = self.decision_service.decide(
            envelope=envelope,
            context=InteractionContext(
                sender_id="simulation",
                source="simulation",
                permissions={"admin", "moderator"},
            ),
            simulation=self,
            stage="moderation",
        )
        if decision.decision != "allow":
            return

        action = cmd.get("command")
        agent_id = cmd.get("agent_id")
        if action == "mute" and agent_id:
            await self.event_kernel.schedule_immediate(
                lambda aid=str(agent_id): self.mute_agent(aid),
                vector=self.vector,
            )
        elif action == "unmute" and agent_id:
            await self.event_kernel.schedule_immediate(
                lambda aid=str(agent_id): self.unmute_agent(aid),
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
        predecessor: "Agent | None" = None,
        inherit_role: bool = True,
        inherit_context: bool = False,
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
                await self.knowledge_board_service.post_event(
                    actor_id=parent.agent_id,
                    content=f"Agent {parent.agent_id} spawned child {agent.agent_id}",
                    event_type=KnowledgeEntryType.SPAWN_EVENT,
                    tags=["population", "spawn"],
                    reference_metadata={"child_id": agent.agent_id},
                    causal_source="simulation.spawn_agent.parent",
                )
        else:
            if hasattr(agent.state, "mutate_genes"):
                agent.state.mutate_genes(mutation_rate)

        if predecessor is not None:
            succession = self.lifecycle_service.register_successor(
                predecessor=predecessor,
                successor=agent,
                inherit_role=inherit_role,
                inherit_context=inherit_context,
            )
            if self.knowledge_board:
                await self.knowledge_board_service.post_event(
                    actor_id=predecessor.agent_id,
                    content=(
                        f"Agent {agent.agent_id} designated successor of {predecessor.agent_id}"
                    ),
                    event_type=KnowledgeEntryType.SPAWN_EVENT,
                    tags=["population", "succession"],
                    reference_metadata=succession,
                    causal_source="simulation.spawn_agent.successor",
                )

        self.agents.append(agent)
        await self.world_map.add_agent(agent.agent_id, x=len(self.agents) - 1, y=0)
        self._update_collective_metrics()
        ACTIVE_AGENT_COUNT.set(len(self.agents))

    async def retire_agent(
        self: Self,
        agent: "Agent",
        *,
        remove_from_simulation: bool = False,
        lifecycle_state: AgentLifecycleState = AgentLifecycleState.RETIRED,
        reason: str = "",
    ) -> None:
        """Retire an agent, compute inheritance, and optionally remove from simulation."""
        transition = self.lifecycle_service.transition(
            agent=agent,
            to_state=lifecycle_state,
            step=self.current_step,
            reason=reason,
            projects=self.projects,
        )
        lifecycle_event = log_event(
            {
                "type": "agent_lifecycle_transition",
                "step": self.current_step,
                "agent_id": agent.agent_id,
                "from_state": transition.from_state.value,
                "to_state": transition.to_state.value,
                "reason": reason,
                "legacy_artifacts": transition.artifacts,
                "memory_archival_policy": transition.archival_policy,
            }
        )
        if lifecycle_event is not None:
            await emit_event(
                SimulationEvent(type="agent_lifecycle_transition", data=lifecycle_event)
            )

        if self.knowledge_board:
            await self.knowledge_board_service.post_lifecycle_transition(
                actor_id=agent.agent_id,
                from_state=transition.from_state.value,
                to_state=transition.to_state.value,
                reason=reason,
                legacy_artifacts=transition.artifacts,
                causal_source="lifecycle_service.transition",
            )
        agent.update_state(agent.state)
        await self.world_map.remove_agent(agent.agent_id)
        if remove_from_simulation:
            try:
                self.agents.remove(agent)
            except ValueError:  # pragma: no cover - defensive
                pass
        self._update_collective_metrics()

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

    @property
    def world_tick_index(self: Self) -> int:
        return self.environment_state.world_tick

    @world_tick_index.setter
    def world_tick_index(self: Self, value: int) -> None:
        self.environment_state.world_tick = int(value)

    @property
    def world_hour(self: Self) -> int:
        return self.environment_state.world_hour

    @world_hour.setter
    def world_hour(self: Self, value: int) -> None:
        self.environment_state.world_hour = int(value)

    @property
    def world_day(self: Self) -> int:
        return self.environment_state.world_day

    @world_day.setter
    def world_day(self: Self, value: int) -> None:
        self.environment_state.world_day = int(value)

    @property
    def world_season(self: Self) -> int | None:
        return self.environment_state.world_season

    @world_season.setter
    def world_season(self: Self, value: int | None) -> None:
        self.environment_state.world_season = int(value) if value is not None else None

    def _build_world_context_projection(self: Self, *, actor_id: str) -> WorldContextProjection:
        return WorldContextProjection.build(
            turn_index=self.current_step,
            actor_id=actor_id,
            environment_system=self.environment_system,
            world_map=self.world_map,
        )

    @staticmethod
    def _environment_context_from_projection(
        projection: WorldContextProjection,
        *,
        effect_hooks: dict[str, Any],
    ) -> dict[str, Any]:
        return projection.to_environment_context(effect_hooks=effect_hooks)

    @staticmethod
    def _world_time_from_projection(projection: WorldContextProjection) -> dict[str, Any]:
        return {
            "world_tick": projection.time.world_tick,
            "world_hour": projection.time.world_hour,
            "world_day": projection.time.world_day,
            "world_season": projection.time.world_season,
            "formatted": projection.time.formatted,
        }

    async def _emit_environment_observability_events(
        self: Self,
        events: Sequence[Mapping[str, Any]],
        *,
        step: int,
        world_projection: WorldContextProjection,
    ) -> None:
        environment_context = self._environment_context_from_projection(
            world_projection,
            effect_hooks=self.environment_system._condition_hooks(),
        )
        world_time = self._world_time_from_projection(world_projection)
        for raw_event in events:
            payload = {
                **dict(raw_event),
                "step": step,
                "environment_context": environment_context,
                "world_time": world_time,
                "world_context_projection": world_projection.to_dict(),
            }
            event = log_event(payload)
            if event is None:
                event = {**payload}
                event["trace_hash"] = TraceHashService.compute(payload)
            event_name = str(payload.get("event_name", "environment"))
            await emit_event(SimulationEvent(type="environment", data=event))
            if event_name == "world_time":
                await self.send_discord_update(message=f"🕒 {world_time['formatted']}")

    async def send_discord_update(
        self: Self,
        message: str | None = None,
        embed: object | None = None,
        agent_id: str | None = None,
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

    def _record_experience_signals(
        self: Self,
        *,
        action_intent: str,
        requested_action_intent: str,
        message_recipient_id: str | None,
        mood_before: float,
        mood_after: float,
        rule_allowed: bool,
    ) -> ExperienceSignal:
        social_outcome = 0.15 if action_intent != AgentActionIntent.IDLE.value else -0.05
        if message_recipient_id is not None:
            social_outcome += 0.2

        conflict_outcome = 0.0
        if not rule_allowed:
            conflict_outcome -= 0.4
        if requested_action_intent == AgentActionIntent.REQUEST_ROLE_CHANGE.value:
            conflict_outcome += 0.1

        task_outcome = (
            0.2
            if action_intent
            in {
                AgentActionIntent.PERFORM_DEEP_ANALYSIS.value,
                AgentActionIntent.PROPOSE_IDEA.value,
                AgentActionIntent.BUILD.value,
                AgentActionIntent.GATHER.value,
            }
            else 0.0
        )

        governance_participation = (
            0.1 if requested_action_intent != AgentActionIntent.IDLE.value else 0.0
        )

        return ExperienceSignal(
            social_outcome=social_outcome,
            conflict_outcome=conflict_outcome,
            task_outcome=task_outcome,
            mood_trajectory=float(mood_after - mood_before),
            governance_participation=governance_participation,
        )

    async def _run_agent_turn(self: Self, agent_index: int) -> None:
        """Execute a single agent turn and schedule the next."""
        if not self.agents:
            logger.warning("No agents in simulation to run.")
            return

        # Increment turn index, then sync world time to the configured tick boundary.
        self.current_step += 1
        environment_events = self.environment_system.tick(self.current_step)
        agent = self.agents[agent_index]
        agent_id = agent.agent_id
        world_projection = self._build_world_context_projection(actor_id=agent_id)
        await self._emit_environment_observability_events(
            environment_events, step=self.current_step, world_projection=world_projection
        )
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

        # Per-turn contract: perceive -> decide -> act -> record signals -> update personality
        # perceive
        async with self._msg_lock:
            perception_data["perceived_messages"] = list(self.messages_to_perceive_this_round)
        if self.knowledge_board:
            perception_data["knowledge_board_content"] = (
                self.knowledge_board.get_recent_entries_for_prompt()
            )
        effect_hooks = self.environment_system._condition_hooks()
        environment_context = self._environment_context_from_projection(
            world_projection,
            effect_hooks=effect_hooks,
        )
        world_time = self._world_time_from_projection(world_projection)
        perception_data["environment_context"] = environment_context
        mood_before = float(current_agent_state.mood_level)

        # decide
        with trace_agent_action("agent_activation", agent_id=agent_id, step=self.current_step):
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
        requested_action_intent = action_intent_str
        map_action = agent_output.get("map_action")

        explain_why_payload = (
            {**agent_output.get("explain_why", {})}
            if isinstance(agent_output.get("explain_why"), dict)
            else {}
        )
        kb_excerpt = perception_data.get("knowledge_board_content", [])
        if not isinstance(kb_excerpt, list):
            kb_excerpt = [kb_excerpt] if kb_excerpt else []
        explain_why_payload.setdefault(
            "knowledge_board_entries", [str(entry) for entry in kb_excerpt]
        )
        explain_why_payload.setdefault("memories", [])
        explain_why_payload.setdefault("tool_calls", [])
        if "rag_summary" not in explain_why_payload:
            explain_why_payload["rag_summary"] = None

        # act
        policy_allowed = await evaluate_policy(action_intent_str)
        governance_outcome = governance_rules_engine.pre_action_check(
            action_intent_str,
            policy_allowed=policy_allowed,
        )

        if not governance_outcome.allowed:
            action_intent_str = AgentActionIntent.IDLE.value
            message_content = None
            message_recipient_id = None
            map_action = None

        if governance_outcome.penalties:
            for penalty in governance_outcome.penalties:
                current_agent_state.ip = max(
                    0.0, float(current_agent_state.ip) - float(penalty.get("ip", 0.0))
                )
                current_agent_state.du = max(
                    0.0, float(current_agent_state.du) - float(penalty.get("du", 0.0))
                )

        governance_audit = governance_rules_engine.post_action_enforcement(
            governance_outcome,
            agent_id=agent_id,
            step=self.current_step,
        )

        if self.knowledge_board:
            await self.knowledge_board_service.post_event(
                actor_id=agent_id,
                content=(
                    f"Governance enforcement {governance_outcome.outcome} for action "
                    f"'{requested_action_intent}' by {agent_id}"
                ),
                event_type=KnowledgeEntryType.GOVERNANCE_DECISION,
                tags=["governance", governance_outcome.outcome, governance_outcome.decision],
                governance_rule_id=(
                    governance_outcome.rule_ids[0] if governance_outcome.rule_ids else None
                ),
                reference_metadata={
                    "decision": governance_outcome.decision,
                    "outcome": governance_outcome.outcome,
                    "rule_ids": governance_outcome.rule_ids,
                    "policy_allowed": policy_allowed,
                    "rule_reason": governance_outcome.reason,
                    "penalties": governance_outcome.penalties,
                    "audit": governance_audit,
                },
                causal_source="governance_rules_engine.post_action_enforcement",
            )
        if message_content:
            msg_data = cast(
                SimulationMessage,
                {
                    "step": self.current_step,
                    "turn_index": self.current_step,
                    "environment_context": environment_context,
                    "world_time": world_time,
                    "world_context_projection": world_projection.to_dict(),
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
                await self.knowledge_board_service.post_idea(
                    actor_id=agent_id,
                    content=message_content,
                    tags=["agent"],
                    causal_source="agent_action.propose_idea",
                )

        self.agents[agent_index] = agent

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

        # record signals -> update personality
        experience_signals = self._record_experience_signals(
            action_intent=action_intent_str,
            requested_action_intent=requested_action_intent,
            message_recipient_id=message_recipient_id,
            mood_before=mood_before,
            mood_after=float(current_agent_state.mood_level),
            rule_allowed=bool(governance_outcome.allowed),
        )
        trait_records = self.personality_engine.apply_experience_drift(
            current_agent_state,
            experience_signals,
            source="simulation.turn",
        )
        self._assert_trait_update_invariants(
            current_agent_state,
            trait_records,
            max_step=0.01,
        )
        self.agents[agent_index].update_state(current_agent_state)

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

        with tracer.start_as_current_span("simulation.agent_action") as span:
            span.set_attribute("agent.id", agent_id)
            span.set_attribute("simulation.step", self.current_step)
            span.set_attribute("llm.tokens.prompt", 0)
            span.set_attribute("llm.tokens.completion", 0)
            span.set_attribute("llm.tokens.total", 0)
            start = time.perf_counter()
            try:
                kb_state = {
                    k: v for k, v in self.knowledge_board.to_snapshot().items() if k != "vector"
                }
                wm_state = {k: v for k, v in self.world_map.to_dict().items() if k != "vector"}
                payload = {
                    "type": "agent_action",
                    "agent_id": agent_id,
                    "step": self.current_step,
                    "turn_index": self.current_step,
                    "environment_context": environment_context,
                    "world_time": world_time,
                    "world_context_projection": world_projection.to_dict(),
                    "action_intent": action_intent_str,
                    "governance_enforcement": {
                        "decision": governance_outcome.decision,
                        "outcome": governance_outcome.outcome,
                        "reason": governance_outcome.reason,
                        "violated_rules": governance_outcome.rule_ids,
                        "penalties": governance_outcome.penalties,
                    },
                    "ip": current_agent_state.ip,
                    "du": current_agent_state.du,
                    "knowledge_board": kb_state,
                    "world_map": wm_state,
                    "explain_why": explain_why_payload,
                }
                event = log_event(payload)
                if event is None:
                    event = {**payload}
                    event["trace_hash"] = TraceHashService.compute(payload)
                trace_hash = event["trace_hash"]
                await emit_event(SimulationEvent(type="agent_action", data=event))
            finally:
                span.set_attribute("simulation.latency_ms", (time.perf_counter() - start) * 1000)

        if self.current_step % int(config.SNAPSHOT_INTERVAL_STEPS) == 0:
            from src.infra.checkpoint import capture_rng_state

            snapshot = {
                "snapshot_schema_version": CURRENT_SNAPSHOT_SCHEMA_VERSION,
                "step": self.current_step,
                "collective_ip": self.collective_ip,
                "collective_du": self.collective_du,
                "knowledge_board": self.knowledge_board.to_snapshot(),
                "world_map": self.world_map.to_dict(),
                "agents": [
                    {
                        "agent_id": ag.agent_id,
                        "ip": ag.state.ip,
                        "du": ag.state.du,
                        "mood": ag.state.mood_level,
                        "lifecycle_state": getattr(
                            ag.state, "lifecycle_state", AgentLifecycleState.ACTIVE
                        ).value,
                        "lifecycle_history": list(getattr(ag.state, "lifecycle_history", [])),
                        "legacy_artifacts": dict(getattr(ag.state, "legacy_artifacts", {})),
                        "memory_archival_policy": dict(
                            getattr(ag.state, "memory_archival_policy", {})
                        ),
                        "predecessor_id": getattr(ag.state, "predecessor_id", None),
                        "successor_id": getattr(ag.state, "successor_id", None),
                    }
                    for ag in self.agents
                ],
                "seed": self.seed,
                "rng_state": capture_rng_state(),
                "world_hour": self.world_hour,
                "world_day": self.world_day,
                "world_season": self.world_season,
                "world_tick": self.world_tick_index,
                "turns_per_world_tick": self.turns_per_world_tick,
                "environment_state": {
                    "world_tick": self.environment_state.world_tick,
                    "world_hour": self.environment_state.world_hour,
                    "world_day": self.environment_state.world_day,
                    "world_season": self.environment_state.world_season,
                    "weather": self.environment_state.weather,
                    "season_effects": self.environment_state.season_effects,
                    "active_global_modifiers": self.environment_state.active_global_modifiers,
                    "council_window_active": self.environment_state.council_window_active,
                },
                "metadata": {"world_context_projection": world_projection.to_dict()},
                "trace_hash": self._last_trace_hash,
            }
            snapshot["trace_hash"] = SnapshotPersistenceService.compute_hash(snapshot)
            self._last_trace_hash = snapshot["trace_hash"]
            SnapshotPersistenceService.save(self.current_step, snapshot)
            SnapshotPersistenceService.upload(self.current_step)
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
            and self.world_day > self._last_consolidation_day
        ):
            turns_per_world_day = self.turns_per_world_tick * self.world_ticks_per_day
            start_step = max(1, self.current_step - turns_per_world_day + 1)
            await self.event_kernel.schedule_immediate(
                lambda start=start_step: self._consolidate_memory_event(start),
                vector=self.vector,
            )
            self._last_consolidation_day = self.world_day

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
                    with tracer.start_as_current_span("simulation.evaluation_hook") as span:
                        span.set_attribute("hook.name", getattr(hook, "__name__", repr(hook)))
                        span.set_attribute("simulation.step", self.current_step)
                        result = hook(self, []) or {}
                        for key, value in result.items():
                            span.set_attribute(f"metric.{key}", value)
                        metrics.update(result)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Evaluation hook failed")
            metrics["beat"] = self.beats[self._beat_index]
            metrics_entry: dict[str, Any] = {"step": self.current_step, **metrics}
            self.metrics.append(metrics_entry)

            target_summary = self._summarize_evaluation_targets()
            target_status: str | None = None
            target_alerts: list[str] = []
            if target_summary:
                target_status = self._aggregate_target_status(target_summary)
                target_alerts = self._collect_target_alerts(target_summary)
                metrics_entry["_target_summary"] = copy.deepcopy(target_summary)
                if target_status is not None:
                    metrics_entry["_target_status"] = target_status
                if target_alerts:
                    metrics_entry["_target_alerts"] = target_alerts

            event_payload: dict[str, Any] = {
                "type": "evaluation",
                "step": self.current_step,
                **metrics,
            }
            if target_summary:
                event_payload["_target_summary"] = target_summary
                if target_status is not None:
                    event_payload["_target_status"] = target_status
                if target_alerts:
                    event_payload["_target_alerts"] = target_alerts

            eval_event = log_event(event_payload)
            if eval_event is None:
                eval_event = {
                    "type": "evaluation",
                    "step": self.current_step,
                    **metrics,
                }
                if target_summary:
                    eval_event["_target_summary"] = target_summary
                    if target_status is not None:
                        eval_event["_target_status"] = target_status
                    if target_alerts:
                        eval_event["_target_alerts"] = target_alerts
                eval_event["trace_hash"] = TraceHashService.compute(eval_event)
            elif target_summary:
                # Ensure downstream consumers receive the summary even if the
                # event log implementation returned a cached payload.
                eval_event.setdefault("_target_summary", target_summary)
                if target_status is not None:
                    eval_event.setdefault("_target_status", target_status)
                if target_alerts:
                    eval_event.setdefault("_target_alerts", target_alerts)

            if target_status == "fail":
                logger.warning(
                    "Evaluation targets violated at step %s: %s",
                    self.current_step,
                    target_alerts or target_summary,
                )

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
        await self.external_event_ingestion._event_listener_loop()

    async def _handle_incoming_event(self: Self, evt: SimulationEvent) -> None:
        """Route a ``SimulationEvent`` to agents as a message."""
        await self.external_event_ingestion.route_event(evt)

    async def start_event_listener(self: Self) -> None:
        """Start background processing of ``event_queue`` events."""
        await self.external_event_ingestion.start()

    async def stop_event_listener(self: Self) -> None:
        """Stop the background event listener task."""
        await self.external_event_ingestion.stop()

    def add_evaluation_hook(
        self: Self, hook: Callable[[Self, list[Any]], dict[str, Any] | None]
    ) -> None:
        """Register a callback to collect metrics after each step."""
        self.evaluation_hooks.append(hook)

    def _collect_metrics(self: Self, _events: list[Any]) -> dict[str, Any]:
        """Default metrics: coalition count, sentiment, and collective resources."""
        coalition_count = sum(
            1 for proj in self.projects.values() if len(proj.get("members", [])) > 1
        )
        avg_sentiment = (
            sum(agent.state.mood_level for agent in self.agents) / len(self.agents)
            if self.agents
            else 0.0
        )
        return {
            "coalitions": coalition_count,
            "sentiment": avg_sentiment,
            "collective_ip": self.collective_ip,
            "collective_du": self.collective_du,
        }

    def _summarize_evaluation_targets(self: Self) -> dict[str, dict[str, object]]:
        """Aggregate collected metrics and compare them to ``evaluation_targets``."""

        summary: dict[str, dict[str, object]] = {}
        if not self.evaluation_targets:
            return summary

        history: dict[str, list[float]] = {}
        for record in self.metrics:
            for key, value in record.items():
                if not isinstance(key, str):
                    continue
                if key.startswith("_") or key in {"step", "beat"}:
                    continue
                if isinstance(value, (int, float)):
                    history.setdefault(key, []).append(float(value))

        for metric, target in self.evaluation_targets.items():
            if not isinstance(metric, str) or not isinstance(target, Mapping):
                continue

            values = history.get(metric, [])
            if not values:
                summary[metric] = {
                    "status": "missing",
                    "reason": "No samples recorded for this metric.",
                }
                continue

            checks: dict[str, dict[str, object]] = {}
            status: str = "pass"
            for field, raw_threshold in target.items():
                field_name = str(field)
                if not isinstance(raw_threshold, (int, float)):
                    checks[field_name] = {
                        "threshold": raw_threshold,
                        "passed": None,
                        "reason": "Non-numeric threshold is not evaluated.",
                    }
                    if status == "pass":
                        status = "unknown"
                    continue

                threshold = float(raw_threshold)
                observed: float
                passed: bool
                reason: str | None = None

                if field_name in {"max_count", "max_value"}:
                    observed = max(values)
                    passed = observed <= threshold
                    if not passed:
                        reason = f"observed {observed:.3f} exceeds max {threshold:.3f}"
                elif field_name == "min_value":
                    observed = min(values)
                    passed = observed >= threshold
                    if not passed:
                        reason = f"observed {observed:.3f} below min {threshold:.3f}"
                elif field_name == "max_delta":
                    deltas = [abs(curr - prev) for prev, curr in pairwise(values)]
                    observed = max(deltas) if deltas else 0.0
                    passed = observed <= threshold
                    if not passed:
                        reason = f"largest delta {observed:.3f} exceeds max {threshold:.3f}"
                elif field_name == "max_variance":
                    observed = statistics.pvariance(values) if len(values) > 1 else 0.0
                    passed = observed <= threshold
                    if not passed:
                        reason = f"variance {observed:.3f} exceeds max {threshold:.3f}"
                else:
                    checks[field_name] = {
                        "threshold": threshold,
                        "passed": None,
                        "reason": "Unsupported target field.",
                    }
                    if status == "pass":
                        status = "unknown"
                    continue

                checks[field_name] = {
                    "threshold": threshold,
                    "observed": observed,
                    "passed": passed,
                }
                if reason is not None:
                    checks[field_name]["reason"] = reason
                if not passed:
                    status = "fail"

            if not checks:
                summary[metric] = {
                    "status": "unknown",
                    "reason": "No recognized target fields for evaluation.",
                }
                continue

            summary[metric] = {"status": status, "checks": checks}

        return summary

    @staticmethod
    def _aggregate_target_status(summary: Mapping[str, Mapping[str, object]]) -> str:
        """Collapse per-metric statuses into a single aggregate label."""

        priority = {"fail": 0, "missing": 1, "unknown": 2, "pass": 3}
        overall = "pass"
        best_score = priority[overall]
        for details in summary.values():
            status = str(details.get("status", "unknown")).lower()
            score = priority.get(status, priority["unknown"])
            if score < best_score:
                best_score = score
                overall = status
            if score == 0:
                break
        return overall

    @staticmethod
    def _collect_target_alerts(summary: Mapping[str, Mapping[str, object]]) -> list[str]:
        """Generate alert messages for any metric that did not pass."""

        alerts: list[str] = []
        for metric, details in summary.items():
            status = str(details.get("status", "")).lower()
            if status in {"", "pass"}:
                continue

            note: str | None = None
            checks = details.get("checks")
            if isinstance(checks, Mapping):
                fragments: list[str] = []
                for field, check in checks.items():
                    if not isinstance(check, Mapping):
                        continue
                    passed = check.get("passed")
                    if passed is False or status in {"missing", "unknown"}:
                        reason = check.get("reason")
                        if isinstance(reason, str) and reason:
                            fragments.append(f"{field}: {reason}")
                        else:
                            observed = check.get("observed")
                            threshold = check.get("threshold")
                            parts: list[str] = []
                            if isinstance(observed, (int, float)):
                                parts.append(f"observed={observed:.3f}")
                            if isinstance(threshold, (int, float)):
                                parts.append(f"target={threshold:.3f}")
                            if parts:
                                fragments.append(f"{field}: {', '.join(parts)}")
                if fragments:
                    note = "; ".join(fragments)

            if note is None:
                reason = details.get("reason")
                if isinstance(reason, str) and reason:
                    note = reason

            message = f"{metric} {status}"
            if note:
                message = f"{message} ({note})"
            alerts.append(message)

        return alerts

    def register_named_evaluation_hooks(self: Self, hook_names: Sequence[str]) -> None:
        """Register evaluation hooks by symbolic ``hook_names``.

        Each name is looked up in :data:`EVALUATION_HOOKS`; unknown names are ignored
        with a warning. The default ``_collect_metrics`` hook is removed before
        registering the provided hooks to avoid duplicate metric entries.
        """

        requested = [str(name) for name in hook_names]
        self.evaluation_hooks = []
        self.evaluation_hook_names = []
        for name in requested:
            hook = EVALUATION_HOOKS.get(name)
            if hook is None:
                logger.warning("Unknown evaluation hook '%s'", name)
                continue
            self.evaluation_hook_names.append(name)
            self.add_evaluation_hook(hook)
        if not self.evaluation_hook_names:
            if requested:
                logger.warning(
                    "No valid evaluation hooks configured; using default metrics collector"
                )
            self.add_evaluation_hook(self._collect_metrics)

    @staticmethod
    def _set_labeled_gauge(gauge: Any, *, phase: str, value: float | int) -> None:
        """Set a labeled gauge when labels are supported; fallback to plain set."""

        if hasattr(gauge, "labels"):
            gauge.labels(phase=phase).set(value)
            return
        gauge.set(value)

    async def run_step(self: Self, max_turns: int = 1) -> int:
        """Dispatch up to ``max_turns`` events via the deterministic simulation engine."""
        return await self.engine.run_step(max_turns=max_turns)

    def _build_step_perception_snapshot(
        self: Self, turn_index: int, *, actor_id: str | None = None
    ) -> Mapping[str, Any]:
        """Build an immutable perception snapshot for a pipeline tick."""

        projection = self._build_world_context_projection(
            actor_id=actor_id or self.agents[self.current_agent_index].agent_id
        )
        environment_context = self._environment_context_from_projection(
            projection,
            effect_hooks=self.environment_system._condition_hooks(),
        )
        snapshot: dict[str, Any] = {
            "perceived_messages": copy.deepcopy(self.messages_to_perceive_this_round),
            "knowledge_board_content": (
                copy.deepcopy(self.knowledge_board.get_recent_entries_for_prompt())
                if self.knowledge_board
                else []
            ),
            "turn_index": turn_index,
            "environment_context": copy.deepcopy(environment_context),
            "world_context_projection": projection.to_dict(),
        }
        return MappingProxyType(snapshot)

    def _build_tick_read_snapshot(self: Self, *, turn_index: int) -> Mapping[str, Any]:
        """Capture an immutable read-only snapshot for all decision workers in a tick."""

        snapshot = {
            "turn_index": turn_index,
            "messages": copy.deepcopy(self.messages_to_perceive_this_round),
            "knowledge_board": (
                copy.deepcopy(self.knowledge_board.get_recent_entries_for_prompt())
                if self.knowledge_board
                else []
            ),
            "world_map": copy.deepcopy(self.world_map.to_dict()),
            "governance": {
                "current_rules": copy.deepcopy(governance_rules_engine.current_rules()),
                "active_offices": copy.deepcopy(governance_rules_engine.active_offices()),
            },
            "vector": copy.deepcopy(self.vector.to_dict()),
        }
        return MappingProxyType(snapshot)

    @staticmethod
    def _action_resource_key(intent: Mapping[str, Any]) -> str:
        """Derive contention key used for deterministic conflict resolution."""

        map_action = intent.get("map_action")
        if isinstance(map_action, Mapping):
            action = str(map_action.get("action", "unknown"))
            if action in {"gather", "build", "move"}:
                x = map_action.get("x", map_action.get("dx", ""))
                y = map_action.get("y", map_action.get("dy", ""))
                return f"map:{action}:{x}:{y}"
        recipient = intent.get("message_recipient_id")
        if recipient:
            return f"message:{recipient}"
        return f"agent:{intent.get('agent_id', '')}"

    @staticmethod
    def _is_governance_sensitive(intent: Mapping[str, Any]) -> bool:
        action = str(intent.get("action_intent", ""))
        return action in {
            AgentActionIntent.REQUEST_ROLE_CHANGE.value,
            AgentActionIntent.PROPOSE_IDEA.value,
            AgentActionIntent.CREATE_PROJECT.value,
            AgentActionIntent.JOIN_PROJECT.value,
            AgentActionIntent.LEAVE_PROJECT.value,
        }

    def _resolve_tick_conflicts(
        self: Self, intents: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Resolve resource/message/governance conflicts in deterministic order."""

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        locked_resources: set[str] = set()
        governance_claimed = False

        for intent in sorted(intents, key=self._deterministic_commit_sort_key):
            resource_key = str(intent.get("resource", ""))
            if resource_key in locked_resources:
                intent["merge_outcome"] = "rejected_resource_conflict"
                rejected.append(intent)
                continue
            if self._is_governance_sensitive(intent):
                if governance_claimed:
                    intent["merge_outcome"] = "rejected_governance_conflict"
                    rejected.append(intent)
                    continue
                governance_claimed = True

            locked_resources.add(resource_key)
            intent["merge_outcome"] = "accepted"
            accepted.append(intent)

        return accepted, rejected

    @staticmethod
    def _deterministic_commit_sort_key(plan: Mapping[str, Any]) -> tuple[str, str, str, int]:
        """Sort key that stabilizes replay for same-resource/target commit conflicts."""

        return (
            str(plan.get("resource", "agent_turn")),
            str(plan.get("target", "")),
            str(plan.get("agent_id", "")),
            int(plan.get("batch_index", 0)),
        )

    async def _run_step_pipeline(
        self: Self,
        max_turns: int,
        *,
        parallel_decision: bool = True,
    ) -> list[dict[str, Any]]:
        """Run a phased step pipeline: perception, parallel planning, deterministic commit."""

        if not self.agents:
            return []

        batch_size = min(max_turns, len(self.agents))
        base_step = self.current_step + 1

        tick_snapshot = self._build_tick_read_snapshot(turn_index=base_step)

        phase_start = time.perf_counter()
        plans: list[dict[str, Any]] = []
        for idx in range(batch_size):
            agent_index = (self.current_agent_index + idx) % len(self.agents)
            agent = self.agents[agent_index]
            plans.append(
                {
                    "batch_index": idx,
                    "agent_index": agent_index,
                    "agent_id": agent.agent_id,
                    "simulation_step": base_step + idx,
                    "resource": "agent_turn",
                    "target": agent.agent_id,
                    "tick_snapshot": tick_snapshot,
                    "snapshot": self._build_step_perception_snapshot(base_step + idx),
                }
            )
        self._set_labeled_gauge(
            STEP_PHASE_LATENCY_MS,
            phase="perception_snapshot",
            value=(time.perf_counter() - phase_start) * 1000,
        )
        self._set_labeled_gauge(
            STEP_PHASE_QUEUE_DEPTH, phase="planning_batch_size", value=len(plans)
        )

        phase_start = time.perf_counter()
        async def _run_plan(plan: Mapping[str, Any]) -> Mapping[str, Any]:
            return await self.agents[int(plan["agent_index"])].run_turn(
                simulation_step=int(plan["simulation_step"]),
                environment_perception=dict(cast(Mapping[str, Any], plan["snapshot"])),
                memory_service=self.memory_service,
                vector_store_manager=self.vector_store_manager,
                knowledge_board=self.knowledge_board,
            )

        if parallel_decision:
            planning_results = await asyncio.gather(*[_run_plan(plan) for plan in plans])
        else:
            planning_results = []
            for plan in plans:
                planning_results.append(await _run_plan(plan))
        self._set_labeled_gauge(
            STEP_PHASE_LATENCY_MS,
            phase="concurrent_planning",
            value=(time.perf_counter() - phase_start) * 1000,
        )

        phase_start = time.perf_counter()
        intents: list[dict[str, Any]] = []
        for plan, output in zip(plans, planning_results, strict=False):
            plan["output"] = output
            if isinstance(output, Mapping):
                action_intent = str(output.get("action_intent", AgentActionIntent.IDLE.value))
                intent = {
                    "batch_index": int(plan["batch_index"]),
                    "agent_index": int(plan["agent_index"]),
                    "agent_id": str(plan["agent_id"]),
                    "simulation_step": int(plan["simulation_step"]),
                    "action_intent": action_intent,
                    "requested_action_intent": action_intent,
                    "message_content": output.get("message_content"),
                    "message_recipient_id": output.get("message_recipient_id"),
                    "map_action": output.get("map_action"),
                    "resource": self._action_resource_key(
                        {
                            "agent_id": plan["agent_id"],
                            "action_intent": action_intent,
                            "map_action": output.get("map_action"),
                            "message_recipient_id": output.get("message_recipient_id"),
                        }
                    ),
                    "target": str(output.get("target", plan["target"])),
                    "metadata": {
                        "governance_sensitive": self._is_governance_sensitive(
                            {"action_intent": action_intent}
                        ),
                        "tick_snapshot_turn": tick_snapshot["turn_index"],
                    },
                }
                intents.append(intent)

        ordered, rejected = self._resolve_tick_conflicts(intents)
        committed: list[dict[str, Any]] = []
        for intent in ordered:
            committed.append(intent)
            self.total_turns_executed += 1
        committed.extend(rejected)
        self.current_step += len(intents)
        self.current_agent_index = (self.current_agent_index + len(intents)) % len(self.agents)
        self._set_labeled_gauge(
            STEP_PHASE_LATENCY_MS,
            phase="deterministic_commit_apply",
            value=(time.perf_counter() - phase_start) * 1000,
        )
        self._set_labeled_gauge(
            STEP_PHASE_QUEUE_DEPTH, phase="commit_batch_size", value=len(committed)
        )
        return committed

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
            actual_hash = TraceHashService.compute(
                {k: v for k, v in event.items() if k != "trace_hash"}
            )
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Trace hash mismatch for event at step {event.get('step')}:"
                    f" expected {expected_hash}, computed {actual_hash}"
                )
        if event.get("type") == "agent_action":
            environment_context = event.get("environment_context")
            world_time = None
            if isinstance(environment_context, dict):
                world_time = environment_context.get("time")
            if world_time is None:
                world_time = event.get("world_time")
            if isinstance(world_time, dict):
                self.world_tick_index = int(world_time.get("world_tick", self.world_tick_index))
                self.world_hour = int(world_time.get("world_hour", self.world_hour))
                self.world_day = int(world_time.get("world_day", self.world_day))
                if world_time.get("world_season") is not None:
                    self.world_season = int(world_time["world_season"])
                elif self.world_season_length_days is None:
                    self.world_season = None
            else:
                if "world_hour" in event:
                    self.world_hour = int(event.get("world_hour", self.world_hour))
                if "world_day" in event:
                    self.world_day = int(event.get("world_day", self.world_day))
                if "world_season" in event:
                    world_season = event.get("world_season")
                    self.world_season = int(world_season) if world_season is not None else None
            aid = event.get("agent_id")
            for agent in self.agents:
                if agent.agent_id == aid:
                    if "ip" in event:
                        agent.state.ip = float(event["ip"])
                    if "du" in event:
                        agent.state.du = float(event["du"])
                    break
            kb = event.get("knowledge_board")
            if isinstance(kb, dict):
                entries = kb.get("entries", [])
                if isinstance(entries, list):
                    self.knowledge_board.replace_entries(entries)
            wm = event.get("world_map")
            if isinstance(wm, dict):
                self.world_map.width = int(wm.get("width", self.world_map.width))
                self.world_map.height = int(wm.get("height", self.world_map.height))
                self.world_map.agent_positions = {
                    k: tuple(v) for k, v in wm.get("agents", {}).items()
                }
                self.world_map.resources = wm.get("resources", {})
                self.world_map.buildings = wm.get("buildings", {})
                self.world_map.agent_resources = wm.get("agent_resources", {})
                self.world_map.obstacles = set(wm.get("obstacles", []))
            step = event.get("step")
            if isinstance(step, int) and step > self.current_step:
                self.current_step = step
        elif event.get("type") == "environment_change":
            env = event.get("env", {})
            if isinstance(env, dict):
                from src.infra.checkpoint import restore_environment

                restore_environment(env)
        elif event.get("type") == "moderation":
            action = event.get("action")
            agent_id = event.get("agent_id")

            if not isinstance(agent_id, str):
                return

            from .resources import apply_penalty as _apply_penalty
            from .resources import mute_agent as _mute_agent
            from .resources import reset_memory as _reset_memory
            from .resources import unmute_agent as _unmute_agent

            if action == "mute":
                _mute_agent(self, agent_id)
            elif action == "unmute":
                _unmute_agent(self, agent_id)
            elif action == "reset_memory":
                _reset_memory(self, agent_id)
            elif action == "penalty":
                ip_val = event.get("ip", 0.0)
                du_val = event.get("du", 0.0)

                try:
                    ip = float(ip_val)
                except (TypeError, ValueError):
                    ip = 0.0

                try:
                    du = float(du_val)
                except (TypeError, ValueError):
                    du = 0.0

                _apply_penalty(self, agent_id, ip, du)
        elif event.get("type") == "tick":
            step = event.get("step")
            if isinstance(step, int) and step > self.current_step:
                self.current_step = step
            if "ip" in event:
                self.collective_ip = float(event["ip"])
            if "du" in event:
                self.collective_du = float(event["du"])
        elif event.get("type") == "human_command":
            step = event.get("step")
            if isinstance(step, int) and step > self.current_step:
                self.current_step = step
            sender = str(event.get("sender_id", "human"))
            default_step = int(step) if isinstance(step, int) else self.current_step
            raw_messages = event.get("messages", [])
            reconstructed: list[SimulationMessage] = []
            if isinstance(raw_messages, list):
                for msg in raw_messages:
                    if not isinstance(msg, Mapping):
                        continue
                    msg_step = msg.get("step", default_step)
                    try:
                        msg_step_int = int(msg_step)
                    except (TypeError, ValueError):
                        msg_step_int = default_step
                    reconstructed.append(
                        cast(
                            SimulationMessage,
                            {
                                "step": msg_step_int,
                                "turn_index": int(msg.get("turn_index", msg_step_int)),
                                "world_time": cast(dict[str, Any], msg.get("world_time") or {}),
                                "sender_id": str(msg.get("sender_id", sender)),
                                "recipient_id": msg.get("recipient_id"),
                                "content": str(msg.get("content", "")),
                                "action_intent": msg.get(
                                    "action_intent",
                                    AgentActionIntent.SEND_DIRECT_MESSAGE.value,
                                ),
                                "sentiment_score": msg.get("sentiment_score"),
                            },
                        )
                    )
            if not reconstructed:
                text = str(event.get("text", ""))
                target = event.get("target_agent_id")
                recipients: list[str | None]
                if event.get("broadcast"):
                    recipients = [agent.agent_id for agent in self.agents]
                elif isinstance(target, str):
                    recipients = [target]
                else:
                    recipients = [None]
                reconstructed = [
                    cast(
                        SimulationMessage,
                        {
                            "step": default_step,
                            "turn_index": int(event.get("turn_index", default_step)),
                            "world_time": cast(dict[str, Any], event.get("world_time") or {}),
                            "sender_id": sender,
                            "recipient_id": recipient,
                            "content": text,
                            "action_intent": AgentActionIntent.SEND_DIRECT_MESSAGE.value,
                            "sentiment_score": None,
                        },
                    )
                    for recipient in recipients
                ]
            budget_id = event.get("budget_agent_id") or event.get("target_agent_id")
            try:
                ip_cost = float(event.get("ip_cost", 0.0))
            except (TypeError, ValueError):
                ip_cost = 0.0
            try:
                du_cost = float(event.get("du_cost", 0.0))
            except (TypeError, ValueError):
                du_cost = 0.0
            if isinstance(budget_id, str):
                for agent in self.agents:
                    if agent.agent_id == budget_id:
                        agent.state.ip -= ip_cost
                        agent.state.du -= du_cost
                        break
            self.pending_messages_for_next_round.extend(reconstructed)
            self.messages_to_perceive_this_round.extend(reconstructed)
        elif event.get("type") == "snapshot":
            step = event.get("step")
            if isinstance(step, int):
                snapshot = SnapshotPersistenceService.load(step)
                if snapshot.get("trace_hash") != expected_hash:
                    raise ValueError(
                        f"Snapshot hash mismatch at step {step}:"
                        f" event {expected_hash} != file {snapshot.get('trace_hash')}"
                    )
                self._last_trace_hash = snapshot.get("trace_hash", "")
        elif event.get("type") == "agent_lifecycle_transition":
            aid = event.get("agent_id")
            if not isinstance(aid, str):
                return
            for agent in self.agents:
                if agent.agent_id != aid:
                    continue
                to_state_raw = str(event.get("to_state", AgentLifecycleState.ACTIVE.value))
                to_state = AgentLifecycleState(to_state_raw)
                history = list(getattr(agent.state, "lifecycle_history", []) or [])
                history.append(
                    {
                        "step": int(event.get("step", self.current_step)),
                        "from": str(event.get("from_state", "active")),
                        "to": to_state.value,
                        "reason": str(event.get("reason", "")),
                    }
                )
                agent.state.lifecycle_state = to_state
                agent.state.lifecycle_history = history
                agent.state.legacy_artifacts = dict(event.get("legacy_artifacts") or {})
                agent.state.memory_archival_policy = dict(
                    event.get("memory_archival_policy") or {}
                )
                break

    @classmethod
    def from_snapshot(cls: type[Self], snapshot: dict[str, Any], seed: int | None = None) -> Self:
        """Create a ``Simulation`` instance from a snapshot dictionary."""
        return cast(Self, PersistenceEngine().from_snapshot(cls, snapshot, seed=seed))

    @classmethod
    def _from_snapshot_impl(cls: type[Self], snapshot: dict[str, Any], seed: int | None = None) -> Self:
        from src.agents.core.base_agent import Agent  # avoid circular import at module level

        snapshot = migrate_snapshot(snapshot)

        agents_data = snapshot.get("agents", [])
        agents = [Agent(agent_id=a.get("agent_id", str(i))) for i, a in enumerate(agents_data)]
        sim_seed = seed if seed is not None else snapshot.get("seed")
        class _OfflineTokenizer:
            def encode(self, text: str) -> list[int]:
                return [ord(ch) for ch in text]

        sim = cls(
            agents=agents,
            scenario="",
            seed=sim_seed,
            memory_service=MemoryService(tokenizer=_OfflineTokenizer()),
        )
        if seed is None and snapshot.get("rng_state") is not None:
            from src.infra.checkpoint import restore_rng_state

            restore_rng_state(snapshot["rng_state"])
        sim.current_step = int(snapshot.get("step", 0))
        env_snapshot = snapshot["environment_state"]
        sim.world_hour = int(env_snapshot["world_hour"])
        sim.world_day = int(env_snapshot["world_day"])
        sim.world_tick_index = int(env_snapshot["world_tick"])
        world_season_value = env_snapshot.get("world_season")
        sim.world_season = int(world_season_value) if world_season_value is not None else None
        sim.environment_state.weather = str(env_snapshot["weather"])
        sim.environment_state.season_effects = dict(env_snapshot["season_effects"])
        sim.environment_state.active_global_modifiers = list(
            env_snapshot["active_global_modifiers"]
        )
        sim.environment_state.council_window_active = bool(env_snapshot["council_window_active"])
        snapshot_turn_quantum = int(snapshot.get("turns_per_world_tick", sim.turns_per_world_tick))
        sim.turns_per_world_tick = max(1, snapshot_turn_quantum)
        sim.environment_system.turns_per_world_tick = sim.turns_per_world_tick
        if sim.world_tick_index < 0 and sim.current_step > 0:
            sim.world_tick_index = (sim.current_step - 1) // sim.turns_per_world_tick
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
                    ag.state.lifecycle_state = AgentLifecycleState(
                        str(a_data.get("lifecycle_state", AgentLifecycleState.ACTIVE.value))
                    )
                    ag.state.lifecycle_history = list(a_data.get("lifecycle_history", []))
                    ag.state.legacy_artifacts = dict(a_data.get("legacy_artifacts", {}))
                    ag.state.memory_archival_policy = dict(
                        a_data.get("memory_archival_policy", {})
                    )
                    ag.state.predecessor_id = a_data.get("predecessor_id")
                    ag.state.successor_id = a_data.get("successor_id")
                    break

        kb = snapshot.get("knowledge_board", {})
        if isinstance(kb, dict):
            sim.knowledge_board.from_snapshot(kb)

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
        seed: int | None = None,
        events_path: str | Path | None = None,
    ) -> Self:
        """Load a snapshot and replay events from the event log."""
        sim = cast(
            Self,
            PersistenceEngine().replay_from_snapshot(
                cls,
                snapshot_path,
                seed=seed,
                stop_step=end_step,
            ),
        )
        if start_step is None and events_path is None:
            return sim

        # Preserve legacy filtering arguments with a post-load replay pass.
        return cls._replay_from_snapshot_impl(
            snapshot_path,
            start_step=start_step,
            end_step=end_step,
            seed=seed,
            events_path=events_path,
        )

    @classmethod
    def _replay_from_snapshot_impl(
        cls: type[Self],
        snapshot_path: str | Path,
        *,
        start_step: int | None = None,
        end_step: int | None = None,
        seed: int | None = None,
        events_path: str | Path | None = None,
        stop_step: int | None = None,
    ) -> Self:
        snap = SnapshotPersistenceService.load(snapshot_path)
        sim = cls._from_snapshot_impl(snap, seed=seed)
        from src.infra import event_log

        effective_end_step = stop_step if stop_step is not None else end_step
        after_step = sim.current_step
        if start_step is not None:
            after_step = max(after_step, start_step - 1)

        for event in event_log.stream_events(
            after_step=after_step, end_step=effective_end_step, path=events_path
        ):
            step = int(event.get("step", 0))
            if start_step is not None and step < start_step:
                continue
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

        _ = agents
        return await self._run_step_pipeline(max_turns=len(agents))

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
        project_description: str | None = None,
    ) -> str | None:
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

    def current_rules(self: Self) -> list[dict[str, Any]]:
        """Read API for agents/UI: current executable governance rules."""
        return governance_rules_engine.current_rules()

    def pending_votes(self: Self) -> list[dict[str, Any]]:
        """Read API for agents/UI: pending governance votes."""
        return governance_rules_engine.pending_votes()

    def active_offices(self: Self) -> list[dict[str, Any]]:
        """Read API for agents/UI: active governance offices."""
        return governance_rules_engine.active_offices()

    def sanctions(self: Self) -> list[dict[str, Any]]:
        """Read API for agents/UI: sanctions and enforcement records."""
        return governance_rules_engine.sanctions()

    def get_governance_read_model(self: Self) -> dict[str, Any]:
        """Return governance read models for rules, voting, offices, sanctions, and stances."""
        read_model: dict[str, Any] = {
            "rules": governance_rules_engine.current_rules(),
            "current_rules": governance_rules_engine.current_rules(),
            "pending_votes": governance_rules_engine.pending_votes(),
            "active_offices": governance_rules_engine.active_offices(),
            "sanctions": governance_rules_engine.sanctions(),
        }
        board = self.knowledge_board
        try:
            semantic_queries = as_semantic_query_store(board)
        except UnsupportedKnowledgeBoardCapabilityError:
            semantic_queries = None

        if semantic_queries is not None:
            proposals = semantic_queries.get_active_proposals(limit=20)
            read_model["active_proposals"] = proposals
            read_model["consensus_status"] = [
                semantic_queries.get_consensus_status(str(p.get("entry_id", "")))
                for p in proposals
                if p.get("entry_id")
            ]
            read_model["agent_stance_history"] = {
                agent.agent_id: semantic_queries.get_agent_stance_history(agent.agent_id)
                for agent in self.agents
            }
        return read_model

    async def propose_law(
        self: Self, proposer_id: str, text: str, vote_weights: dict[str, int] | None = None
    ) -> bool:
        """Allow an agent to propose a law and trigger a weighted vote."""
        from src.governance.service import governance

        proposer = next((a for a in self.agents if a.agent_id == proposer_id), None)
        if proposer is None:
            return False

        result = await governance.propose_law(
            proposer,
            text,
            self.agents,
            vote_weights,
        )
        if isinstance(result, bool):
            return result
        return bool(result.get("approved"))

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


def _hook_coalitions(sim: "Simulation", _events: list[Any]) -> dict[str, Any]:
    """Return the number of active coalitions (projects with >1 member)."""

    coalition_count = sum(1 for proj in sim.projects.values() if len(proj.get("members", [])) > 1)
    return {"coalitions": coalition_count}


def _hook_sentiment(sim: "Simulation", _events: list[Any]) -> dict[str, Any]:
    """Return the average sentiment (mood level) across agents."""

    avg_sentiment = (
        sum(agent.state.mood_level for agent in sim.agents) / len(sim.agents)
        if sim.agents
        else 0.0
    )
    return {"sentiment": avg_sentiment}


def _hook_collective_du(sim: "Simulation", _events: list[Any]) -> dict[str, Any]:
    """Return the collective DU across all agents."""

    return {"collective_du": sim.collective_du}


def _hook_collective_ip(sim: "Simulation", _events: list[Any]) -> dict[str, Any]:
    """Return the collective IP across all agents."""

    return {"collective_ip": sim.collective_ip}


EVALUATION_HOOKS: dict[str, Callable[["Simulation", list[Any]], dict[str, Any] | None]] = {
    "coalitions": _hook_coalitions,
    "sentiment": _hook_sentiment,
    "collective_du": _hook_collective_du,
    "collective_ip": _hook_collective_ip,
}


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
