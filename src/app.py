import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.export_traces import export_latest
from src.agents.core.base_agent import Agent
from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.extensions import load_plugins
from src.infra import config, event_log
from src.infra.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from src.infra.checkpoint import (
    restore_environment as _restore_environment,
)
from src.infra.checkpoint import (
    restore_rng_state as _restore_rng_state,
)
from src.infra.llm_client import LLMClientInitError, get_llm_client
from src.infra.logging_config import setup_logging
from src.infra.settings import settings
from src.infra.warning_filters import configure_warning_filters
from src.interfaces.dashboard_backend import DEFAULT_CONTEXT, SimulationEvent
from src.sim.context import SimulationContext
from src.sim.graph_knowledge_board import GraphKnowledgeBoard
from src.sim.knowledge_board import KnowledgeBoard
from src.sim.simulation import Simulation
from src.utils.loop_helper import use_uvloop_if_available

restore_rng_state = _restore_rng_state
restore_environment = _restore_environment


async def start_simulation(ctx: SimulationContext | None = None) -> None:
    """Enqueue a control command to start the simulation."""
    context = ctx or DEFAULT_CONTEXT
    await context.get_event_queue().put(SimulationEvent(type="control", data={"command": "start"}))


async def stop_simulation(ctx: SimulationContext | None = None) -> None:
    """Enqueue a control command to stop the simulation."""
    context = ctx or DEFAULT_CONTEXT
    await context.get_event_queue().put(SimulationEvent(type="control", data={"command": "stop"}))


async def spawn_agent_command(agent_id: str, ctx: SimulationContext | None = None) -> None:
    """Request spawning of a new agent via the event queue."""
    context = ctx or DEFAULT_CONTEXT
    await context.get_event_queue().put(
        SimulationEvent(type="control", data={"command": "spawn", "agent_id": agent_id})
    )


def _simple_yaml(path: Path) -> dict[str, object]:
    """Very small YAML parser for simple key/value files."""
    data: dict[str, object] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            value = value.strip().strip('"').strip("'")
            if value.isdigit():
                data[key.strip()] = int(value)
            else:
                data[key.strip()] = value
    return data


def load_scenario(
    value: str,
) -> tuple[
    str,
    int | None,
    int | None,
    list[str],
    list[str],
    dict[str, object],
    dict[str, object],
]:
    """Return scenario description and optional overrides from a file.

    The returned tuple contains the scenario description, optional overrides for
    ``steps`` and ``agents``, narrative beats, evaluation hook names, evaluation
    target definitions, and success metric specifications. When ``value`` does
    not reference a file the remaining items are empty defaults.
    """
    path = Path(value)
    if path.is_file():
        try:  # Try full YAML parsing if PyYAML is available
            import yaml

            content = yaml.safe_load(path.read_text())
        except Exception:  # pragma: no cover - fallback
            content = _simple_yaml(path)

        if isinstance(content, dict):
            desc = str(content.get("description") or content.get("scenario") or "")
            steps = content.get("steps")
            agents = content.get("agents")
            beats = content.get("beats")
            beat_list = [str(b) for b in beats] if isinstance(beats, list) else []
            hook_values = content.get("evaluation_hooks")
            if isinstance(hook_values, list):
                evaluation_hooks = [str(h) for h in hook_values if h is not None]
            elif hook_values is None:
                evaluation_hooks = []
            else:
                evaluation_hooks = [str(hook_values)]
            targets = content.get("evaluation_targets")
            evaluation_targets = (
                {str(key): value for key, value in targets.items()}
                if isinstance(targets, dict)
                else {}
            )
            success_defs = content.get("success_metrics")
            success_metrics = (
                {str(key): value for key, value in success_defs.items()}
                if isinstance(success_defs, dict)
                else {}
            )
            return (
                desc,
                int(steps) if steps is not None else None,
                int(agents) if agents is not None else None,
                beat_list,
                evaluation_hooks,
                evaluation_targets,
                success_metrics,
            )
        if isinstance(content, str):
            return content, None, None, [], [], {}, {}
        return str(content), None, None, [], [], {}, {}
    return value, None, None, [], [], {}, {}


use_uvloop_if_available()

# Discord bot integration is imported lazily to avoid circular imports when
# ``src.interfaces.discord_bot`` references functions from this module.
if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from src.interfaces.discord_bot import SimulationDiscordBot

simulation_discord_bot_class: type["SimulationDiscordBot"] | None = None

DEFAULT_SCENARIO = "Agents collaborate to design a specification for a communication protocol."


def create_simulation(
    num_agents: int = 3,
    steps: int = 10,
    scenario: str = DEFAULT_SCENARIO,
    beats: list[str] | None = None,
    evaluation_hook_names: list[str] | None = None,
    evaluation_targets: dict[str, object] | None = None,
    success_metrics: dict[str, object] | None = None,
    use_discord: bool = False,
    use_vector_store: bool = False,
    vector_store_dir: str = "./chroma_db",
    use_semantic_memory: bool = False,
    semantic_db_uri: str = "bolt://localhost:7687",
    semantic_user: str = "neo4j",
    semantic_password: str = "test",
    seed: int | None = None,
) -> Simulation:
    """Construct a Simulation instance with basic defaults."""

    try:
        ollama_client = get_llm_client()
    except LLMClientInitError as exc:
        logging.error("Failed to connect to LLM backend: %s", exc)
        sys.exit(1)

    discord_bot = None
    if use_discord:
        global simulation_discord_bot_class
        if simulation_discord_bot_class is None:
            try:
                from src.interfaces import discord_moderation  # noqa: F401
                from src.interfaces.discord_bot import SimulationDiscordBot

                simulation_discord_bot_class = SimulationDiscordBot
            except ImportError:  # pragma: no cover - optional dependency
                logging.warning(
                    "Discord bot module not found, running without Discord integration."
                )
        if simulation_discord_bot_class:
            bot_token_raw = str(settings.DISCORD_BOT_TOKEN)
            channel_id = settings.DISCORD_CHANNEL_ID
            if bot_token_raw and channel_id:
                tokens = [tok.strip() for tok in bot_token_raw.split(",") if tok.strip()]
                bot = asyncio.run(
                    simulation_discord_bot_class.create(
                        tokens if len(tokens) > 1 else tokens[0], int(channel_id)
                    )
                )
                if bot.is_ready:
                    discord_bot = bot
                else:
                    logging.warning("Discord bot not ready, running without integration.")

    agents = [Agent(agent_id=f"agent_{i + 1}", name=f"Agent_{i + 1}") for i in range(num_agents)]

    vector_store = (
        None
        if not use_vector_store
        else (
            ChromaVectorStoreManager(persist_directory=vector_store_dir)
            if ChromaVectorStoreManager is not None
            else None
        )
    )

    semantic_manager = None
    if use_semantic_memory and vector_store is not None:
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(semantic_db_uri, auth=(semantic_user, semantic_password))
            semantic_manager = SemanticMemoryManager(vector_store, driver)
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.error("Failed to connect to semantic DB: %s", exc)

    sim = Simulation(
        agents=agents,
        vector_store_manager=vector_store,
        semantic_manager=semantic_manager,
        scenario=scenario,
        beats=beats,
        discord_bot=discord_bot,
        seed=seed,
        evaluation_hook_names=evaluation_hook_names,
        evaluation_targets=evaluation_targets,
        success_metrics=success_metrics,
    )
    if config.KNOWLEDGE_BOARD_BACKEND == "graph":
        sim.knowledge_board = GraphKnowledgeBoard()
    else:
        sim.knowledge_board = KnowledgeBoard()
    sim.steps_to_run = steps
    return sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Culture.ai simulation.")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the Culture.ai version and exit.",
    )
    parser.add_argument("--agents", type=int, default=3, help="Number of agents.")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for Python and NumPy random number generators.",
    )
    parser.add_argument(
        "--scenario", type=str, default=DEFAULT_SCENARIO, help="Simulation scenario."
    )
    parser.add_argument("--discord", action="store_true", help="Enable Discord integration.")
    parser.add_argument(
        "--vector-store", action="store_true", help="Use ChromaDB for agent memory."
    )
    parser.add_argument(
        "--vector-dir",
        type=str,
        default="./chroma_db",
        help="Directory for vector store persistence.",
    )
    parser.add_argument(
        "--semantic-memory",
        action="store_true",
        help="Enable semantic memory consolidation.",
    )
    parser.add_argument(
        "--semantic-db",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j connection URI for semantic memory.",
    )
    parser.add_argument(
        "--semantic-user",
        type=str,
        default="neo4j",
        help="Neo4j username for semantic memory.",
    )
    parser.add_argument(
        "--semantic-password",
        type=str,
        default="test",
        help="Neo4j password for semantic memory.",
    )
    parser.add_argument(
        "--no-warning-filters",
        action="store_true",
        help="Disable default warning filters.",
    )
    parser.add_argument(
        "--log-suppressed-warnings",
        action="store_true",
        help="Log warnings that would otherwise be suppressed.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a checkpoint file to load and save simulation state.",
    )
    parser.add_argument(
        "--replay",
        type=str,
        metavar="SNAPSHOT",
        help="Replay a previous run from the given snapshot using the event log.",
    )
    parser.add_argument(
        "--replay-start",
        type=int,
        default=None,
        help="Start tick for event log replay.",
    )
    parser.add_argument(
        "--replay-end",
        type=int,
        default=None,
        help="End tick for event log replay.",
    )
    parser.add_argument(
        "--proposal",
        type=str,
        help="Submit a law proposal before running the simulation.",
    )
    parser.add_argument(
        "--proposer-id",
        type=str,
        default="agent_1",
        help="Agent ID submitting the proposal.",
    )
    parser.add_argument(
        "--export-dataset",
        type=str,
        metavar="PATH",
        help="Write a JSONL dataset from the latest snapshots after the run.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    # Load optional plugins before argument parsing
    asyncio.run(load_plugins())
    args = parse_args()
    (
        desc,
        file_steps,
        file_agents,
        beats,
        evaluation_hooks,
        evaluation_targets,
        success_metrics,
    ) = load_scenario(args.scenario)
    if file_steps is not None:
        args.steps = file_steps
    if file_agents is not None:
        args.agents = file_agents
    args.scenario = desc

    if args.version:
        from src import __version__

        print(__version__)
        return

    configure_warning_filters(
        apply_filters=not args.no_warning_filters,
        log_suppressed=args.log_suppressed_warnings,
    )

    if args.seed is None and args.replay:
        stored = event_log.get_seed()
        if stored is not None:
            logging.info("Using seed %s from event log", stored)
            args.seed = stored
    if args.seed is not None:
        event_log.set_seed(args.seed)

    sim: Simulation
    meta: dict[str, object] | None = None
    if args.replay:
        replay_kwargs = {
            "start_step": args.replay_start,
            "end_step": args.replay_end,
        }
        if args.seed is not None:
            replay_kwargs["seed"] = args.seed
        Simulation.replay_from_snapshot(
            args.replay,
            **replay_kwargs,
        )
        if args.export_dataset:
            out_path = Path(args.export_dataset)
            with out_path.open("w", encoding="utf-8") as fh:
                after = (args.replay_start or 0) - 1
                for ev in event_log.stream_events(after_step=after, end_step=args.replay_end):
                    step = int(ev.get("step", 0))
                    if args.replay_start is not None and step < args.replay_start:
                        continue
                    if args.replay_end is not None and step > args.replay_end:
                        break
                    fh.write(json.dumps(ev))
                    fh.write("\n")
        return

    if args.checkpoint and Path(args.checkpoint).exists():
        logging.info("Loading simulation from checkpoint %s", args.checkpoint)
        sim, meta = load_checkpoint(args.checkpoint)
        sim.steps_to_run = args.steps
    else:
        sim_kwargs = {
            "num_agents": args.agents,
            "steps": args.steps,
            "scenario": args.scenario,
            "beats": beats,
            "use_discord": args.discord,
            "use_vector_store": args.vector_store,
            "vector_store_dir": args.vector_dir,
            "use_semantic_memory": args.semantic_memory,
            "semantic_db_uri": args.semantic_db,
            "semantic_user": args.semantic_user,
            "semantic_password": args.semantic_password,
        }
        if evaluation_hooks:
            sim_kwargs["evaluation_hook_names"] = evaluation_hooks
        if evaluation_targets:
            sim_kwargs["evaluation_targets"] = evaluation_targets
        if success_metrics:
            sim_kwargs["success_metrics"] = success_metrics
        if args.seed is not None:
            sim_kwargs["seed"] = args.seed
        sim = create_simulation(**sim_kwargs)

    if args.proposal:
        asyncio.run(sim.forward_proposal(args.proposer_id, args.proposal))

    asyncio.run(sim.async_run(args.steps))

    if args.checkpoint:
        save_checkpoint(sim, args.checkpoint)

    if args.export_dataset:
        export_latest(output=args.export_dataset)


if __name__ == "__main__":
    main()
