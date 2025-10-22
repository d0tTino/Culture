from __future__ import annotations

import logging
import os
import pickle
import platform
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from src.agents.core.base_agent import Agent

if TYPE_CHECKING:
    from src.agents.memory.vector_store import ChromaVectorStoreManager
else:  # pragma: no cover - optional dependency
    try:
        from src.agents.memory.vector_store import ChromaVectorStoreManager
    except Exception:
        ChromaVectorStoreManager = None
from src.infra import config
from src.infra.event_log import log_event
from src.sim.graph_knowledge_board import GraphKnowledgeBoard
from src.sim.knowledge_board import BoardEntry, KnowledgeBoard
from src.sim.simulation import Simulation

logger = logging.getLogger(__name__)

# Prefixes for project-specific environment variables that should be captured
ENV_PREFIXES: tuple[str, ...] = (
    "CULTURE_",
    "OLLAMA_",
    "REDIS_",
    "WEAVIATE_",
    "OPENAI_",
    "ANTHROPIC_",
    "DISCORD_",
    "MEMORY_",
    "ROLE_",
    "IP_",
    "DU_",
)


def capture_rng_state() -> dict[str, Any]:
    """Return the current RNG state for ``random`` and ``numpy`` if available."""

    state = {"random": random.getstate()}
    try:  # pragma: no cover - optional dependency
        import numpy as np

        np_state_tuple = cast(tuple[Any, ...], np.random.get_state())
        state["numpy"] = (
            np_state_tuple[0],
            np_state_tuple[1].tolist(),
            np_state_tuple[2],
            np_state_tuple[3],
            np_state_tuple[4],
        )
    except ImportError:
        # ``numpy`` is optional; ignore if unavailable
        pass

    return state


def restore_rng_state(state: Any) -> None:
    """Restore RNG state for ``random`` and ``numpy``.

    Accepts states from :func:`capture_rng_state` or the legacy tuple-based
    format used in older checkpoints.
    """
    if isinstance(state, dict):
        random_state = state.get("random")
    else:  # backward compatibility
        random_state = state

    if random_state is not None:
        # JSON serialization converts tuples to lists; convert back recursively
        def _to_tuple(obj: Any) -> Any:
            if isinstance(obj, list):
                return tuple(_to_tuple(x) for x in obj)
            return obj

        random.setstate(_to_tuple(random_state))

    if isinstance(state, dict) and "numpy" in state:
        try:  # pragma: no cover - optional dependency
            import numpy as np

            np.random.set_state(_to_tuple(state["numpy"]))
        except ImportError:
            pass


def capture_environment() -> dict[str, Any]:
    """Capture relevant environment variables and system info."""
    from src.infra.config import DEFAULT_CONFIG

    env: dict[str, Any] = {}
    prefixes = ENV_PREFIXES
    for key, value in os.environ.items():
        if key in DEFAULT_CONFIG or any(key.startswith(p) for p in prefixes):
            env[key] = value
    env["python_version"] = sys.version
    env["platform"] = platform.platform()
    return env


def restore_environment(env: dict[str, Any]) -> None:
    """Restore environment variables captured by :func:`capture_environment`."""
    from src.infra import config

    for key, value in env.items():
        if key in {"python_version", "platform"}:
            continue
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    config.load_config(validate_required=False)
    log_event({"type": "environment_change", "env": env})


def _serialize_simulation(sim: Simulation) -> dict[str, Any]:
    """Convert a ``Simulation`` instance into a serializable dictionary."""
    return {
        "agents": [agent.state.to_dict(exclude_none=True) for agent in sim.agents],
        "current_step": sim.current_step,
        "current_agent_index": sim.current_agent_index,
        "scenario": sim.scenario,
        "vector_store_dir": getattr(
            sim.vector_store_manager,
            "persist_directory",
            None,
        ),
        "rng_state": capture_rng_state(),
        "environment": capture_environment(),
        "collective_ip": sim.collective_ip,
        "collective_du": sim.collective_du,
        "knowledge_board": sim.knowledge_board.to_dict(),
    }


def save_checkpoint(sim: Simulation, path: str | Path) -> None:
    """Serialize ``sim`` to ``path`` using pickle."""
    data = _serialize_simulation(sim)
    p = Path(path)
    with p.open("wb") as fh:
        pickle.dump(data, fh)
    logger.info("Checkpoint saved to %s", p)
    log_event({"type": "checkpoint", "path": str(p), "step": data.get("current_step")})


def load_checkpoint(
    path: str | Path, *, replay: bool = False
) -> tuple[Simulation, dict[str, Any]]:
    """Restore a ``Simulation`` instance and metadata from ``path``.

    If ``replay`` is ``True`` the function will retrieve events from the
    Redpanda log and apply them to the simulation after loading.
    """
    p = Path(path)
    with p.open("rb") as fh:
        data = pickle.load(fh)

    vector_store_manager = None
    vector_store_dir = data.get("vector_store_dir")
    if vector_store_dir:
        if ChromaVectorStoreManager is None:
            raise ImportError("chromadb is required to load vector store from checkpoint")
        vector_store_manager = ChromaVectorStoreManager(persist_directory=vector_store_dir)

    agents = [
        Agent(
            agent_id=agent_data.get("agent_id"),
            initial_state=agent_data,
            name=agent_data.get("name"),
            vector_store_manager=vector_store_manager,
        )
        for agent_data in data.get("agents", [])
    ]

    sim = Simulation(
        agents=agents,
        vector_store_manager=vector_store_manager,
        scenario=data.get("scenario", ""),
    )
    kb_entries = data.get("knowledge_board", {}).get("entries", [])
    backend = os.getenv("KNOWLEDGE_BOARD_BACKEND", config.KNOWLEDGE_BOARD_BACKEND)
    if backend == "graph":
        board = GraphKnowledgeBoard()
        for e in kb_entries:
            board.add_entry(
                BoardEntry(
                    content_full=e.get("content_full", ""),
                    entry_type=e.get("entry_type", "note"),
                    content_summary=e.get("content_summary"),
                    tags=e.get("tags"),
                    reference_metadata=e.get("reference_metadata"),
                ),
                e.get("agent_id", "unknown"),
                int(e.get("step", 0)),
            )
        sim.knowledge_board = board
    else:
        sim.knowledge_board = cast(
            GraphKnowledgeBoard | KnowledgeBoard, KnowledgeBoard(entries=kb_entries)
        )
    sim.current_step = data.get("current_step", 0)
    sim.current_agent_index = data.get("current_agent_index", 0)
    sim.collective_ip = data.get("collective_ip", 0.0)
    sim.collective_du = data.get("collective_du", 0.0)

    meta = {
        "rng_state": data.get("rng_state"),
        "environment": data.get("environment"),
    }
    log_event({"type": "load_checkpoint", "path": str(p), "step": data.get("current_step")})
    if replay:
        from src.infra.event_log import fetch_events

        events = fetch_events(after_step=sim.current_step)
        for event in events:
            sim.apply_event(event)
    return sim, meta
