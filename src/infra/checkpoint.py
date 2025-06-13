from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from src.agents.core.base_agent import Agent

try:  # pragma: no cover - optional dependency
    from src.agents.memory.vector_store import ChromaVectorStoreManager
except Exception:  # pragma: no cover - fallback when chromadb missing
    ChromaVectorStoreManager = None  # type: ignore[misc, assignment]
from src.sim.knowledge_board import KnowledgeBoard
from src.sim.simulation import Simulation

logger = logging.getLogger(__name__)


def _serialize_simulation(sim: Simulation) -> dict[str, Any]:
    """Convert a ``Simulation`` instance into a serializable dictionary."""
    return {
        "agents": [
            agent.state.model_dump() if hasattr(agent.state, "model_dump") else agent.state.dict()
            for agent in sim.agents
        ],

        "current_step": sim.current_step,
        "current_agent_index": sim.current_agent_index,
        "scenario": sim.scenario,
        "vector_store_dir": getattr(
            sim.vector_store_manager,
            "persist_directory",
            None,
        ),
    }


def save_checkpoint(sim: Simulation, path: str | Path) -> None:
    """Serialize ``sim`` to ``path`` using pickle."""
    data = _serialize_simulation(sim)
    p = Path(path)
    with p.open("wb") as fh:
        pickle.dump(data, fh)
    logger.info("Checkpoint saved to %s", p)


def load_checkpoint(path: str | Path) -> Simulation:
    """Restore a ``Simulation`` instance from ``path``."""
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
    sim.knowledge_board = KnowledgeBoard()
    sim.current_step = data.get("current_step", 0)
    sim.current_agent_index = data.get("current_agent_index", 0)
    return sim
