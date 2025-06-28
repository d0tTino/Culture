#!/usr/bin/env python
"""Initialize agent memories in a ChromaDB vector store."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create agents with initial memories.")
    parser.add_argument("--n", type=int, default=1, help="Number of agents to initialize")
    parser.add_argument(
        "seed_text",
        nargs="?",
        default="Hello world",
        help="Seed memory text for each agent",
    )
    parser.add_argument(
        "--chroma_dir",
        type=str,
        default="./chroma_db",
        help="Directory where ChromaDB data is stored",
    )
    return parser.parse_args()


if TYPE_CHECKING:  # pragma: no cover - for type checking only
    pass


def main() -> int:
    from importlib import import_module

    ChromaVectorStoreManager: type[Any] = getattr(
        import_module("src.agents.memory.vector_store"),
        "ChromaVectorStoreManager",
    )
    store: Any

    args = parse_args()
    try:
        store = ChromaVectorStoreManager(persist_directory=args.chroma_dir)
    except Exception as exc:  # pragma: no cover - initialization errors
        print(f"Failed to initialize vector store: {exc}", file=sys.stderr)
        return 1

    for i in range(args.n):
        agent_id = f"agent_{i + 1}"
        mem_id = store.add_memory(agent_id, 0, "seed", args.seed_text)
        print(f"Initialized {agent_id} memory {mem_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
