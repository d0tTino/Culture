"""
Agent memory module for Culture.ai.

This module contains components related to agent memory management,
including persistence, retrieval, and memory utility operations.
"""

from typing import TYPE_CHECKING

from src.agents.memory.memory_tracking_manager import MemoryTrackingManager
from src.agents.memory.semantic_memory_manager import SemanticMemoryManager

if TYPE_CHECKING:
    from src.agents.memory.vector_store import ChromaVectorStoreManager
else:  # pragma: no cover - optional dependency
    try:
        from src.agents.memory.vector_store import ChromaVectorStoreManager
    except Exception:
        ChromaVectorStoreManager = None

__all__ = ["MemoryTrackingManager", "SemanticMemoryManager"]
if ChromaVectorStoreManager is not None:
    __all__.insert(0, "ChromaVectorStoreManager")
