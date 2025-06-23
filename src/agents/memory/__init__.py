"""
Agent memory module for Culture.ai.

This module contains components related to agent memory management,
including persistence, retrieval, and memory utility operations.
"""

from src.agents.memory.memory_tracking_manager import MemoryTrackingManager
from src.agents.memory.semantic_memory_manager import SemanticMemoryManager

try:  # pragma: no cover - optional dependency
    from src.agents.memory.vector_store import ChromaVectorStoreManager
except Exception:  # pragma: no cover - fallback when chromadb missing
    ChromaVectorStoreManager = None  # type: ignore[misc, assignment]

__all__ = ["MemoryTrackingManager", "SemanticMemoryManager"]
if ChromaVectorStoreManager is not None:
    __all__.insert(0, "ChromaVectorStoreManager")
