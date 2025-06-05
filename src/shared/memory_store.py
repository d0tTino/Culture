from __future__ import annotations

import uuid
from typing import Any

from typing_extensions import Self


class ChromaMemoryStore:
    """Minimal in-memory stand-in for ChromaDB used in tests."""

    def __init__(self: Self, persist_directory: str | None = None) -> None:
        self._store: list[dict[str, Any]] = []

    def add_documents(self: Self, documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        for doc, meta in zip(documents, metadatas):
            entry = {"content": doc, "metadata": meta, "id": str(uuid.uuid4())}
            self._store.append(entry)

    def query(self: Self, query: str, top_k: int = 1) -> list[dict[str, Any]]:
        # Ignore query text for this simplified implementation
        return list(self._store[-top_k:])
