from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Protocol, runtime_checkable

from typing_extensions import Self


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for simple memory stores used in tests."""

    def add_documents(self: Any, documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        """Add a batch of documents with associated metadata."""

    def query(self: Any, query: str, top_k: int = 1) -> list[dict[str, Any]]:
        """Return the ``top_k`` most relevant documents."""

    def prune(self: Any, ttl_seconds: int) -> None:
        """Remove entries older than ``ttl_seconds`` based on ``timestamp`` metadata."""


class ChromaMemoryStore(MemoryStore):
    """Minimal in-memory stand-in for ChromaDB used in tests.

    If ``persist_directory`` is provided, the store will load from and save to
    ``chroma_memory.json`` in that directory.
    """

    def __init__(self: Self, persist_directory: str | None = None) -> None:
        self._store: list[dict[str, Any]] = []
        self._persist_path: str | None = None
        if persist_directory is not None:
            os.makedirs(persist_directory, exist_ok=True)
            self._persist_path = os.path.join(persist_directory, "chroma_memory.json")
            if os.path.exists(self._persist_path):
                try:
                    with open(self._persist_path, encoding="utf-8") as fh:
                        self._store = json.load(fh)
                except Exception:
                    self._store = []

    def add_documents(self: Self, documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        for doc, meta in zip(documents, metadatas):
            entry = {"content": doc, "metadata": meta, "id": str(uuid.uuid4())}
            self._store.append(entry)
        if self._persist_path is not None:
            with open(self._persist_path, "w", encoding="utf-8") as fh:
                json.dump(self._store, fh)

    def query(self: Self, query: str, top_k: int = 1) -> list[dict[str, Any]]:
        # Ignore query text for this simplified implementation
        return list(self._store[-top_k:])

    def prune(self: Self, ttl_seconds: int) -> None:
        threshold = time.time() - ttl_seconds
        remaining: list[dict[str, Any]] = []
        for entry in self._store:
            ts = entry.get("metadata", {}).get("timestamp")
            if ts is None or ts >= threshold:
                remaining.append(entry)
        self._store = remaining
        if self._persist_path is not None:
            with open(self._persist_path, "w", encoding="utf-8") as fh:
                json.dump(self._store, fh)


class WeaviateMemoryStore(MemoryStore):
    """Very small shim around a mocked Weaviate collection for tests."""

    def __init__(
        self: Self,
        client: Any,
        collection_name: str = "Memory",
    ) -> None:
        self.client = client
        if hasattr(client, "collections"):
            collections = client.collections
            if getattr(collections, "exists", lambda _name: True)(collection_name):
                self.collection = collections.get(collection_name)
            else:
                self.collection = collections.create(collection_name)
        else:
            raise RuntimeError("Invalid Weaviate client provided")
        self._store: list[dict[str, Any]] = []

    def add_documents(self: Self, documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        objects = []
        for doc, meta in zip(documents, metadatas):
            uid = meta.get("uuid", str(uuid.uuid4()))
            entry = {"content": doc, "metadata": meta, "id": uid}
            self._store.append(entry)
            obj = {"text": doc, **meta, "uuid": uid}
            objects.append(obj)
        if hasattr(self.collection.data, "insert_many"):
            self.collection.data.insert_many(objects)

    def query(self: Self, query: str, top_k: int = 1) -> list[dict[str, Any]]:
        result = None
        if hasattr(self.collection.query, "near_text"):
            result = self.collection.query.near_text(query=query, limit=top_k)
        objects = getattr(result, "objects", []) if result else []
        docs = []
        for obj in objects:
            props = getattr(obj, "properties", {})
            metadata = dict(props)
            content = metadata.pop("text", "")
            docs.append(
                {"content": content, "metadata": metadata, "id": str(getattr(obj, "uuid", ""))}
            )
        return docs

    def prune(self: Self, ttl_seconds: int) -> None:
        threshold = time.time() - ttl_seconds
        remaining: list[dict[str, Any]] = []
        for entry in self._store:
            ts = entry.get("metadata", {}).get("timestamp")
            if ts is None or ts >= threshold:
                remaining.append(entry)
            else:
                if hasattr(self.collection.data, "delete_by_id"):
                    self.collection.data.delete_by_id(entry["id"])
        self._store = remaining
