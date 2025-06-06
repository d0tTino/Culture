from __future__ import annotations

import time
import uuid
from typing import Any, Protocol

from typing_extensions import Self


class MemoryStore(Protocol):
    """Protocol for simple memory stores used in tests."""

    def add_documents(self, documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        """Add a batch of documents with associated metadata."""

    def query(self, query: str, top_k: int = 1) -> list[dict[str, Any]]:
        """Return the ``top_k`` most relevant documents."""

    def prune(self, ttl_seconds: int) -> None:
        """Remove entries older than ``ttl_seconds`` based on ``timestamp`` metadata."""


class ChromaMemoryStore(MemoryStore):
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

    def prune(self: Self, ttl_seconds: int) -> None:
        threshold = time.time() - ttl_seconds
        remaining: list[dict[str, Any]] = []
        for entry in self._store:
            ts = entry.get("metadata", {}).get("timestamp")
            if ts is None or ts >= threshold:
                remaining.append(entry)
        self._store = remaining


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


class ChromaVectorStoreAdapter(MemoryStore):
    """Adapter to use :class:`ChromaVectorStoreManager` via the ``MemoryStore`` interface."""

    def __init__(self: Self, manager: Any, agent_id: str = "memory_store") -> None:
        self.manager = manager
        self.agent_id = agent_id
        self._step = 0

    def add_documents(self: Self, documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        for doc, meta in zip(documents, metadatas):
            self._step += 1
            self.manager.add_memory(
                agent_id=meta.get("agent_id", self.agent_id),
                step=meta.get("step", self._step),
                event_type=meta.get("event_type", "note"),
                content=doc,
                memory_type=meta.get("memory_type"),
                metadata=meta,
            )

    def query(self: Self, query: str, top_k: int = 1) -> list[dict[str, Any]]:
        return self.manager.query_memories(
            agent_id=self.agent_id, query=query, k=top_k, include_metadata=True
        )

    def prune(self: Self, ttl_seconds: int) -> None:
        cutoff = time.time() - ttl_seconds
        try:
            results = self.manager.collection.get(include=["metadatas", "ids"])
        except Exception:
            return
        ids_to_delete = []
        metadatas = results.get("metadatas") or []
        for i, meta in enumerate(metadatas):
            ts = meta.get("timestamp")
            if isinstance(ts, (int, float)) and ts < cutoff:
                ids_to_delete.append(results["ids"][i])
        if ids_to_delete:
            try:
                self.manager.delete_memories_by_ids(ids_to_delete)
            except Exception:
                pass


class WeaviateVectorStoreAdapter(MemoryStore):
    """Adapter for :class:`WeaviateVectorStoreManager` using the ``MemoryStore`` interface."""

    def __init__(self: Self, manager: Any) -> None:
        self.manager = manager
        self.embed = getattr(manager, "embedding_function", None)

    def add_documents(self: Self, documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        if not self.embed:
            raise RuntimeError("embedding_function is required for adding documents")
        vectors = [self.embed(doc) for doc in documents]
        self.manager.add_memories(documents, metadatas, vectors)

    def query(self: Self, query: str, top_k: int = 1) -> list[dict[str, Any]]:
        if not self.embed:
            return []
        vec = self.embed(query)
        return self.manager.query_memories(vec, n_results=top_k)

    def prune(self: Self, ttl_seconds: int) -> None:
        # TTL pruning not implemented for Weaviate adapter
        pass


def create_chroma_adapter(persist_directory: str = "./chroma_db") -> ChromaVectorStoreAdapter:
    """Convenience helper to build a ``ChromaVectorStoreAdapter``."""
    from src.agents.memory.vector_store import ChromaVectorStoreManager

    manager = ChromaVectorStoreManager(persist_directory=persist_directory)
    return ChromaVectorStoreAdapter(manager)


def create_weaviate_adapter(
    url: str = "http://localhost:8080", collection_name: str = "AgentMemory"
) -> WeaviateVectorStoreAdapter:
    """Convenience helper to build a ``WeaviateVectorStoreAdapter``."""
    from src.agents.memory.weaviate_vector_store_manager import WeaviateVectorStoreManager

    manager = WeaviateVectorStoreManager(url=url, collection_name=collection_name)
    return WeaviateVectorStoreAdapter(manager)
