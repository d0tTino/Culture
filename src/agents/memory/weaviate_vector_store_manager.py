import logging
import time
import uuid
from typing import Any, Callable, Optional, cast

import weaviate
import weaviate.classes as wvc
from typing_extensions import Self

from src.shared.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class WeaviateVectorStoreManager(MemoryStore):
    """
    Manages a vector store for agent memories using Weaviate with external
    (pre-computed) embeddings.
    - Connects to a Weaviate instance
    - Ensures the collection exists with the correct schema
    - Supports add, query, and delete operations
    """

    url: str
    collection_name: str
    embedding_function: Optional[Callable[[str], list[float]]]
    client: Any
    collection: Any

    def __init__(
        self: Self,
        url: str = "http://localhost:8080",
        collection_name: str = "AgentMemory",
        embedding_function: Optional[Callable[[str], list[float]]] = None,
    ) -> None:
        """
        Initialize the Weaviate vector store manager.
        Args:
            url (str): Weaviate endpoint URL
            collection_name (str): Name of the Weaviate collection
            embedding_function: Callable to generate embeddings (should match
            the one used for Chroma)
        """
        self.url = url
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.client = self._connect_client(url)
        self.collection = self._ensure_collection_exists()
        self._store: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # MemoryStore protocol implementation
    # ------------------------------------------------------------------
    def add_documents(self: Self, documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        """Add documents with freshly generated vectors using ``embedding_function``."""
        if self.embedding_function is None:
            raise RuntimeError("embedding_function must be provided to add documents")
        vectors = [self.embedding_function(doc) for doc in documents]
        for meta in metadatas:
            meta.setdefault("timestamp", time.time())
            meta.setdefault("uuid", str(uuid.uuid4()))
        self.add_memories(documents, metadatas, vectors)

    def query(self: Self, query: str, top_k: int = 1) -> list[dict[str, Any]]:
        if self.embedding_function is None:
            raise RuntimeError("embedding_function must be provided to query")
        vector = self.embedding_function(query)
        return self.query_memories(vector, n_results=top_k)

    def prune(self: Self, ttl_seconds: int) -> None:
        cutoff = time.time() - ttl_seconds
        ids_to_delete = []
        remaining = []
        for entry in self._store:
            ts = entry.get("metadata", {}).get("timestamp")
            if ts is not None and ts < cutoff:
                ids_to_delete.append(entry["id"])
            else:
                remaining.append(entry)
        self._store = remaining
        if ids_to_delete:
            self.delete_memories(ids_to_delete)

    def _connect_client(self: Self, url: str) -> object:
        # Parse host/port from URL
        import re

        m = re.match(r"https?://([\w\.-]+)(?::(\d+))?", url)
        if m:
            host = m.group(1)
            port = int(m.group(2)) if m.group(2) else 8080
        else:
            host = "localhost"
            port = 8080
        return weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=url.startswith("https"),
            grpc_host=host,
            grpc_port=50051,
            grpc_secure=False,
        )

    def _ensure_collection_exists(self: Self) -> object:
        # Check if collection exists, else create with correct schema (vectorizer: none)
        if self.client.collections.exists(self.collection_name):
            return cast(
                object,
                self.client.collections.get(self.collection_name),
            )
        else:
            from weaviate.classes.config import Configure, DataType, Property

            properties = [
                Property(name="text", data_type=DataType.TEXT),
                Property(name="timestamp", data_type=DataType.NUMBER),
                Property(name="agent_id", data_type=DataType.TEXT),
                Property(name="uuid", data_type=DataType.TEXT),
            ]
            collection = self.client.collections.create(
                name=self.collection_name,
                properties=properties,
                vectorizer_config=Configure.Vectorizer.none(),
            )
            logger.info(
                f"Created Weaviate collection '{self.collection_name}' with "
                f"external vector support."
            )
            # Mypy: Weaviate client returns Mapping, but we require dict[str, Any] for strict
            # compliance
            return cast(object, collection)

    def add_memories(
        self: Self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> None:
        """
        Add a batch of memories with pre-computed vectors.
        Args:
            texts: List of text chunks
            metadatas: List of metadata dicts (must include timestamp, agent_id, uuid)
            vectors: List of embedding vectors (same order as texts)
        """
        if not (len(texts) == len(metadatas) == len(vectors)):
            raise ValueError("Lengths of texts, metadatas, and vectors must match.")
        data_objects: list[Any] = []  # DataObject is not public, so use Any
        for text, meta, vector in zip(texts, metadatas, vectors):
            props = dict(meta)
            props["text"] = text
            uuid_val = meta.get("uuid", str(uuid.uuid4()))
            data_objects.append(
                wvc.data.DataObject(properties=props, vector=vector, uuid=uuid_val)
            )
            self._store.append({"content": text, "metadata": meta, "id": uuid_val})
        try:
            self.collection.data.insert_many(data_objects)
        except Exception as e:
            logger.error(f"Failed to add objects to Weaviate: {e}")

    def query_memories(
        self: Self,
        query_vector: list[float],
        n_results: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Query memories by vector similarity, with optional metadata filtering.
        Args:
            query_vector: The embedding vector to search with
            n_results: Number of results to return
            filter_dict: Optional dict for property-based filtering (e.g., {"agent_id": "agent_1"})
        Returns:
            List of dicts with text, metadata, and distance
        """
        filters: Any = None
        if filter_dict:
            # Build Weaviate v4 filter
            filters = None
            for k, v in filter_dict.items():
                f = wvc.query.Filter.by_property(k).equal(v)
                filters = f if filters is None else filters & f
        try:
            result = self.collection.query.near_vector(
                near_vector=query_vector,
                limit=n_results,
                filters=filters,
                return_properties=["text", "timestamp", "agent_id", "uuid"],
                return_metadata=wvc.query.MetadataQuery(distance=True),
            )
            return [
                {
                    **o.properties,
                    "uuid": str(o.uuid),
                    "distance": getattr(o.metadata, "distance", None),
                }
                for o in result.objects
            ]
        except Exception as e:
            logger.error(f"Weaviate query failed: {e}")
            return []

    def delete_memories(self: Self, uuids: list[str]) -> None:
        """
        Delete objects by UUID.
        Args:
            uuids: List of UUIDs to delete
        """
        for uid in uuids:
            try:
                self.collection.data.delete_by_id(uid)
                self._store = [e for e in self._store if e["id"] != uid]
            except Exception as e:
                logger.error(f"Failed to delete object {uid} from Weaviate: {e}")

    def delete_collection(self: Self) -> None:
        """
        Delete the entire collection from Weaviate.
        """
        try:
            self.client.collections.delete(self.collection_name)
            logger.info(f"Deleted Weaviate collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to delete Weaviate collection '{self.collection_name}': {e}")

    def close(self: Self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    async def aretrieve_relevant_memories(
        self: Self,
        agent_id: str,
        query: str,
        k: int = 3,
        include_usage_stats: bool = False,
    ) -> list[dict[str, Any]]:
        import asyncio

        # Use embedding_function to get the query vector
        if self.embedding_function is None:
            raise RuntimeError("embedding_function must be set for async retrieval")
        query_vector = await asyncio.to_thread(self.embedding_function, query)
        return await asyncio.to_thread(
            self.query_memories,
            query_vector,
            k,
            {"agent_id": agent_id},
        )
