import logging
from typing import Any, Callable, Optional, cast

import weaviate
import weaviate.classes as wvc
from weaviate import WeaviateClient
from weaviate.collections import Collection

logger = logging.getLogger(__name__)


class WeaviateVectorStoreManager:
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
    client: WeaviateClient
    collection: Collection[dict[str, Any], None]

    def __init__(
        self,
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

    def _connect_client(self, url: str) -> WeaviateClient:
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

    def _ensure_collection_exists(self) -> Collection[dict[str, Any], None]:
        # Check if collection exists, else create with correct schema (vectorizer: none)
        if self.client.collections.exists(self.collection_name):
            return cast(
                Collection[dict[str, Any], None],
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
            # Mypy: Weaviate client returns Mapping, but we require dict[str, Any] for strict compliance
            return cast(Collection[dict[str, Any], None], collection)

    def add_memories(
        self,
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
            data_objects.append(
                wvc.data.DataObject(properties=props, vector=vector, uuid=meta.get("uuid"))
            )
        try:
            self.collection.data.insert_many(data_objects)
        except Exception as e:
            logger.error(f"Failed to add objects to Weaviate: {e}")

    def query_memories(
        self,
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

    def delete_memories(self, uuids: list[str]) -> None:
        """
        Delete objects by UUID.
        Args:
            uuids: List of UUIDs to delete
        """
        for uuid in uuids:
            try:
                self.collection.data.delete_by_id(uuid)
            except Exception as e:
                logger.error(f"Failed to delete object {uuid} from Weaviate: {e}")

    def delete_collection(self) -> None:
        """
        Delete the entire collection from Weaviate.
        """
        try:
            self.client.collections.delete(self.collection_name)
            logger.info(f"Deleted Weaviate collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to delete Weaviate collection '{self.collection_name}': {e}")

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    async def aretrieve_relevant_memories(
        self,
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
