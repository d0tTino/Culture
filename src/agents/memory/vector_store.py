#!/usr/bin/env python
"""
Provides management for vector storage of agent memories using ChromaDB.

This module is part of the agent memory system, handling persistence and retrieval
of agent memories using vector embeddings for semantic search capabilities.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, TypeVar, Union, cast

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pydantic import ValidationError
from typing_extensions import Self

from src.shared.memory_store import MemoryStore

# Attempt a more standard import for SentenceTransformerEmbeddingFunction
try:
    from chromadb.exceptions import ChromaDBException
except ImportError:
    ChromaDBException = Exception

# Constants for memory usage tracking
USAGE_TRACKING_FIELDS = [
    "retrieval_count",
    "last_retrieved_timestamp",
    "accumulated_relevance_score",
    "retrieval_relevance_count",
]

# Configure logger
logger = logging.getLogger(__name__)

# Add type alias for ChromaDB metadata
ChromaMeta = dict[str, Any]
ChromaMetaDict = dict[str, Union[str, int, float, bool]]
ChromaMetaList = list[ChromaMeta]

# Add helper to safely get first element of a list of lists
T = TypeVar("T")

# Ensure chromadb version compatibility (optional, for debugging)
logger.info(f"Using ChromaDB version: {chromadb.__version__}")


def first_list_element(lst: list[list[T]] | object) -> list[T]:
    if isinstance(lst, list) and len(lst) > 0 and isinstance(lst[0], list):
        return lst[0]
    return []


class ChromaVectorStoreManager(MemoryStore):
    """
    Manages a vector store for agent memories using ChromaDB with SentenceTransformer embeddings.

    This class handles:
    - Initialization and connection to a persistent ChromaDB instance
    - Adding agent memory events to the vector store with appropriate metadata
    - Retrieving relevant memories based on semantic similarity
    - Tracking role changes and retrieving role-specific memories
    - Usage statistics tracking for advanced memory pruning
    """

    def __init__(
        self: Self,
        persist_directory: str = "./chroma_db",
        embedding_function: Any | None = None,
    ) -> None:
        """Initialize the vector store manager.

        Args:
            persist_directory: Path where ChromaDB will persist data.
            embedding_function: Optional embedding function to use. If not
                provided, ``SentenceTransformerEmbeddingFunction`` is used. This
                parameter allows tests to provide a lightweight stub so that the
                heavy ``sentence-transformers`` dependency isn't required.
        """
        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)

        if embedding_function is None:
            try:
                embedding_function = SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            except Exception:  # pragma: no cover - dependency missing
                logger.warning("sentence-transformers not available, using dummy embeddings")

                def _dummy_embedding(texts: Sequence[str]) -> list[list[float]]:
                    return [[0.0] * 384 for _ in texts]

                embedding_function = _dummy_embedding

        self.embedding_function: Any = embedding_function

        # Initialize the persistent ChromaDB client
        logger.info(
            f"Initializing ChromaDB client with persistence directory: {persist_directory}"
        )
        if hasattr(chromadb, "PersistentClient"):
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:  # pragma: no cover - old chromadb
            from chromadb.config import Settings

            self.client = chromadb.Client(
                Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
            )

        # Get or create a collection for agent memories
        collection_name = "agent_memories"
        self.collection: Any = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=cast(Any, self.embedding_function),
        )

        # Create or get a separate collection for role changes
        role_collection_name = "agent_roles"
        self.roles_collection: Any = self.client.get_or_create_collection(
            name=role_collection_name,
            embedding_function=cast(Any, self.embedding_function),
        )

        logger.info(
            f"Chroma collections '{collection_name}' and "
            f"'{role_collection_name}' loaded/created from "
            f"'{persist_directory}'."
        )

        # Import lazily to avoid a circular dependency with memory_tracking_manager
        from .memory_tracking_manager import MemoryTrackingManager

        self.tracking_manager = MemoryTrackingManager(self.collection)

        # For tracking memory retrieval performance
        self.retrieval_times: list[float] = []

        # For tracking event count
        self.event_count = 0

    # ------------------------------------------------------------------
    # MemoryStore protocol implementation
    # ------------------------------------------------------------------
    def add_documents(self: Self, documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        """Add a batch of documents with metadata via Chroma."""
        ids = [str(uuid.uuid4()) for _ in documents]
        embeddings = [self.get_embedding(doc) for doc in documents]
        for meta in metadatas:
            meta.setdefault("timestamp", time.time())
        try:
            self.collection.add(
                ids=ids,
                embeddings=cast(list[Sequence[float]], embeddings),
                metadatas=cast(list[ChromaMeta], metadatas),
                documents=documents,
            )
        except (
            ChromaDBException,
            ValidationError,
            OSError,
        ) as exc:  # pragma: no cover - defensive
            logger.error("Error adding documents: %s", exc)

    def query(self: Self, query: str, top_k: int = 1) -> list[dict[str, Any]]:
        """Return the ``top_k`` most relevant documents."""
        embedding = self.get_embedding(query)
        try:
            results = self.collection.query(
                query_embeddings=cast(list[Sequence[float]], [embedding]),
                n_results=top_k,
                include=["documents", "metadatas", "ids"],
            )
        except (
            ChromaDBException,
            ValidationError,
            OSError,
        ) as exc:  # pragma: no cover - defensive
            logger.error("Error querying documents: %s", exc)
            return []

        docs: list[dict[str, Any]] = []
        ids: list[str] = first_list_element(results.get("ids")) if results else []
        metas: list[Mapping[str, Any]] = (
            first_list_element(results.get("metadatas")) if results else []
        )
        texts: list[str] = first_list_element(results.get("documents")) if results else []
        for doc_id, meta, text in zip(ids, metas, texts):
            docs.append({"content": text, "metadata": dict(meta), "id": doc_id})
        return docs

    def prune(self: Self, ttl_seconds: int) -> None:
        """Remove entries older than ``ttl_seconds``."""
        cutoff = time.time() - ttl_seconds
        try:
            results = self.collection.get(
                where={"timestamp": {"$lt": cutoff}},
                include=["ids"],
            )
            ids = results.get("ids", []) if results else []
            if ids:
                self.collection.delete(ids=ids)
        except (
            ChromaDBException,
            ValidationError,
            OSError,
        ) as exc:  # pragma: no cover - defensive
            logger.error("Error pruning documents: %s", exc)

    def add_memory(
        self: Self,
        agent_id: str,
        step: int,
        event_type: str,
        content: str,
        memory_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a memory to the agent's memory store.

        Args:
            agent_id (str): Unique identifier for the agent
            step (int): The simulation step when this memory occurred
            event_type (str): Type of memory (e.g., "thought", "action", "observation")
            content (str): The actual content of the memory
            memory_type (str, optional): Optional type for grouping memories
                (e.g., "raw", "consolidated_summary")
            metadata (dict[str, Any], optional): Optional additional metadata

        Returns:
            str: Unique ID of the stored memory
        """
        try:
            embedding = self.get_embedding(content)
            memory_id = f"{agent_id}_{step}_{event_type}_{uuid.uuid4()}"
            metadata_dict = metadata or {}
            metadata_dict.update(
                {
                    "agent_id": agent_id,
                    "step": step,
                    "event_type": event_type,
                    "memory_type": memory_type or "raw",
                    "timestamp": datetime.utcnow().isoformat(),
                    "retrieval_count": 0,
                    "last_retrieved_timestamp": "",
                    "accumulated_relevance_score": 0.0,
                    "retrieval_relevance_count": 0,
                }
            )
            try:
                self.collection.add(
                    ids=[memory_id],
                    embeddings=cast(list[Sequence[float]], [embedding]),
                    metadatas=cast(list[ChromaMeta], [metadata_dict]),
                    documents=[content],
                )
            except (
                ChromaDBException,
                OSError,
                ValidationError,
                json.JSONDecodeError,
            ) as e:
                logger.error(
                    f"ChromaDB add_memory failed: agent_id={agent_id}, step={step}, "
                    f"event_type={event_type}, error={e}",
                    exc_info=True,
                )
                return ""
            return memory_id
        except (ChromaDBException, ValidationError, OSError) as e:
            logger.error(
                f"Error adding memory. agent_id={agent_id}, step={step}, "
                f"event_type={event_type}: {e}",
                exc_info=True,
            )
            return ""

    def record_role_change(
        self: Self, agent_id: str, step: int, previous_role: str, new_role: str
    ) -> str:
        """
        Records a role change event in the role-specific collection.

        Args:
            agent_id (str): The unique ID of the agent
            step (int): The simulation step when this role change occurred
            previous_role (str): The role the agent had before the change
            new_role (str): The new role assigned to the agent

        Returns:
            str: The unique ID of the stored role change event
        """
        # Create a formatted document text for the role change
        document_text = f"Step {step} [ROLE_CHANGE]: Changed from {previous_role} to {new_role}"

        # Create metadata dictionary
        metadata_dict = {
            "agent_id": agent_id,
            "step": step,
            "event_type": "role_change",
            "previous_role": previous_role,
            "new_role": new_role,
            "timestamp": str(uuid.uuid4()),
        }

        # Generate a unique ID for this role change entry
        unique_id = f"{agent_id}_{step}_role_change_{uuid.uuid4()}"

        # Add the document to the role collection
        try:
            self.roles_collection.add(
                documents=[document_text],
                metadatas=cast(list[ChromaMeta], [metadata_dict]),
                ids=[unique_id],
            )
            logger.info(
                f"Added role change to Chroma: Agent={agent_id}, Step={step}, "
                f"{previous_role} -> {new_role}"
            )
            return unique_id
        except (ChromaDBException, ValidationError, OSError) as e:
            logger.error(f"Failed to add role change to Chroma: {e}")
            return ""

    def _update_memory_usage_stats(
        self: Self,
        memory_ids: list[str],
        relevance_scores: list[float] | None = None,
        increment_count: bool = True,
    ) -> None:
        """
        Update usage tracking metadata for memories that have been retrieved.
        This is a critical part of the advanced memory pruning strategy, tracking:
        1. How often each memory is retrieved (retrieval_count)
        2. When it was last accessed (last_retrieved_timestamp)
        3. How relevant it is when retrieved (accumulated_relevance_score)

        Args:
            memory_ids (list[str]): List of memory IDs that were retrieved
            relevance_scores (Optional[list[float]]): Optional list of relevance scores
                (0.0-1.0) for the retrieved memories
            increment_count (bool): Whether to increment the retrieval count
                (set to False for metadata-only lookups)
        """
        self.tracking_manager.update_usage_stats(
            memory_ids,
            relevance_scores,
            increment_count=increment_count,
        )

    def retrieve_relevant_memories(
        self: Self, agent_id: str, query: str, k: int = 3, include_usage_stats: bool = False
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories relevant to the query text, filtered to only include
        the specified agent's memories.

        Args:
            agent_id (str): The unique ID of the agent whose memories to search
            query (str): The text to find semantically similar memories to
            k (int): Maximum number of results to return
            include_usage_stats (bool): Whether to include usage statistics in the results

        Returns:
            List[Dict[str, Any]]: List of memory entries with content and metadata
        """
        # Start timing
        start_time = time.time()

        try:
            # Get embedding for query
            query_embedding = self.get_embedding(query)

            # Query the collection
            results = self.collection.query(
                query_embeddings=cast(list[Sequence[float]], [query_embedding]),
                n_results=k,
                where={"agent_id": agent_id},
                include=["documents", "metadatas", "distances"],
            )

            # Extract and format the results
            formatted_results = []
            memory_ids: list[str] = []
            relevance_scores: list[float] = []

            if results and "metadatas" in results and results["metadatas"]:
                metadatas: list[Mapping[str, str | int | float | bool]] = first_list_element(
                    results["metadatas"]
                )
                documents: list[str] = (
                    first_list_element(results.get("documents"))
                    if results.get("documents")
                    else []
                )
                memory_ids = first_list_element(results.get("ids")) if results.get("ids") else []

                # Calculate relevance scores (1 - distance)
                if results.get("distances") and results["distances"]:
                    distances: list[float] = first_list_element(results["distances"])
                    relevance_scores = [1.0 - float(distance) for distance in distances]

                # Update usage statistics
                if memory_ids:
                    self._update_memory_usage_stats(
                        memory_ids, relevance_scores, increment_count=True
                    )

                # Combine metadata and document content
                for i, metadata in enumerate(metadatas):
                    if i < len(documents):
                        memory_data = dict(metadata)
                        memory_data["content"] = documents[i]
                        if i < len(relevance_scores):
                            memory_data["relevance_score"] = relevance_scores[i]
                        if i < len(memory_ids):
                            memory_data["memory_id"] = memory_ids[i]
                        formatted_results.append(memory_data)

                # Record timing for benchmarking
                elapsed_time = time.time() - start_time
                self.retrieval_times.append(elapsed_time)

            # Remove usage tracking fields if not requested
            if not include_usage_stats:
                for memory in formatted_results:
                    for field in USAGE_TRACKING_FIELDS:
                        if field in memory:
                            del memory[field]

            return formatted_results

        except (ChromaDBException, OSError, ValidationError, json.JSONDecodeError) as e:
            logger.error(
                f"ChromaDB retrieve_relevant_memories failed: "
                f"agent_id={agent_id}, query={query}, error={e}",
                exc_info=True,
            )
            return []
        except (ChromaDBException, ValidationError, OSError) as e:
            logger.error(
                f"Unexpected error in retrieve_relevant_memories: "
                f"agent_id={agent_id}, query={query}, error={e}",
                exc_info=True,
            )
            return []

    async def aretrieve_relevant_memories(
        self: Self, agent_id: str, query: str, k: int = 3, include_usage_stats: bool = False
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self.retrieve_relevant_memories, agent_id, query, k, include_usage_stats
        )

    def retrieve_filtered_memories(
        self: Self,
        agent_id: str,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
        include_usage_stats: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories based on metadata filters.

        Args:
            agent_id (str): The agent ID to retrieve memories for
            filters (Optional[Dict[str, Any]]): Additional metadata filters to apply
            limit (Optional[int]): Maximum number of results to return
            include_usage_stats (bool): Whether to include usage statistics in the results

        Returns:
            List[Dict[str, Any]]: List of retrieved memories
        """
        try:
            # Build the where clause properly with $and operator
            where_conditions = [{"agent_id": agent_id}]

            if filters:
                for key, value in filters.items():
                    where_conditions.append({key: value})

            where_clause = (
                {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]
            )

            # Query the collection
            results = self.collection.get(
                where=cast(Any, where_clause),
                limit=limit,
                include=["metadatas", "documents", "embeddings"],
            )

            # Process the results
            memories: list[dict[str, Any]] = []
            memory_ids: list[str] = []

            if results and "ids" in results and results["ids"]:
                metadatas: list[Mapping[str, str | int | float | bool]] | None = results.get(
                    "metadatas"
                )
                documents: list[str] | None = results.get("documents")
                for i, memory_id in enumerate(results["ids"]):
                    if (
                        metadatas is not None
                        and documents is not None
                        and i < len(metadatas)
                        and i < len(documents)
                    ):
                        memory_data = dict(metadatas[i])
                        memory_data["content"] = documents[i]
                        memory_data["memory_id"] = memory_id
                        memories.append(memory_data)
                        memory_ids.append(memory_id)

            # Update memory usage statistics
            if memory_ids:
                self._update_memory_usage_stats(memory_ids, increment_count=True)

            # Remove usage tracking fields if not requested
            if not include_usage_stats:
                for memory in memories:
                    for field in USAGE_TRACKING_FIELDS:
                        if field in memory:
                            del memory[field]

            return memories

        except (ChromaDBException, ValidationError, OSError) as e:
            logger.error(f"Error retrieving filtered memories: {e}")
            return []

    def get_role_history(self: Self, agent_id: str) -> list[dict[str, Any]]:
        """
        Retrieve the role history for a specific agent.

        Args:
            agent_id (str): The unique ID of the agent

        Returns:
            List[Dict[str, Any]]: List of role periods, each containing role,
            start_step, and end_step
        """
        try:
            # Query the role changes collection using proper where clause format
            where_clause = {"$and": [{"agent_id": agent_id}, {"event_type": "role_change"}]}

            results = self.roles_collection.get(
                where=where_clause, include=["metadatas"]
            )

            role_history = []

            # Process the results to build role periods
            if results and results.get("ids"):
                metadatas = results.get("metadatas")
                if metadatas is not None:
                    # Sort by step to ensure chronological order
                    sorted_indices = sorted(
                        range(len(metadatas)),
                        key=lambda i: int(metadatas[i].get("step", 0)),
                    )

                    prev_role = "unknown"
                    start_step = 0

                    for idx in sorted_indices:
                        metadata = metadatas[idx]
                        step = int(metadata.get("step", 0))
                        new_role = str(metadata.get("new_role", "unknown"))

                        # Add the previous role period
                        if prev_role != "unknown":
                            role_history.append(
                                {
                                    "role": prev_role,
                                    "start_step": start_step,
                                    "end_step": step - 1,
                                }
                            )

                        # Update for next period
                        prev_role = new_role
                        start_step = step

                    # Add the final role period (which continues to the present)
                    if prev_role != "unknown":
                        role_history.append(
                            {
                                "role": prev_role,
                                "start_step": start_step,
                                "end_step": None,  # None means "to the present"
                            }
                        )

            return role_history

        except (ChromaDBException, ValidationError, OSError) as e:
            logger.error(f"Error retrieving role history for agent {agent_id}: {e}")
            return []  # Return empty list on error instead of raising an exception

    def retrieve_role_specific_memories(
        self: Self,
        agent_id: str,
        query: str | None = None,
        role: str | None = None,
        k: int = 3,
        include_usage_stats: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories that are relevant to a specific role.
        Either retrieves memories from when the agent had the specified role,
        or memories semantically relevant to that role's responsibilities.

        Args:
            agent_id (str): The unique ID of the agent
            query (Optional[str]): The text to find semantically similar memories to (optional)
            role (Optional[str]): The role to filter memories by (optional)
            k (int): Maximum number of results to return
            include_usage_stats (bool): Whether to include usage statistics in the results

        Returns:
            List[Dict[str, Any]]: List of memory entries with content and metadata
        """
        # If no role specified, just do a normal query retrieval
        if not role:
            if not query:
                logger.warning(
                    "Neither role nor query provided for role-specific memory retrieval"
                )
            return []
            return self.retrieve_relevant_memories(
                agent_id, query, k, include_usage_stats=include_usage_stats
            )

        # If role is provided, get the role's start and end steps
        role_periods = []
        try:
            role_history = self.get_role_history(agent_id)
            for period in role_history:
                if period["role"] == role:
                    role_periods.append(period)
        except Exception as e:
            logger.error(f"Error retrieving role history for agent {agent_id}: {e}")
            # If we can't get role history, we'll just proceed with semantic search

        # If we have role periods and query is None, retrieve memories from those periods
        if role_periods and query is None:
            memories = []
            for period in role_periods:
                # For each role period, get memories from that time range
                start_step = period.get("start_step")
                end_step = period.get("end_step", float("inf"))  # If no end, use infinity

                where_clause = {"step": {"$gte": start_step}}

                if end_step != float("inf"):
                    where_clause["step"]["$lte"] = end_step

                period_memories = self.retrieve_filtered_memories(
                    agent_id,
                    filters=where_clause,
                    limit=k,
                    include_usage_stats=include_usage_stats,
                )

                memories.extend(period_memories)

                # Limit to top k memories if we have more
                if len(memories) > k:
                    memories = memories[:k]

            return memories

        # If no role periods or query is provided, do semantic search
        else:
            # Create a role-focused query if needed
            role_query = query
            if query and role:
                role_query = f"From the perspective of a {role}, {query}"

            # Do a semantic search
            return self.retrieve_relevant_memories(
                agent_id,
                query=role_query or f"Responsibilities and insights of a {role}",
                k=k,
                include_usage_stats=include_usage_stats,
            )

    def query_memories(
        self: Self,
        agent_id: str,
        query: str,
        k: int = 3,
        threshold: float | None = None,
        include_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        """Query memories based on semantic similarity to the query."""
        logger.debug(f"Retrieving relevant memories for agent={agent_id}, query='{query}', k={k}")

        # Get embedding for query
        query_embedding = self.get_embedding(query)

        # Search for similar memories
        results = self.collection.query(
            query_embeddings=cast(list[Sequence[float]], [query_embedding]),
            where={"agent_id": agent_id},
            n_results=k,
        )

        # Defensive: check for None in results
        memory_ids = (
            results["ids"][0]
            if results
            and "ids" in results
            and results["ids"]
            and isinstance(results["ids"], list)
            and len(results["ids"]) > 0
            else []
        )
        distances: list[float] = (
            results["distances"][0]
            if results
            and "distances" in results
            and results["distances"]
            and isinstance(results["distances"], list)
            and len(results["distances"]) > 0
            else []
        )

        # Convert distance to relevance score (1 - distance, as smaller distance = more relevant)
        relevance_scores = [1 - float(distance) for distance in distances]

        # Filter by threshold if provided
        filtered_memories = []
        if results and "metadatas" in results and results["metadatas"]:
            metadatas: list[Mapping[str, str | int | float | bool]] = first_list_element(
                results["metadatas"]
            )
            documents: list[str] = (
                first_list_element(results.get("documents")) if results.get("documents") else []
            )
            for i, memory_id in enumerate(memory_ids):
                relevance_score = relevance_scores[i] if i < len(relevance_scores) else 0.0
                if threshold is None or relevance_score >= threshold:
                    if i < len(metadatas):
                        memory_data = dict(metadatas[i])
                        if i < len(documents):
                            memory_data["content"] = documents[i]
                        memory_data["relevance_score"] = relevance_score
                        memory_data["memory_id"] = memory_id
                        filtered_memories.append(memory_data)

        # Update usage statistics for retrieved memories
        if memory_ids:
            self._update_memory_usage_stats(memory_ids, relevance_scores, increment_count=True)

        logger.debug(f"Retrieved {len(filtered_memories)} relevant memories for agent {agent_id}")

        if not include_metadata:
            # Remove metadata fields if not requested
            for memory in filtered_memories:
                for field in USAGE_TRACKING_FIELDS:
                    if field in memory:
                        del memory[field]

        return filtered_memories

    def get_l2_summaries_older_than(self: Self, max_age_days: int) -> list[str]:
        """
        Retrieve IDs of Level 2 summaries that are older than the specified maximum age in days.

        Args:
            max_age_days (int): Maximum age in days for an L2 summary before it's
                considered for pruning

        Returns:
            list[str]: List of document IDs for L2 summaries older than max_age_days
        """
        logger.debug(f"Checking for L2 summaries older than {max_age_days} days")

        # Get current time for age comparison
        current_time = datetime.utcnow()

        # Calculate the cutoff date (current time minus max_age_days)
        cutoff_date = current_time - timedelta(days=max_age_days)
        cutoff_date_str = cutoff_date.isoformat()

        logger.debug(f"Cutoff date for L2 pruning: {cutoff_date_str}")

        # Query all L2 summaries
        try:
            # Query for all memories with memory_type "chapter_summary"
            results = self.collection.get(
                where={"memory_type": "chapter_summary"}, include=["metadatas"]
            )
        except Exception as e:
            logger.error(f"Error querying L2 summaries: {e!s}")
            return []

        if not results or not results.get("ids"):
            logger.debug("No L2 summaries found")
            return []

        ids_to_prune = []

        # Check each summary's age
        metadatas = results.get("metadatas")
        if metadatas is not None:
            for i, doc_id in enumerate(results["ids"]):
                try:
                    metadata = metadatas[i]
                    if not metadata or "simulation_step_end_timestamp" not in metadata:
                        logger.warning(f"L2 summary {doc_id} missing timestamp metadata, skipping")
                        continue
                    timestamp_str = str(metadata["simulation_step_end_timestamp"])
                    if not isinstance(timestamp_str, str):
                        continue
                    summary_date = datetime.fromisoformat(timestamp_str)
                    if summary_date < cutoff_date:
                        logger.debug(
                            f"L2 summary {doc_id} with timestamp {timestamp_str} "
                            f"is older than cutoff ({cutoff_date_str})"
                        )
                        ids_to_prune.append(doc_id)
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"Error processing L2 summary {doc_id}: {e!s}")
                    continue

        logger.info(f"Found {len(ids_to_prune)} L2 summaries older than {max_age_days} days")
        return ids_to_prune

    def delete_memories_by_ids(self: Self, ids: list[str]) -> bool:
        """
        Delete memories from ChromaDB by their IDs.

        Args:
            ids (List[str]): List of document IDs to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not ids:
            logger.warning("No IDs provided for deletion")
            return False

        try:
            logger.debug(
                f"Attempting to delete {len(ids)} memories with IDs: "
                f"{ids[:5]}{'...' if len(ids) > 5 else ''}"
            )

            # Verify these IDs exist before trying to delete them
            count_before = self.collection.count()

            # Get at least one of the IDs to verify it exists
            sample_ids = ids[:1]
            sample_results = self.collection.get(ids=sample_ids, include=["metadatas"])

            if sample_results.get("ids"):
                logger.debug(f"Verified at least one ID exists: {sample_ids[0]}")
                if (
                    sample_results.get("metadatas")
                    and sample_results["metadatas"] is not None
                    and len(sample_results["metadatas"]) > 0
                ):
                    metadata = sample_results["metadatas"][0]
                    logger.debug(f"Sample metadata for ID {sample_ids[0]}: {metadata}")
            else:
                logger.warning(f"None of the sample IDs {sample_ids} exist in the collection!")

            # Delete the documents by their IDs
            logger.debug(
                f"Executing collection.delete with IDs: {ids[:5]}{'...' if len(ids) > 5 else ''}"
            )
            self.collection.delete(ids=ids)

            # Verify deletion by checking count
            count_after = self.collection.count()
            deleted_count = count_before - count_after

            if deleted_count > 0:
                logger.info(
                    f"Successfully deleted {deleted_count} memories from "
                    f"ChromaDB (expected {len(ids)})"
                )
                return True
            else:
                logger.warning(
                    f"Delete operation completed but no records were removed. "
                    f"Count before: {count_before}, after: {count_after}"
                )
                # If no records were deleted, check if any of the IDs still exist
                still_exist = []
                for id_chunk in [
                    ids[i : i + 10] for i in range(0, len(ids), 10)
                ]:  # Check in chunks of 10
                    try:
                        check_results = self.collection.get(ids=id_chunk, include=["metadatas"])
                        if check_results.get("ids") and check_results["ids"] is not None:
                            still_exist.extend(check_results["ids"])
                    except Exception as check_error:
                        logger.error(
                            f"Error checking existence of IDs after deletion: {check_error}"
                        )

                if still_exist:
                    logger.warning(
                        f"{len(still_exist)} IDs still exist after deletion attempt: {still_exist}"
                    )

                return False

        except Exception as e:
            logger.error(f"Error deleting memories from ChromaDB: {e}", exc_info=True)
            return False

    def get_memory_ids_in_step_range(
        self: Self, agent_id: str, memory_type: str, start_step: int, end_step: int
    ) -> list[str]:
        """
        Get IDs of memories within a specific step range for an agent.

        Args:
            agent_id (str): The agent ID to filter by
            memory_type (str): The memory type to filter by (e.g., "consolidated_summary")
            start_step (int): The inclusive start of the step range
            end_step (int): The inclusive end of the step range

        Returns:
            List[str]: List of memory IDs matching the criteria
        """
        try:
            logger.debug(
                f"Searching for memory IDs with agent_id={agent_id}, "
                f"memory_type={memory_type}, step range={start_step}-{end_step}"
            )

            # Build the where filter with correct ChromaDB syntax
            where_filter = {
                "$and": [
                    {"agent_id": {"$eq": agent_id}},
                    {"memory_type": {"$eq": memory_type}},
                    {"step": {"$gte": start_step}},
                    {"step": {"$lte": end_step}},
                ]
            }

            logger.debug(f"ChromaDB filter condition: {where_filter}")

            # Get the IDs only
            results = self.collection.get(
                where=cast(Any, where_filter),
                include=["metadatas"],
            )

            if results.get("ids") and results["ids"] is not None:
                # For debugging, log some of the metadata to verify what was found
                log_message = (
                    f"Found {len(results['ids'])} memories of type '{memory_type}' "
                    f"for agent {agent_id} in step range {start_step}-{end_step}"
                )

                # Add detailed metadata info for a few results to help with debugging
                if (
                    "metadatas" in results
                    and results["metadatas"]
                    and len(results["metadatas"]) > 0
                ):
                    log_message += ". Sample metadata:"
                    for i, metadata in enumerate(results["metadatas"]):
                        if i < 3:  # Show metadata for up to 3 entries
                            log_message += (
                                f"\n - ID {results['ids'][i]}: step="
                                f"{metadata.get('step', 'unknown')}, "
                                f"type={metadata.get('memory_type', 'unknown')}"
                            )

                logger.info(log_message)
                return [str(x) for x in results["ids"]]

            logger.info(
                f"No memories found for agent {agent_id} of type '{memory_type}' "
                f"in step range {start_step}-{end_step}"
            )

            # For debugging, try a broader search to see if any memories exist for this agent
            debug_results = self.collection.get(
                where=cast(Any, {"$and": [{"agent_id": {"$eq": agent_id}}]}),
                include=["metadatas"],
            )

            if debug_results.get("metadatas") and debug_results["metadatas"] is not None:
                metadatas = debug_results["metadatas"]
                # Check if there are any memories of this type for the agent
                memory_type_count = sum(
                    True for meta in metadatas if meta.get("memory_type") == memory_type
                )
                logger.debug(
                    f"For debugging: found {memory_type_count} total memories of type "
                    f"'{memory_type}' for agent {agent_id} across all steps"
                )

                # Check memory step distribution to see if any are in the range
                # but not being returned
                step_count = sum(
                    True
                    for meta in metadatas
                    if meta.get("memory_type") == memory_type
                    and meta.get("step") is not None
                    and start_step <= int(str(meta.get("step"))) <= end_step
                )

                if step_count > 0:
                    logger.warning(
                        f"Potential issue: found {step_count} memories of type "
                        f"'{memory_type}' within step range {start_step}-{end_step} "
                        f"in broader search, "
                        f"but they weren't returned by the filtered query"
                    )

            return []

        except Exception as e:
            logger.error(f"Error retrieving memory IDs in step range: {e}", exc_info=True)
            return []

    def _calculate_mus(self: Self, metadata: dict[str, Any]) -> float:
        """Calculate the Memory Utility Score (MUS) for a memory."""
        mus = self.tracking_manager.calculate_mus(metadata)
        if metadata.get("retrieval_count", 0) > 5 or mus > 0.7:
            logger.debug("High-usage memory MUS calculation: %.3f", mus)
        return mus

    def get_l1_memories_for_mus_pruning(
        self: Self, mus_threshold: float, min_age_days: int
    ) -> list[str]:
        """
        Return IDs of L1 summaries (memory_type == 'consolidated_summary') that are older than
        min_age_days and have MUS < mus_threshold.
        """
        from datetime import datetime

        logger.info(
            f"Checking for L1 summaries eligible for MUS-based pruning "
            f"(threshold={mus_threshold}, min_age_days={min_age_days})"
        )
        now = datetime.utcnow()
        # Query all L1 summaries
        try:
            results = self.collection.get(
                where={"memory_type": "consolidated_summary"}, include=["metadatas"]
            )
        except Exception as e:
            logger.error(f"Error querying L1 summaries for MUS pruning: {e}")
            return []
        if not results or not results.get("ids"):
            logger.info("No L1 summaries found for MUS pruning.")
            return []
        ids_to_prune = []
        checked = 0
        for doc_id, metadata in zip(results["ids"], (results["metadatas"] or [])):
            timestamp_str = metadata.get("simulation_step_timestamp")
            if not timestamp_str or not isinstance(timestamp_str, str):
                logger.debug(
                    f"L1 summary {doc_id} missing simulation_step_timestamp, skipping age check."
                )
                continue
            try:
                created_dt = datetime.fromisoformat(timestamp_str)
                age_days = (now - created_dt).days
            except Exception as e:
                logger.warning(
                    f"Invalid simulation_step_timestamp for L1 summary {doc_id}: "
                    f"{timestamp_str} ({e})"
                )
                continue
            if age_days < min_age_days:
                continue
            mus = self._calculate_mus(dict(metadata))
            checked += 1
            if mus < mus_threshold:
                ids_to_prune.append(doc_id)
        logger.info(
            f"Checked {checked} L1 summaries for MUS pruning, identified "
            f"{len(ids_to_prune)} candidates."
        )
        return ids_to_prune

    def get_l2_memories_for_mus_pruning(
        self: Self, mus_threshold: float, min_age_days: int
    ) -> list[str]:
        """
        Return IDs of L2 summaries (memory_type == 'chapter_summary') that are older than
        min_age_days and have MUS < mus_threshold.

        Args:
            mus_threshold (float): Minimum Memory Utility Score to retain a memory
            min_age_days (int): Minimum age in days for an L2 summary to be considered
                for pruning

        Returns:
            list[str]: List of document IDs for L2 summaries to be pruned
        """
        from datetime import datetime

        logger.info(
            f"Checking for L2 summaries eligible for MUS-based pruning "
            f"(threshold={mus_threshold}, min_age_days={min_age_days})"
        )

        now = datetime.utcnow()

        # Query all L2 summaries
        try:
            results = self.collection.get(
                where={"memory_type": "chapter_summary"}, include=["metadatas"]
            )
        except Exception as e:
            logger.error(f"Error querying L2 summaries for MUS pruning: {e}")
            return []

        if not results or not results.get("ids"):
            logger.info("No L2 summaries found for MUS pruning.")
            return []

        ids_to_prune = []
        checked = 0

        # Process each L2 summary
        for doc_id, metadata in zip(results["ids"], (results["metadatas"] or [])):
            timestamp_str = metadata.get("simulation_step_end_timestamp")
            if not timestamp_str or not isinstance(timestamp_str, str):
                logger.debug(
                    f"L2 summary {doc_id} missing simulation_step_end_timestamp, "
                    f"skipping age check."
                )
                continue
            try:
                created_dt = datetime.fromisoformat(timestamp_str)
                age_days = (now - created_dt).days
            except Exception as e:
                logger.warning(
                    f"Invalid simulation_step_end_timestamp for L2 summary {doc_id}: "
                    f"{timestamp_str} ({e})"
                )
                continue

            # Skip summaries that are too young
            if age_days < min_age_days:
                continue

            # Calculate MUS and check against threshold
            mus = self._calculate_mus(dict(metadata))
            checked += 1

            if mus < mus_threshold:
                ids_to_prune.append(doc_id)

        logger.info(
            f"Checked {checked} L2 summaries for MUS pruning, identified "
            f"{len(ids_to_prune)} candidates."
        )
        return ids_to_prune

    def prune_memories_hybrid(
        self: Self,
        l1_mus_threshold: float = 0.2,
        l2_mus_threshold: float = 0.3,
        l2_age_days: int = 30,
        l1_min_age_days: int = 0,
        l2_min_age_days: int = 0,
    ) -> int:
        """Prune memories using both MUS and age-based criteria."""

        ids_to_delete: list[str] = []

        l1_ids = self.get_l1_memories_for_mus_pruning(l1_mus_threshold, l1_min_age_days)
        ids_to_delete.extend(l1_ids)

        l2_ids = self.get_l2_memories_for_mus_pruning(l2_mus_threshold, l2_min_age_days)
        ids_to_delete.extend(l2_ids)

        old_l2 = self.get_l2_summaries_older_than(l2_age_days)
        ids_to_delete.extend([i for i in old_l2 if i not in ids_to_delete])

        if ids_to_delete:
            self.delete_memories_by_ids(ids_to_delete)

        return len(ids_to_delete)

    def get_embedding(self: Self, text: str) -> list[float]:
        """
        Get the embedding for a piece of text using the embedding function.

        Args:
            text (str): The text to embed

        Returns:
            List[float]: The embedding vector
        """
        try:
            embeddings = self.embedding_function([text])
            return list(embeddings[0])
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 384

    def get_metadata_without_tracking(self: Self, memory_ids: list[str]) -> list[dict[str, Any]]:
        """
        Get metadata for memories without incrementing tracking stats.

        Args:
            memory_ids (List[str]): List of memory IDs to get metadata for

        Returns:
            List[Dict[str, Any]]: List of metadata dictionaries
        """
        try:
            # Use direct collection access to get metadata without tracking
            results = self.collection.get(ids=memory_ids, include=["metadatas"])

            if not results or "metadatas" not in results or not results["metadatas"]:
                return []

            return [dict(m) for m in results["metadatas"]]
        except Exception as e:
            logger.error(f"Error getting metadata without tracking: {e}")
            return []
