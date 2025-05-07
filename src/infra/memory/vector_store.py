"""
Provides management for vector storage of agent memories using ChromaDB.
"""

import uuid
import logging
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import Dict, Any, Optional, List, Tuple

# Configure logger
logger = logging.getLogger(__name__)

class ChromaVectorStoreManager:
    """
    Manages a vector store for agent memories using ChromaDB with SentenceTransformer embeddings.
    
    This class handles:
    - Initialization and connection to a persistent ChromaDB instance
    - Adding agent memory events to the vector store with appropriate metadata
    - Retrieving relevant memories based on semantic similarity
    - Tracking role changes and retrieving role-specific memories
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store manager with a persistent ChromaDB client.
        
        Args:
            persist_directory (str): Path where ChromaDB will persist data
        """
        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embedding function using sentence-transformers
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize the persistent ChromaDB client
        logger.info(f"Initializing ChromaDB client with persistence directory: {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create a collection for agent memories
        collection_name = "agent_memories"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        # Create or get a separate collection for role changes
        role_collection_name = "agent_roles"
        self.role_collection = self.client.get_or_create_collection(
            name=role_collection_name,
            embedding_function=self.embedding_function
        )
        
        logger.info(f"Chroma collections '{collection_name}' and '{role_collection_name}' loaded/created from '{persist_directory}'.")
    
    def add_memory(self, agent_id: str, step: int, event_type: str, content: str) -> str:
        """
        Add a memory event to the vector store with appropriate metadata.
        
        Args:
            agent_id (str): The unique ID of the agent
            step (int): The simulation step when this memory occurred
            event_type (str): Type of memory (thought, broadcast_sent, broadcast_perceived)
            content (str): The content of the memory
            
        Returns:
            str: The unique ID of the stored memory
        """
        # Create a formatted document text that includes context
        document_text = f"Step {step} [{event_type.upper()}]: {content}"
        
        # Create metadata dictionary with essential indexing fields
        metadata_dict = {
            'agent_id': agent_id,
            'step': step,
            'event_type': event_type,
            'timestamp': str(uuid.uuid4())  # Add a unique timestamp-like field
        }
        
        # Generate a unique ID for this memory entry
        unique_id = f"{agent_id}_{step}_{event_type}_{uuid.uuid4()}"
        
        # Add the document to the collection
        try:
            self.collection.add(
                documents=[document_text],  # The formatted memory text
                metadatas=[metadata_dict],  # Metadata for filtering
                ids=[unique_id]             # Unique identifier
            )
            logger.debug(f"Added memory to Chroma: ID={unique_id}, Agent={agent_id}, Step={step}, Type={event_type}")
            return unique_id
        except Exception as e:
            logger.error(f"Failed to add memory to Chroma: {e}")
            return ""
    
    def record_role_change(self, agent_id: str, step: int, previous_role: str, new_role: str) -> str:
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
            'agent_id': agent_id,
            'step': step,
            'event_type': 'role_change',
            'previous_role': previous_role,
            'new_role': new_role,
            'timestamp': str(uuid.uuid4())
        }
        
        # Generate a unique ID for this role change entry
        unique_id = f"{agent_id}_{step}_role_change_{uuid.uuid4()}"
        
        # Add the document to the role collection
        try:
            self.role_collection.add(
                documents=[document_text],
                metadatas=[metadata_dict],
                ids=[unique_id]
            )
            logger.info(f"Added role change to Chroma: Agent={agent_id}, Step={step}, {previous_role} -> {new_role}")
            return unique_id
        except Exception as e:
            logger.error(f"Failed to add role change to Chroma: {e}")
            return ""
    
    def retrieve_relevant_memories(self, agent_id: str, query_text: str, k: int = 3) -> List[str]:
        """
        Retrieve memories relevant to the query text, filtered to only include the specified agent's memories.
        
        Args:
            agent_id (str): The unique ID of the agent whose memories to search
            query_text (str): The text to find semantically similar memories to
            k (int): Maximum number of results to return
            
        Returns:
            List[str]: List of formatted memory strings
        """
        logger.debug(f"Retrieving relevant memories for agent={agent_id}, query='{query_text}', k={k}")
        
        try:
            # Query with agent_id filter to ensure privacy between agents
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k,
                where={"agent_id": agent_id},  # Critical: only retrieve this agent's memories
                include=["documents"]  # Only need document text
            )
            
            # Extract and return just the document strings
            if results and "documents" in results and results["documents"]:
                documents = results["documents"][0]  # First (only) query result
                logger.debug(f"Retrieved {len(documents)} relevant memories for agent {agent_id}")
                return documents
            
            logger.debug(f"No relevant memories found for agent {agent_id}")
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving memories for agent {agent_id}: {e}")
            return []
    
    def retrieve_role_history(self, agent_id: str) -> List[Tuple[int, str, str]]:
        """
        Retrieves the complete role change history for a specific agent.
        
        Args:
            agent_id (str): The unique ID of the agent
            
        Returns:
            List[Tuple[int, str, str]]: List of (step, previous_role, new_role) tuples
        """
        try:
            # Query role collection for all role changes of this agent
            results = self.role_collection.get(
                where={"agent_id": agent_id, "event_type": "role_change"},
                include=["metadatas"]
            )
            
            role_history = []
            if results and "metadatas" in results and results["metadatas"]:
                for metadata in results["metadatas"]:
                    step = metadata.get('step', 0)
                    prev_role = metadata.get('previous_role', 'unknown')
                    new_role = metadata.get('new_role', 'unknown')
                    role_history.append((step, prev_role, new_role))
                
                # Sort by step for chronological order
                role_history.sort(key=lambda x: x[0])
                
            return role_history
        except Exception as e:
            logger.error(f"Error retrieving role history for agent {agent_id}: {e}")
            return []
    
    def retrieve_role_specific_memories(self, agent_id: str, role: str, query_text: str, k: int = 3) -> List[str]:
        """
        Retrieves memories from periods when the agent had a specific role.
        First determines the step ranges when the agent had the specified role,
        then searches for memories within those step ranges.
        
        Args:
            agent_id (str): The unique ID of the agent
            role (str): The role to filter memories by
            query_text (str): The text to find semantically similar memories to
            k (int): Maximum number of results to return
            
        Returns:
            List[str]: List of formatted memory strings from when the agent had the specified role
        """
        # First, get the role history to determine when the agent had this role
        role_history = self.retrieve_role_history(agent_id)
        
        # If no role history, return empty result
        if not role_history:
            logger.debug(f"No role history found for agent {agent_id}")
            return []
        
        # Determine the steps when the agent had the specified role
        step_ranges = []
        current_role = None
        start_step = 0
        
        # Add the initial role if known
        if role_history and role_history[0][0] > 0:
            # We don't know the initial role, assume it's different from the first recorded new role
            current_role = "unknown"
            start_step = 0
        
        # Process role changes to build step ranges
        for step, prev_role, new_role in role_history:
            if current_role == role:
                # Agent was in the target role until this step
                step_ranges.append((start_step, step - 1))
            
            current_role = new_role
            start_step = step
            
        # Add the final range if the agent ended in the target role
        if current_role == role:
            step_ranges.append((start_step, float('inf')))  # Until the end of simulation
        
        # If no relevant step ranges, return empty result
        if not step_ranges:
            logger.debug(f"Agent {agent_id} never had role '{role}'")
            return []
        
        # Now query for memories within these step ranges
        all_memories = []
        for start, end in step_ranges:
            try:
                # Query with agent_id filter and step range
                where_clause = {
                    "agent_id": agent_id,
                    "step": {"$gte": start}
                }
                if end != float('inf'):
                    where_clause["step"]["$lte"] = end
                
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=k,
                    where=where_clause,
                    include=["documents"]
                )
                
                if results and "documents" in results and results["documents"]:
                    all_memories.extend(results["documents"][0])
            
            except Exception as e:
                logger.error(f"Error querying memories for agent {agent_id} in step range {start}-{end}: {e}")
        
        # Deduplicate and limit results
        unique_memories = list(dict.fromkeys(all_memories))
        return unique_memories[:k]
    
    def query_memories(self, 
                      query_text: str,
                      agent_id: Optional[str] = None, 
                      event_types: Optional[List[str]] = None,
                      n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store for memories similar to the query text.
        
        Args:
            query_text (str): The text to find similar memories to
            agent_id (Optional[str]): Filter by specific agent ID
            event_types (Optional[List[str]]): Filter by specific event types
            n_results (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of memory entries with their metadata
        """
        # Build where filter if any filters are specified
        where_filter = {}
        if agent_id:
            where_filter["agent_id"] = agent_id
        
        if event_types:
            # Note: ChromaDB has limited filter capabilities
            # This simplified approach will only work for a single event type
            if len(event_types) == 1:
                where_filter["event_type"] = event_types[0]
        
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter if where_filter else None
            )
            
            # Process and return the results
            if results and "documents" in results and results["documents"]:
                formatted_results = []
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if "metadatas" in results else {}
                    distance = results["distances"][0][i] if "distances" in results else None
                    
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "relevance": 1.0 - (distance or 0) if distance is not None else None
                    })
                return formatted_results
            return []
        except Exception as e:
            logger.error(f"Error querying memories: {e}")
            return [] 