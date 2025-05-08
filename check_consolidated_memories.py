#!/usr/bin/env python
"""
Script to check for consolidated memory summaries in the ChromaDB vector store.
"""

import chromadb
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("memory_check")

def main():
    # Connect to the ChromaDB vector store
    logger.info("Connecting to ChromaDB vector store...")
    client = chromadb.PersistentClient("./chroma_db")
    
    try:
        # Get the agent_memories collection
        collection = client.get_collection("agent_memories")
        logger.info("Successfully connected to agent_memories collection")
        
        # Query for consolidated summaries
        logger.info("Querying for consolidated summaries...")
        results = collection.get(
            where={"memory_type": "consolidated_summary"}
        )
        
        if not results or "documents" not in results or not results["documents"]:
            logger.warning("No consolidated summaries found in the vector store")
            return
        
        # Count summaries by agent
        summaries_by_agent = {}
        for i, doc in enumerate(results["documents"]):
            agent_id = results["metadatas"][i]["agent_id"]
            if agent_id not in summaries_by_agent:
                summaries_by_agent[agent_id] = []
            summaries_by_agent[agent_id].append(doc)
        
        # Print results
        logger.info(f"Found {len(results['documents'])} consolidated summaries across {len(summaries_by_agent)} agents")
        
        for agent_id, summaries in summaries_by_agent.items():
            logger.info(f"Agent {agent_id}: {len(summaries)} consolidated summaries")
            for i, summary in enumerate(summaries):
                if i < 2:  # Show just the first 2 summaries per agent
                    logger.info(f"  Summary {i+1}: {summary[:150]}...")
        
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")

if __name__ == "__main__":
    main() 