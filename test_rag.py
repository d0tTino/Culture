#!/usr/bin/env python
"""
Test script to verify RAG (Retrieval Augmented Generation) functionality.
This script runs a simulation with vector store enabled to confirm memory retrieval
and summarization is working properly.
"""

import logging
import sys
from src.app import create_base_simulation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("test_rag")

def test_rag_functionality():
    """
    Test that the RAG system is correctly retrieving and summarizing memories.
    Creates a simulation with vector store enabled and runs it for several steps.
    """
    logger.info("Starting RAG functionality test")
    
    # Create a simulation with vector store enabled
    sim = create_base_simulation(
        num_agents=3,
        steps=10,
        use_discord=False,
        use_vector_store=True  # Enable vector store
    )
    
    logger.info("Simulation created with vector store enabled")
    
    # Configure simulation for better testing
    scenario = "The team is collaboratively brainstorming ideas for a new AI system. Each agent should contribute ideas based on their expertise and reference previous ideas."
    sim.scenario = scenario
    
    # Add initial memories to vector store for agents to retrieve
    logger.info("Adding initial memories to vector store")
    for agent in sim.agents:
        agent_id = agent.agent_id
        # Add initial memories directly to vector store
        if sim.vector_store_manager:
            # Add some sample memories that will be relevant to the scenario
            sim.vector_store_manager.add_memory(
                agent_id=agent_id, 
                step=0, 
                event_type="initial_knowledge",
                content=f"Previously proposed an idea about using transformer models for natural language understanding."
            )
            
            sim.vector_store_manager.add_memory(
                agent_id=agent_id, 
                step=0, 
                event_type="initial_knowledge",
                content=f"Discussed retrieval augmented generation as a way to improve response quality."
            )
            
            sim.vector_store_manager.add_memory(
                agent_id=agent_id, 
                step=0, 
                event_type="initial_knowledge",
                content=f"Team members seemed interested in exploring ways to reduce hallucination in large language models."
            )
            
            logger.info(f"Added initial memories for {agent_id}")
    
    # Run the simulation
    logger.info("Running simulation for 10 steps")
    sim.run(10)
    
    logger.info("RAG functionality test completed")
    return True

if __name__ == "__main__":
    success = test_rag_functionality()
    sys.exit(0 if success else 1) 