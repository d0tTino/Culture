#!/usr/bin/env python
"""
Query Agent Memory

This utility script retrieves memories from an agent's ChromaDB store based on a query,
and returns the relevant context and a generated answer suitable for RAG assessment.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Import needed components from the project
from src.memory.chroma_vector_store import ChromaVectorStoreManager
from src.agents.dspy_programs.rag_context_synthesizer import RAGContextSynthesizer
from src.infra.dspy_ollama_integration import configure_dspy_ollama

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Query an agent's memory store.")
    parser.add_argument("--agent_id", type=str, required=True, help="ID of the agent to query")
    parser.add_argument("--query_file", type=str, required=True, 
                       help="Path to file containing the query text")
    parser.add_argument("--chroma_dir", type=str, default="./chroma_db",
                       help="Path to ChromaDB directory")
    parser.add_argument("--ollama_model", type=str, default="llama3:8b",
                       help="Ollama model to use for synthesis")
    parser.add_argument("--max_context_items", type=int, default=10,
                       help="Maximum number of memory items to retrieve")
    return parser.parse_args()

def load_query(query_file):
    """Load query from file."""
    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading query from {query_file}: {e}")
        sys.exit(1)

def query_agent_memory(agent_id, query, chroma_dir, max_context_items=10, ollama_model="llama3:8b"):
    """
    Query an agent's memory and generate a response.
    
    Args:
        agent_id: ID of the agent to query
        query: Query text
        chroma_dir: Path to ChromaDB directory
        max_context_items: Maximum number of memory items to retrieve
        ollama_model: Ollama model to use for synthesis
        
    Returns:
        Dict with retrieved context and synthesized answer
    """
    try:
        # Initialize ChromaDB vector store manager
        vector_store = ChromaVectorStoreManager(chroma_dir)
        
        # Configure DSPy with Ollama
        configure_dspy_ollama(ollama_model)
        
        # Initialize RAG synthesizer
        rag_synthesizer = RAGContextSynthesizer()
        
        # Retrieve relevant memory items
        memory_results = vector_store.search(
            collection_name=agent_id,
            query_texts=[query],
            n_results=max_context_items
        )
        
        # Extract documents from results
        if memory_results and len(memory_results) > 0:
            retrieved_items = memory_results[0]
            
            # Format context for synthesis
            context_items = []
            for idx, (doc, metadata) in enumerate(zip(retrieved_items["documents"], retrieved_items["metadatas"])):
                # Format metadata for display
                meta_str = " | ".join([f"{k}: {v}" for k, v in metadata.items() 
                                     if k in ["memory_type", "timestamp", "mus"]])
                
                # Add formatted context item
                context_items.append(f"[{idx+1}] {meta_str}\n{doc}")
            
            # Join all context items
            context = "\n\n".join(context_items)
            
            # Generate answer using RAG synthesizer
            synthesis_result = rag_synthesizer.forward(
                query=query,
                context=context
            )
            
            answer = synthesis_result.answer if hasattr(synthesis_result, 'answer') else "No answer generated"
            
            return {
                "retrieved_context": context,
                "answer": answer
            }
        else:
            return {
                "retrieved_context": "",
                "answer": "No relevant memories found for this query."
            }
    
    except Exception as e:
        print(f"Error querying agent memory: {e}", file=sys.stderr)
        return {
            "retrieved_context": "",
            "answer": f"Error: {str(e)}",
            "error": str(e)
        }

def main():
    """Main function."""
    args = parse_args()
    
    # Load query from file
    query = load_query(args.query_file)
    
    # Query agent memory
    result = query_agent_memory(
        agent_id=args.agent_id,
        query=query,
        chroma_dir=args.chroma_dir,
        max_context_items=args.max_context_items,
        ollama_model=args.ollama_model
    )
    
    # Format and print the result
    print(f"QUERY: {query}\n")
    print(f"RETRIEVED CONTEXT:\n{result['retrieved_context']}\n")
    print(f"ANSWER:\n{result['answer']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 