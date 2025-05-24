#!/usr/bin/env python
"""
Query Agent Memory

This utility script retrieves memories from an agent's ChromaDB store based on a query,
and returns the relevant context and a generated answer suitable for RAG assessment.
"""

import argparse
import sys
from typing import Any

from src.agents.dspy_programs.rag_context_synthesizer import RAGContextSynthesizer
from src.infra.dspy_ollama_integration import configure_dspy_with_ollama
from src.memory.chroma_vector_store import ChromaVectorStoreManager


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Query an agent's memory store.")
    parser.add_argument("--agent_id", type=str, required=True, help="ID of the agent to query")
    parser.add_argument(
        "--query_file", type=str, required=True, help="Path to file containing the query text"
    )
    parser.add_argument(
        "--chroma_dir", type=str, default="./chroma_db", help="Path to ChromaDB directory"
    )
    parser.add_argument(
        "--ollama_model", type=str, default="llama3:8b", help="Ollama model to use for synthesis"
    )
    parser.add_argument(
        "--max_context_items",
        type=int,
        default=10,
        help="Maximum number of memory items to retrieve",
    )
    return parser.parse_args()


def load_query(query_file: str) -> str:
    """Load query from file."""
    try:
        with open(query_file, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to load query file: {e}")


def query_agent_memory(
    agent_id: str,
    query: str,
    chroma_dir: str,
    max_context_items: int = 10,
    ollama_model: str = "llama3:8b",
) -> dict[str, Any]:
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
        configure_dspy_with_ollama(ollama_model)

        # Initialize RAG synthesizer
        rag_synthesizer = RAGContextSynthesizer()

        # Retrieve relevant memory items
        memory_results = vector_store.search(
            collection_name=agent_id, query_texts=[query], n_results=max_context_items
        )

        # Extract documents from results
        if memory_results and len(memory_results) > 0:
            retrieved_items = memory_results[0]

            # Format context for synthesis
            context_items = []
            for idx, (doc, metadata) in enumerate(
                zip(retrieved_items["documents"], retrieved_items["metadatas"])
            ):
                # Format metadata for display
                meta_str = " | ".join(
                    [
                        f"{k}: {v}"
                        for k, v in metadata.items()
                        if k in ["memory_type", "timestamp", "mus"]
                    ]
                )

                # Add formatted context item
                context_items.append(f"[{idx + 1}] {meta_str}\n{doc}")

            # Join all context items
            context = "\n\n".join(context_items)

            # Generate answer using RAG synthesizer
            answer = rag_synthesizer.synthesize(context=context, question=query)

            return {"retrieved_context": context, "answer": answer}
        else:
            return {
                "retrieved_context": "",
                "answer": "No relevant memories found for this query.",
            }

    except Exception as e:
        print(f"Error querying agent memory: {e}", file=sys.stderr)
        return {"retrieved_context": "", "answer": f"Error: {e!s}", "error": str(e)}


def main() -> int:
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
        ollama_model=args.ollama_model,
    )

    # Format and print the result
    print(f"QUERY: {query}\n")
    print(f"RETRIEVED CONTEXT:\n{result['retrieved_context']}\n")
    print(f"ANSWER:\n{result['answer']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
