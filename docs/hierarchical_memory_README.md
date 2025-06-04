# Hierarchical Memory System Overview

The Culture.ai agent memory system uses a hierarchical structure:

- **L1 Summaries:** Short-term, step/session-level memories. Pruned frequently using both age and MUS (Memory Utility Score).
- **L2 Summaries:** Long-term, chapter-level summaries synthesized from L1s. Pruned less frequently, with higher MUS threshold.

## MUS Pruning

- **MUS Formula:** (0.4 × Retrieval Frequency Score) + (0.4 × Relevance Score) + (0.2 × Recency Score)
- **L1 Pruning:** Age-based and MUS < 0.2
- **L2 Pruning:** Age-based and MUS < 0.3
- **Thresholds:** L1=0.2, L2=0.3 (l1_low_l2_medium configuration, see benchmarks/mus_threshold_tuning_report.md)

This approach ensures efficient memory management and optimal retrieval-augmented generation (RAG) performance.

# Hierarchical Memory Consolidation

This document describes the hierarchical memory consolidation system implemented for AI agents in our simulation.

## Overview

The hierarchical memory system allows agents to consolidate their experiences at different levels of abstraction, creating a memory structure similar to how humans organize memories into more compressed and meaningful chunks over time.

The system consists of two levels of memory consolidation:

1. **Level 1 - Session Summaries**: Generated frequently from recent short-term memories. These capture the immediate context and experiences of the agent.

2. **Level 2 - Chapter Summaries**: Generated approximately every 10 steps by consolidating multiple level 1 summaries. These provide a higher-level view of the agent's experiences over longer periods.

## Implementation Details

### Level 1 Memory Consolidation

- Occurs in the `update_state_node` function when enough short-term memories are available
- Uses the agent's LLM to summarize recent memories
- Stores the summary in both the agent's short-term memory and the vector store
- Contextualizes memories with agent's role and identity for better summaries

### Level 2 Memory Consolidation

- Occurs every 10 simulation steps
- Retrieves recent level 1 summaries from the vector store using the `retrieve_filtered_memories` method
- Generates a comprehensive "chapter summary" from these level 1 summaries
- Stores the chapter summary with appropriate metadata in both the agent's memory and vector store
- Tracks periods covered using `last_level_2_consolidation_step` in the agent state

## Vector Store Integration

The implementation uses ChromaDB as a vector store to persist memories for long-term retrieval. Key features:

- Semantic search capabilities for finding relevant memories
- Metadata filtering to distinguish between different memory types
- Long-term persistence across simulation runs

## Testing

Two test scripts are provided to verify the memory consolidation functionality:

1. `test_memory_consolidation.py`: Tests the basic level 1 memory consolidation
2. `test_level2_memory_consolidation.py`: Tests the full hierarchical memory system including level 2 consolidation

Run the tests using:

```bash
python run_level2_memory_test.py
```

## Code Structure

- `src/agents/graphs/basic_agent_graph.py`: Contains the memory consolidation logic in `update_state_node`
- `src/agents/core/agent_state.py`: Defines the agent state model with memory tracking fields
- `src/agents/memory/vector_store.py`: Implements vector store integration with filtering capabilities

## Future Enhancements

Potential improvements for the hierarchical memory system:

1. Level 3 consolidation for "narrative arc" summaries spanning the entire simulation
2. Topic-based memory clustering to organize memories by theme rather than just time
3. Cross-agent memory sharing for collective intelligence
4. Emotional tagging of memories to prioritize emotionally significant experiences
5. Memory decay mechanisms to model forgetting of less important details 
