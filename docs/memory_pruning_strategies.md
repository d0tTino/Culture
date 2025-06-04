# Memory Pruning Strategies for Culture AI

## Introduction

As the "Culture: An AI Genesis Engine" simulation progresses, agents continually generate memories that are stored in ChromaDB. The hierarchical memory structure - with Level 1 session summaries and Level 2 chapter summaries - provides organization but doesn't prevent excessive growth. Without proper management, this can lead to:

1. **Performance degradation**: Larger collections result in slower retrieval operations
2. **Reduced relevance**: A vast sea of memories makes it harder to retrieve the most relevant ones
3. **Resource consumption**: Higher storage and computational demands
4. **Response quality deterioration**: Too many competing memories may cause confusing or diluted agent responses

This document outlines strategies for memory pruning to maintain optimal performance while preserving critical information.

## Strategy 1: Age-Based Pruning with Hierarchical Preservation

### Core Mechanism
This strategy prunes memories based on age (simulation steps), with separate retention policies for different hierarchy levels.

### Implementation Approach
1. **Level 1 Summaries**:
   - Maintain a rolling window of the N most recent Level 1 summaries (e.g., 50-100 steps)
   - Once a Level 1 summary exceeds this age threshold, check if it has been incorporated into a Level 2 summary
   - If incorporated, prune the Level 1 summary; otherwise, preserve it until incorporation

2. **Level 2 Summaries**:
   - Maintain a much longer retention period (e.g., 500-1000 steps)
   - Rely on the condensed nature of Level 2 summaries to maintain long-term information while using less storage

3. **Raw Memories**:
   - Aggressively prune raw memories (non-summarized) after a very short period (e.g., 10-20 steps)
   - Only preserve raw memories explicitly marked as "critical" by agents

### Interaction with Memory Hierarchy
This approach leverages the hierarchical structure by assuming that important information from Level 1 summaries is preserved in Level 2 summaries. It creates a natural progression where detailed recent memories transition to more condensed long-term representations.

### Pros & Cons

**Pros:**
- Simple implementation requiring only timestamp metadata
- Predictable memory usage patterns
- Aligns with natural human memory (detailed short-term, generalized long-term)
- Easy to tune by adjusting window sizes

**Cons:**
- May remove potentially relevant memories based solely on age
- Depends heavily on the quality of Level 2 summaries
- No consideration of memory importance or relevance
- Risk of losing specific details that might become important in future contexts

## Strategy 2: Relevance-Based Pruning with Usage Tracking

### Core Mechanism
This strategy tracks how frequently and effectively memories are retrieved and used during RAG operations, pruning those that consistently show low utility.

### Implementation Approach
1. **Usage Tracking**:
   - Add metadata fields to each memory: `retrieval_count`, `last_retrieved`, and `relevance_score_sum`
   - Update these fields each time a memory is retrieved during RAG
   - Calculate an overall "utility score": `(relevance_score_sum / retrieval_count) * recency_factor`

2. **Tiered Pruning**:
   - Run periodic pruning operations that remove a percentage of memories with the lowest utility scores
   - Implement different thresholds for different hierarchy levels:
     - Level 1: Prune the bottom 30% of rarely used/retrieved memories after reaching collection size threshold
     - Level 2: Prune only the bottom 10%, preserving most higher-level summaries

3. **Relevance Protection**:
   - Never prune memories that have been retrieved with high relevance scores (e.g., top 0.8+) in the past N steps
   - Implement a "cooldown period" for new memories before they become eligible for pruning

### Interaction with Memory Hierarchy
This approach treats the hierarchy as a signal of potential importance but ultimately lets usage patterns determine what to keep. Level 2 summaries naturally tend to have higher relevance scores as they contain condensed, important information.

### Pros & Cons

**Pros:**
- Preserves memories that prove useful regardless of age or type
- Adaptively responds to actual agent behavior and needs
- Less risk of removing memories that are still contextually relevant
- Continuous optimization of the memory corpus

**Cons:**
- More complex implementation requiring additional metadata and tracking
- Potential feedback loop where memories that aren't retrieved once become less likely to be retrieved again
- Computationally more expensive due to tracking requirements
- May require tuning of utility score calculations for optimal performance

## Strategy 3: Semantic Redundancy Reduction

### Core Mechanism
This strategy identifies and removes semantically similar or redundant memories, particularly focusing on consolidating information rather than simply deleting it.

### Implementation Approach
1. **Similarity Detection**:
   - Periodically run similarity detection across memories in the same hierarchy level
   - Use cosine similarity between embeddings to identify clusters of highly similar memories (e.g., similarity > 0.92)
   - For each cluster, identify the most comprehensive or recent memory as the "representative"

2. **Smart Consolidation**:
   - For Level 1 summaries: When similar memories are identified, consider generating a new consolidated memory that preserves unique information from each
   - For raw memories: Simply keep the most recent or most descriptive memory from each similarity cluster

3. **Metadata Enrichment**:
   - When pruning a memory due to redundancy, add a reference to the preserved "representative" memory
   - Maintain a small metadata record of pruned memory IDs and their representatives to avoid information loss in retrieval

### Interaction with Memory Hierarchy
This approach works within each level of the hierarchy independently. It's particularly effective for Level 1 summaries where similar experiences might be recorded multiple times with subtle variations.

### Pros & Cons

**Pros:**
- Reduces bloat without necessarily losing information
- Potentially improves retrieval quality by reducing noise from near-duplicates
- Can be combined with other strategies for a multi-faceted approach
- Mimics human memory consolidation of similar experiences

**Cons:**
- Computationally expensive to perform similarity comparisons across large memory sets
- Risk of false positives (treating distinct memories as similar)
- More complex implementation, especially for the consolidation generation
- Requires tuning of similarity thresholds for different memory types

## Recommendation

For the "Culture: An AI Genesis Engine" project, a **combined approach** leveraging elements from all three strategies would likely be most effective:

1. **Initial Implementation**: Start with the **Age-Based Pruning with Hierarchical Preservation** (Strategy 1) as it's the simplest to implement and provides immediate benefits with minimal risk.

2. **Enhanced Approach**: Gradually incorporate **Usage Tracking** (from Strategy 2) to refine the pruning process based on actual memory utility rather than just age.

3. **Long-term Optimization**: Once the system is more mature, implement **Semantic Redundancy Reduction** (Strategy 3) as a periodic optimization process that runs less frequently than the primary pruning operations.

This phased approach allows for immediate management of memory growth while building toward a more sophisticated system that preserves the most valuable information regardless of age.

## Technical Implementation Considerations for ChromaDB

### Basic Deletion Operations
```python
# Deletion by ID
collection.delete(ids=["memory_123", "memory_456"])

# Delete with where filter (ChromaDB Enterprise or equivalent)
collection.delete(
    where={"metadata.timestamp": {"$lt": one_week_ago}}
)
```

### Periodic Pruning Job
```python
def prune_memories():
    """Run on a schedule or when collection exceeds size threshold"""
    # Get candidate memories for pruning based on strategy
    candidates = get_pruning_candidates()
    
    # Delete the selected memories
    if candidates:
        collection.delete(ids=[c.id for c in candidates])
        
    # Log pruning statistics
    logger.info(f"Pruned {len(candidates)} memories")
```

### Tracking Retrieval Usage
```python
def update_memory_usage(memory_id, relevance_score):
    """Call after each memory retrieval"""
    # Get current metadata
    result = collection.get(ids=[memory_id])
    metadata = result["metadatas"][0]
    
    # Update usage statistics
    metadata["retrieval_count"] = metadata.get("retrieval_count", 0) + 1
    metadata["last_retrieved"] = current_timestamp
    metadata["relevance_score_sum"] = metadata.get("relevance_score_sum", 0) + relevance_score
    
    # Update the metadata
    collection.update(ids=[memory_id], metadatas=[metadata])
```

These examples provide a starting point for implementing the proposed pruning strategies within the ChromaDB framework. 
