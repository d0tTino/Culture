# Advanced Memory Pruning Design Proposal

## Introduction

The "Culture: An AI Genesis Engine" project currently implements a hierarchical memory system with age-based pruning for both Level 1 (session) and Level 2 (chapter) summaries. While effective for basic memory management, this approach relies solely on chronological age and doesn't consider the relevance, importance, or utility of memories. This document proposes more sophisticated pruning strategies to enhance memory management while preserving critical information.

## Current Limitations of Age-Based Pruning

The existing pruning system:

1. **Treats all memories equally**: No distinction is made between high-value and low-value memories of the same age
2. **Lacks contextual awareness**: Cannot preserve memories that remain relevant to current agent goals or ongoing situations
3. **Operates on fixed intervals**: Uses predetermined thresholds rather than adapting to memory access patterns
4. **Disregards memory utility**: Doesn't consider how frequently or effectively memories are retrieved and used
5. **May discard valuable rare insights**: Unique but infrequently accessed memories might be lost despite their potential future value

## Proposed Advanced Pruning Strategies

### Strategy 1: Relevance-Weighted Pruning with Usage Tracking

#### Core Concept
This strategy preserves memories based on their demonstrated utility in RAG operations, retaining those that have proven valuable regardless of age.

#### Metrics for Pruning

1. **Retrieval Frequency Score** (RFS)
   - Tracks how often a memory is retrieved from the vector store
   - Calculated as: `log(1 + retrieval_count)` to reduce the impact of very high counts

2. **Relevance Score** (RS) 
   - Measures how relevant a memory is when retrieved (based on cosine similarity)
   - Calculated as: `avg(relevance_scores)` where relevance_scores are recorded during retrieval

3. **Recency Score** (RecS)
   - Gives higher weight to recently accessed memories
   - Calculated as: `1 / (1 + days_since_last_accessed)` 

4. **Memory Utility Score** (MUS)
   - Composite score combining the above metrics
   - `MUS = (0.4 * RFS) + (0.4 * RS) + (0.2 * RecS)`

#### Pruning Logic

1. **Selective Pruning**:
   - When memory store reaches a size threshold, calculate MUS for all memories
   - Keep all memories with MUS above a configurable threshold
   - For memories below the threshold, retain those with highest MUS up to the target size
   - Apply different thresholds for L1 vs. L2 summaries (L2 more conservatively pruned)

2. **Protection Rules**:
   - Never prune memories accessed within the last N simulation steps (configurable "cooldown period")
   - Memories with exceptionally high relevance scores (e.g., >0.9) get extended protection

3. **Age Integration**:
   - Still consider age as a factor, but weighted against utility
   - Very old memories require higher utility scores to be preserved

#### Integration Points
- Update each memory's metadata during RAG operations to track retrieval frequency and relevance
- Run pruning checks at same intervals as current age-based system
- Add memory protection flags when certain retrieval thresholds are met

#### Data Requirements
- Add metadata fields to each memory:
  - `retrieval_count`: Incremented each time memory is retrieved
  - `last_retrieved`: Timestamp of most recent retrieval
  - `relevance_score_sum`: Running sum of relevance scores
  - `relevance_score_count`: Counter for calculating average
  - `memory_utility_score`: Calculated composite score

#### Pros & Cons

**Pros:**
- Preserves memories based on demonstrated value
- Adapts to agent's actual information needs
- Balances recency with importance
- Still considers age, but with nuance

**Cons:**
- More complex metadata management
- Additional computational overhead for score calculations
- Risk of feedback loop (memories not retrieved once become less likely to be retrieved again)
- Requires tuning to find optimal score thresholds

### Strategy 2: Goal-Aligned Semantic Pruning

#### Core Concept
This strategy preserves memories based on their semantic relevance to current agent goals and priorities, using embedding similarity to evaluate importance.

#### Metrics for Pruning

1. **Goal Relevance Score** (GRS)
   - Measures semantic similarity between memory and current agent goals
   - Calculated as: `max(cosine_similarity(memory_embedding, goal_embedding) for goal in agent.goals)`

2. **Semantic Uniqueness Score** (SUS)
   - Measures how unique a memory is compared to others in the store
   - Calculated as: `1 - max(cosine_similarity(memory_embedding, other_memory_embedding) for other_memory in nearby_memories)`

3. **Memory Importance Score** (MIS)
   - Composite score combining goal relevance and uniqueness
   - `MIS = (0.7 * GRS) + (0.3 * SUS)`

#### Pruning Logic

1. **Semantic Clustering**:
   - Periodically cluster memories using semantic embeddings
   - For each cluster, identify representative memories with highest MIS
   - Prune redundant memories within clusters, preserving the representatives

2. **Goal-Driven Preservation**:
   - Re-evaluate goal relevance when agent goals change
   - Protect memories highly relevant to current goals regardless of age
   - Adjust pruning thresholds based on goal stability (more aggressive pruning during stable periods)

3. **Smart Consolidation**:
   - For semantically similar memories, generate a consolidated version that preserves key information
   - Replace the redundant memories with the consolidated version
   - Track provenance to indicate what original memories contributed to the consolidation

#### Integration Points
- Perform semantic analysis during memory consolidation steps
- Recalculate goal relevance when agent roles or goals change
- Schedule deeper semantic pruning during idle periods to reduce runtime impact

#### Data Requirements
- Enhanced memory metadata:
  - `goal_relevance_score`: Relevance to current goals
  - `semantic_uniqueness_score`: Uniqueness measure
  - `cluster_id`: Identifier for semantic cluster
  - `is_representative`: Flag for cluster representatives
  - `consolidated_from`: List of memory IDs that were merged (if applicable)

#### Pros & Cons

**Pros:**
- Aligns pruning with agent's goals and purpose
- Preserves unique insights even if rarely accessed
- Reduces redundancy while maintaining information diversity
- More cognitively plausible (mimics human memory consolidation)

**Cons:**
- Computationally expensive (embedding comparisons)
- Requires periodic recalculation as goals change
- More complex implementation for consolidation generation
- Challenging to tune semantic similarity thresholds

## Recommended Strategy: Hybrid Utility-Semantic Approach

I recommend implementing a hybrid approach that combines elements from both strategies:

### Core Implementation

1. **Initial Phase: Enhanced Usage Tracking**
   - Implement the metadata tracking from Strategy 1 (retrieval count, relevance scores, etc.)
   - Begin collecting this data without changing pruning logic
   - After sufficient data collection (e.g., 2-4 weeks), activate the utility-based pruning

2. **Second Phase: Goal-Relevance Integration**
   - Add goal relevance calculation from Strategy 2
   - Use a weighted formula that combines utility and relevance:
     - `Final_Score = (0.6 * Memory_Utility_Score) + (0.4 * Goal_Relevance_Score)`
   - Apply this for pruning decisions while maintaining age as a baseline factor

3. **Third Phase: Semantic Clustering & Consolidation**
   - Implement periodic semantic clustering (less frequently than regular pruning)
   - Focus on identifying and consolidating redundant memories
   - Preserve both high-utility and highly goal-relevant memories

### Technical Implementation

```python
# Enhanced memory metadata when storing memories
memory_metadata = {
    # Existing fields
    "agent_id": agent_id,
    "step": step,
    "event_type": event_type,
    "memory_type": memory_type,
    
    # New tracking fields 
    "retrieval_count": 0,
    "last_retrieved": None,
    "relevance_scores": [],
    "memory_utility_score": 0.0,
    "goal_relevance_score": calculate_goal_relevance(content, agent.goals),
}

# Update metadata during retrieval
def update_memory_metadata(memory_id, retrieval_relevance):
    # Get the current memory metadata
    result = collection.get(ids=[memory_id])
    metadata = result["metadatas"][0]
    
    # Update tracking fields
    metadata["retrieval_count"] += 1
    metadata["last_retrieved"] = datetime.utcnow().isoformat()
    metadata["relevance_scores"].append(retrieval_relevance)
    
    # Recalculate utility score
    rfs = math.log(1 + metadata["retrieval_count"])
    rs = sum(metadata["relevance_scores"]) / len(metadata["relevance_scores"])
    recs = 1.0 / (1.0 + days_since(metadata["last_retrieved"]))
    metadata["memory_utility_score"] = (0.4 * rfs) + (0.4 * rs) + (0.2 * recs)
    
    # Update in the database
    collection.update(ids=[memory_id], metadatas=[metadata])

# Enhanced pruning logic
def get_memories_to_prune(agent_id, memory_type, target_count):
    # Get all memories for this agent and type
    memories = collection.get(
        where={"agent_id": agent_id, "memory_type": memory_type},
        include=["metadatas", "embeddings"]
    )
    
    # Calculate final scores (combining utility and goal relevance)
    for i, metadata in enumerate(memories["metadatas"]):
        metadata["final_score"] = (
            (0.6 * metadata.get("memory_utility_score", 0.0)) + 
            (0.4 * metadata.get("goal_relevance_score", 0.0))
        )
    
    # Sort by final score (ascending)
    scored_memories = list(zip(memories["ids"], memories["metadatas"]))
    scored_memories.sort(key=lambda x: x[1]["final_score"])
    
    # Determine pruning candidates (lowest scoring memories)
    excess_count = len(scored_memories) - target_count
    if excess_count <= 0:
        return []
    
    # Get lowest-scored memories, but respect protection rules
    pruning_candidates = []
    for memory_id, metadata in scored_memories:
        # Skip if in cooldown period
        if (metadata.get("last_retrieved") and 
            days_since(metadata["last_retrieved"]) < COOLDOWN_DAYS):
            continue
            
        # Skip if exceptionally relevant
        if metadata.get("relevance_scores") and max(metadata["relevance_scores"]) > 0.9:
            continue
            
        pruning_candidates.append(memory_id)
        if len(pruning_candidates) >= excess_count:
            break
    
    return pruning_candidates
```

## Potential for DSPy Integration

Several aspects of this advanced pruning strategy could benefit from DSPy:

1. **Goal Relevance Assessment**
   - Train a DSPy module to evaluate how relevant a memory is to current agent goals
   - Use few-shot learning to teach the model to recognize goal-aligned content
   - Could be more nuanced than pure embedding similarity

2. **Consolidation Generation**
   - Use DSPy to generate high-quality consolidated memories from clusters of similar memories
   - Focus on preserving unique information while removing redundancy
   - Similar to the L1/L2 summarization but applied horizontally across similar memories

3. **Importance Classification**
   - Create a DSPy module that predicts which memories are likely to be important in the future
   - Could be trained on patterns of which types of memories proved valuable in past interactions

## Next Steps

1. **Data Collection Phase**
   - Modify `vector_store.py` to begin tracking retrieval statistics
   - Create a new class `MemoryTrackingManager` to manage the enhanced metadata
   - Add retrieval tracking to the RAG pipeline

2. **Analysis Scripts**
   - Develop tools to analyze memory usage patterns from collected data
   - Visualize which memories are most frequently retrieved and why
   - Determine optimal thresholds for pruning based on real usage data

3. **Prototype Implementation**
   - Implement the hybrid utility-based approach as an alternative pruning method
   - Create configuration options to enable/disable advanced pruning
   - Run comparative tests against the current age-based pruning

4. **DSPy Integration Research**
   - Develop experimental DSPy modules for relevance assessment
   - Test memory consolidation using DSPy on similar memory clusters
   - Measure quality improvements compared to heuristic-based approaches

## Implementation Progress

### Phase 1: Enhanced Usage Tracking (Completed)

The first phase of the advanced memory pruning strategy has been implemented. This phase focuses on tracking usage statistics for each memory event to inform future pruning decisions.

#### Implementation Details:

1. **Enhanced Metadata Tracking**
   - Added and initialized metadata fields in the ChromaVectorStoreManager:
     - `retrieval_count`: Tracks how many times a memory has been retrieved
     - `last_retrieved_timestamp`: Records when a memory was last accessed  
     - `accumulated_relevance_score`: Stores the sum of relevance scores from each retrieval
     - `retrieval_relevance_count`: Counts how many times a relevance score was added

2. **Memory Retrieval Updates**
   - The `_update_memory_usage_stats` method now updates these metrics whenever memories are retrieved 
   - Usage statistics are persisted in ChromaDB for later analysis and pruning decisions

3. **Memory Utility Score (MUS) Formula Implementation**
   - Demonstration of how the MUS formula can be calculated and used for pruning:
   ```
   MUS = (0.4 * RFS) + (0.4 * RS) + (0.2 * RecS)
   ```
   Where:
   - RFS (Retrieval Frequency Score): log(1 + retrieval_count)
   - RS (Relevance Score): accumulated_relevance_score / retrieval_relevance_count
   - RecS (Recency Score): 1.0 / (1.0 + days_since_last_retrieval)

4. **Testing**
   - Added comprehensive tests to verify the memory usage tracking functionality
   - Created demonstration of how the MUS can be used for pruning decisions

#### Next Steps:

Phase 2 will build on these usage statistics to implement the actual pruning mechanisms based on the calculated Memory Utility Score.

## Final Implementation Details (as of v1.2)

- **MUS Calculation:**
  - MUS = (0.4 × Retrieval Frequency Score) + (0.4 × Relevance Score) + (0.2 × Recency Score)
  - RFS: log(1 + retrieval_count)
  - RS: Average relevance of retrievals
  - RecS: 1 / (1 + days_since_last_accessed)
- **L1 Pruning:**
  - Age-based: L1s pruned after consolidation and delay
  - MUS-based: L1s with MUS < 0.2 are pruned
- **L2 Pruning:**
  - Age-based: L2s older than 30 days pruned
  - MUS-based: L2s with MUS < 0.3 are pruned
- **Thresholds:**
  - L1 MUS: 0.2
  - L2 MUS: 0.3
  - Configuration: l1_low_l2_medium (see benchmarks/mus_threshold_tuning_report.md)

### Rationale

Thresholds were selected based on empirical tuning (see MUS threshold tuning report v1.2) to balance memory efficiency and RAG performance.

## Evolution from Proposal

- The MUS formula and weights were finalized as above.
- The l1_low_l2_medium configuration was selected for optimal tradeoff.
- Any proposal sections referencing alternative formulas, symmetric thresholds, or unused pruning strategies are superseded by the above.

## Conclusion

The proposed hybrid approach balances implementation complexity with memory management effectiveness. By first tracking usage patterns, then incorporating goal relevance, and finally adding semantic clustering, we can create a memory pruning system that better preserves valuable information regardless of age. This aligns with human memory systems, which retain information based on a combination of recency, frequency of access, emotional significance, and relevance to current goals.

The recommended implementation is designed to be incremental, allowing for data collection and analysis before committing to structural changes in the pruning logic. This approach minimizes risk while providing a path toward significantly more sophisticated memory management for Culture.ai agents. 