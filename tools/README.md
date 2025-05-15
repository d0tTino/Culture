# Culture.AI Tools

This directory contains various utility tools for the Culture.AI project.

## Memory Visualization Tool

The Memory Visualization Tool (`visualize_agent_memory.py`) allows you to visualize an agent's hierarchical memory structure from ChromaDB, showing the relationship between L2 (chapter) summaries and their constituent L1 (session) summaries.

### Features

- Displays hierarchical relationships between L2 summaries and their constituent L1 summaries
- Shows key metadata for each memory, including:
  - Creation timestamps and step numbers
  - Memory Utility Score (MUS)
  - Retrieval counts and last retrieval timestamps
  - Content (full or shortened)
- Supports two output formats:
  - Text-based tree view (default)
  - HTML report with enhanced visualization

### Usage

```bash
python tools/visualize_agent_memory.py --agent_id <agent_id> [--output_format text|html] [--max_length 200] [--chroma_dir ./chroma_db]
```

#### Parameters

- `--agent_id` (required): ID of the agent whose memory you want to visualize
- `--output_format` (optional): Output format, either "text" (default) or "html"
- `--max_length` (optional): Maximum length for displayed memory content (default: 200)
- `--chroma_dir` (optional): Path to ChromaDB directory (default: ./chroma_db)

### Output

#### Text Format

The text output uses indentation to represent the hierarchy:

```
L2_Summary_XYZ (Step 50, Period 40-50, MUS: 0.75, Retrievals: 5, Consolidates 3 L1s)
  Overall insight about project Alpha...
    L1_Summary_ABC (Step 42, MUS: 0.8, Retrievals: 5, Last: 2025-05-01 13:45:27)
      Agent 1 proposed...
    L1_Summary_DEF (Step 45, MUS: 0.6, Retrievals: 2, Last: 2025-05-05 10:22:15)
      Agent 2 responded...
    ...
```

#### HTML Format

The HTML output provides a more visually appealing representation with:
- Collapsible sections
- Color-coded MUS scores (green for high, orange for medium, red for low)
- Better formatting for readability
- Statistics and summary information

### Example

```bash
# Generate a text visualization for agent "agent_1"
python tools/visualize_agent_memory.py --agent_id agent_1

# Generate an HTML report with a content limit of 300 characters
python tools/visualize_agent_memory.py --agent_id agent_1 --output_format html --max_length 300
```

### Dependencies

The tool depends on:
- The ChromaVectorStoreManager class from src/infra/memory/vector_store.py
- Python standard libraries (argparse, logging, etc.)
- Access to a valid ChromaDB instance with agent memories

### Design Choices

- **Memory Relationship Determination**: The tool determines which L1 summaries belong to which L2 summaries by examining the `consolidation_period` field in L2 summaries and matching it with L1 summary step numbers.
- **MUS Calculation**: The Memory Utility Score is calculated using the same formula as in the advanced memory pruning system:
  ```
  MUS = (0.4 * RFS) + (0.4 * RS) + (0.2 * RecS)
  ```
  Where:
  - RFS: Retrieval Frequency Score = log(1 + retrieval_count)
  - RS: Relevance Score = accumulated_relevance_score / retrieval_relevance_count
  - RecS: Recency Score = 1.0 / (1.0 + days_since_last_accessed)
- **Output Format**: Text format was chosen as the default for simplicity and ease of use, with HTML as an option for more detailed visualization. 