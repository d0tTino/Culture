#!/usr/bin/env python
"""
Memory Visualization Tool

This script visualizes an agent's hierarchical memory structure from ChromaDB,
showing L2 summaries (chapter level) and their constituent L1 summaries (session level).

Usage:
    python tools/visualize_agent_memory.py --agent_id <agent_id> \
        [--output_format text|html] [--max_length 200]
"""

import argparse
import logging
import math
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

try:
    from src.agents.memory.vector_store import ChromaVectorStoreManager
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the project root.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("memory_visualizer")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize agent memory structure from ChromaDB")
    parser.add_argument(
        "--agent_id", required=True, help="ID of the agent to visualize memories for"
    )
    parser.add_argument(
        "--output_format",
        choices=["text", "html"],
        default="text",
        help="Output format (text tree or HTML)",
    )
    parser.add_argument(
        "--max_length", type=int, default=200, help="Maximum length for displayed memory content"
    )
    parser.add_argument("--chroma_dir", default="./chroma_db", help="Path to ChromaDB directory")

    return parser.parse_args()


def shorten_text(text: str, max_length: int) -> str:
    """Shorten text to max_length, adding ellipsis if needed."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp to a more readable form."""
    if not timestamp_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return timestamp_str


def calculate_mus(metadata: dict[str, Any]) -> float:
    """
    Calculate Memory Utility Score (MUS) for a memory.

    MUS = (0.4 * RFS) + (0.4 * RS) + (0.2 * RecS)
    RFS: Retrieval Frequency Score = log(1 + retrieval_count)
    RS:  Relevance Score = accumulated_relevance_score / retrieval_relevance_count
    RecS: Recency Score = 1.0 / (1.0 + days_since_last_accessed)
    """
    retrieval_count = metadata.get("retrieval_count", 0)
    accumulated_relevance_score = metadata.get("accumulated_relevance_score", 0.0)
    retrieval_relevance_count = metadata.get("retrieval_relevance_count", 0)
    last_retrieved = metadata.get("last_retrieved_timestamp", "")

    # RFS - Retrieval Frequency Score
    rfs = math.log(1 + retrieval_count)

    # RS - Relevance Score
    rs = (
        (accumulated_relevance_score / retrieval_relevance_count)
        if retrieval_relevance_count > 0
        else 0.0
    )

    # RecS - Recency Score
    recs = 0.0
    if last_retrieved:
        try:
            last_dt = datetime.fromisoformat(last_retrieved)
            days_since = (datetime.utcnow() - last_dt).total_seconds() / (24 * 3600)
            days_since = max(0, days_since)
            recs = 1.0 / (1.0 + days_since)
        except Exception as e:
            logger.warning(f"Invalid last_retrieved_timestamp format: {last_retrieved} ({e})")
            recs = 0.0

    # MUS - Final Memory Utility Score
    mus = (0.4 * rfs) + (0.4 * rs) + (0.2 * recs)
    return mus


def get_all_l2_summaries(
    vector_store: ChromaVectorStoreManager, agent_id: str
) -> list[dict[str, Any]]:
    """Get all L2 (chapter) summaries for the agent."""
    try:
        # Query for all L2 summaries
        l2_query = f"all chapter summaries for agent {agent_id}"
        l2_where = {
            "$and": [{"agent_id": {"$eq": agent_id}}, {"memory_type": {"$eq": "chapter_summary"}}]
        }

        results = vector_store.collection.query(
            query_texts=[l2_query],
            where=l2_where,
            n_results=100,  # Large number to get all
            include=["metadatas", "documents", "ids"],
        )

        l2_summaries = []
        if results and "documents" in results and results["documents"]:
            documents = results["documents"][0]
            metadatas = (
                results["metadatas"][0] if "metadatas" in results else [{}] * len(documents)
            )
            ids = results["ids"][0] if "ids" in results else [""] * len(documents)

            for i, doc in enumerate(documents):
                if i < len(metadatas):
                    entry = metadatas[i].copy()
                    entry["content"] = doc
                    entry["id"] = ids[i] if i < len(ids) else ""

                    # Calculate MUS for this memory
                    entry["mus"] = calculate_mus(entry)

                    l2_summaries.append(entry)

        # Sort by step number
        l2_summaries.sort(key=lambda x: x.get("step", 0))

        logger.info(f"Retrieved {len(l2_summaries)} L2 summaries for agent {agent_id}")
        return l2_summaries

    except Exception as e:
        logger.error(f"Error retrieving L2 summaries: {e}")
        return []


def get_all_l1_summaries(
    vector_store: ChromaVectorStoreManager, agent_id: str
) -> list[dict[str, Any]]:
    """Get all L1 (consolidated) summaries for the agent."""
    try:
        # Query for all L1 summaries
        l1_query = f"all consolidated summaries for agent {agent_id}"
        l1_where = {
            "$and": [
                {"agent_id": {"$eq": agent_id}},
                {"memory_type": {"$eq": "consolidated_summary"}},
            ]
        }

        results = vector_store.collection.query(
            query_texts=[l1_query],
            where=l1_where,
            n_results=500,  # Large number to get all
            include=["metadatas", "documents", "ids"],
        )

        l1_summaries = []
        if results and "documents" in results and results["documents"]:
            documents = results["documents"][0]
            metadatas = (
                results["metadatas"][0] if "metadatas" in results else [{}] * len(documents)
            )
            ids = results["ids"][0] if "ids" in results else [""] * len(documents)

            for i, doc in enumerate(documents):
                if i < len(metadatas):
                    entry = metadatas[i].copy()
                    entry["content"] = doc
                    entry["id"] = ids[i] if i < len(ids) else ""

                    # Calculate MUS for this memory
                    entry["mus"] = calculate_mus(entry)

                    l1_summaries.append(entry)

        # Sort by step number
        l1_summaries.sort(key=lambda x: x.get("step", 0))

        logger.info(f"Retrieved {len(l1_summaries)} L1 summaries for agent {agent_id}")
        return l1_summaries

    except Exception as e:
        logger.error(f"Error retrieving L1 summaries: {e}")
        return []


def determine_l1_l2_relationships(
    l1_summaries: list[dict[str, Any]], l2_summaries: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """
    Map each L2 summary to its constituent L1 summaries.

    Returns a dictionary with L2 IDs as keys and lists of L1 summaries as values.
    Also returns a set of L1 IDs that haven't been assigned to any L2.
    """
    # Sort L2 summaries by step number (ascending)
    sorted_l2s = sorted(l2_summaries, key=lambda x: x.get("step", 0))

    # Create a mapping of L2 IDs to their step ranges
    l2_ranges = []
    for l2 in sorted_l2s:
        consolidation_period = l2.get("consolidation_period", "")
        if consolidation_period:
            try:
                start_step, end_step = map(int, consolidation_period.split("-"))
                l2_ranges.append((l2, start_step, end_step))
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid consolidation_period format for L2: {consolidation_period}"
                )

    # Map L1 summaries to L2 summaries based on step ranges
    l2_to_l1s = {l2["id"]: [] for l2 in l2_summaries}
    assigned_l1_ids = set()

    for l1 in l1_summaries:
        l1_step = l1.get("step", 0)
        assigned = False

        # Find the L2 that covers this L1's step
        for l2, start_step, end_step in l2_ranges:
            if start_step <= l1_step <= end_step:
                l2_to_l1s[l2["id"]].append(l1)
                assigned_l1_ids.add(l1["id"])
                assigned = True
                break

        if not assigned:
            logger.debug(f"L1 summary at step {l1_step} not assigned to any L2")

    # Identify unassigned L1 summaries
    unassigned_l1s = [l1 for l1 in l1_summaries if l1["id"] not in assigned_l1_ids]

    return l2_to_l1s, unassigned_l1s


def generate_text_visualization(
    l2_summaries: list[dict[str, Any]],
    l2_to_l1s: dict[str, list[dict[str, Any]]],
    unassigned_l1s: list[dict[str, Any]],
    max_length: int,
) -> str:
    """Generate a text-based tree visualization of the memory structure."""
    output = []

    # Header
    output.append("=" * 80)
    output.append("AGENT MEMORY VISUALIZATION (TEXT-BASED TREE)")
    output.append("=" * 80)
    output.append("")

    # L2 Summaries with their L1s
    output.append("HIERARCHICAL MEMORY STRUCTURE:")
    output.append("-" * 50)

    # Sort L2 summaries by step
    sorted_l2s = sorted(l2_summaries, key=lambda x: x.get("step", 0))

    for l2 in sorted_l2s:
        # Get L1 summaries for this L2
        l1s = l2_to_l1s.get(l2["id"], [])

        # L2 Header
        l2_id = l2["id"][-8:] if l2.get("id") else "unknown"
        step = l2.get("step", "unknown")
        consolidation_period = l2.get("consolidation_period", "unknown")
        mus = l2.get("mus", 0.0)
        retrieval_count = l2.get("retrieval_count", 0)
        content = shorten_text(l2.get("content", ""), max_length)

        output.append(
            f"L2_Summary_{l2_id} (Step {step}, Period {consolidation_period}, MUS: {mus:.2f}, "
            f"Retrievals: {retrieval_count}, Consolidates {len(l1s)} L1s)"
        )

        # Wrap and indent the content
        wrapped_content = textwrap.fill(
            content, width=76, initial_indent="  ", subsequent_indent="  "
        )
        output.append(wrapped_content)

        # L1 Summaries under this L2
        for l1 in sorted(l1s, key=lambda x: x.get("step", 0)):
            l1_id = l1["id"][-8:] if l1.get("id") else "unknown"
            step = l1.get("step", "unknown")
            mus = l1.get("mus", 0.0)
            retrieval_count = l1.get("retrieval_count", 0)
            last_retrieved = format_timestamp(l1.get("last_retrieved_timestamp", ""))
            content = shorten_text(l1.get("content", ""), max_length)

            output.append(
                f"    L1_Summary_{l1_id} (Step {step}, MUS: {mus:.2f}, "
                f"Retrievals: {retrieval_count}, Last: {last_retrieved})"
            )

            # Wrap and indent the content
            wrapped_content = textwrap.fill(
                content, width=72, initial_indent="      ", subsequent_indent="      "
            )
            output.append(wrapped_content)

        output.append("")  # Add space between L2 blocks

    # Unassigned L1 Summaries
    if unassigned_l1s:
        output.append("UNASSIGNED L1 SUMMARIES (not part of any L2):")
        output.append("-" * 50)

        for l1 in sorted(unassigned_l1s, key=lambda x: x.get("step", 0)):
            l1_id = l1["id"][-8:] if l1.get("id") else "unknown"
            step = l1.get("step", "unknown")
            mus = l1.get("mus", 0.0)
            retrieval_count = l1.get("retrieval_count", 0)
            last_retrieved = format_timestamp(l1.get("last_retrieved_timestamp", ""))
            content = shorten_text(l1.get("content", ""), max_length)

            output.append(
                f"L1_Summary_{l1_id} (Step {step}, MUS: {mus:.2f}, Retrievals: {retrieval_count}, "
                f"Last: {last_retrieved})"
            )

            # Wrap and indent the content
            wrapped_content = textwrap.fill(
                content, width=76, initial_indent="  ", subsequent_indent="  "
            )
            output.append(wrapped_content)
            output.append("")  # Add space between L1 blocks

    # Statistics
    output.append("=" * 80)
    output.append("STATISTICS:")
    output.append(f"  Total L2 Summaries: {len(l2_summaries)}")
    assigned_l1_count = sum(len(l1s) for l1s in l2_to_l1s.values())
    output.append(f"  Total L1 Summaries: {len(unassigned_l1s) + assigned_l1_count}")
    output.append(f"    - Assigned to L2s: {assigned_l1_count}")
    output.append(f"    - Unassigned: {len(unassigned_l1s)}")
    output.append("=" * 80)

    return "\n".join(output)


def generate_html_visualization(
    agent_id: str,
    l2_summaries: list[dict[str, Any]],
    l2_to_l1s: dict[str, list[dict[str, Any]]],
    unassigned_l1s: list[dict[str, Any]],
    max_length: int,
) -> str:
    """Generate an HTML visualization of the memory structure."""
    html = []

    # HTML header
    html.append("<!DOCTYPE html>")
    html.append("<html lang='en'>")
    html.append("<head>")
    html.append("  <meta charset='UTF-8'>")
    html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    html.append(f"  <title>Memory Visualization for Agent {agent_id}</title>")
    html.append("  <style>")
    html.append("    body { font-family: Arial, sans-serif; margin: 20px; }")
    html.append("    h1, h2, h3 { color: #333; }")
    html.append(
        "    .l2-summary { border: 1px solid #ddd; margin-bottom: 20px; padding: 10px; "
        "border-radius: 4px; }"
    )
    html.append(
        "    .l2-header { background-color: #f5f5f5; padding: 10px; margin-bottom: 10px; "
        "border-radius: 4px; }"
    )
    html.append(
        "    .l1-summary { border: 1px solid #eee; margin: 10px 0 10px 20px; padding: 10px; "
        "border-radius: 4px; }"
    )
    html.append(
        "    .l1-header { background-color: #f9f9f9; padding: 8px; margin-bottom: 8px; "
        "border-radius: 4px; }"
    )
    html.append("    .content { white-space: pre-wrap; margin-left: 10px; }")
    html.append("    .unassigned { border-left: 5px solid #ffcccb; }")
    html.append(
        "    .stats { background-color: #efffef; padding: 15px; margin-top: 20px; "
        "border-radius: 4px; }"
    )
    html.append("    .metadata { color: #666; font-size: 0.9em; }")
    html.append("    .mus-high { color: green; font-weight: bold; }")
    html.append("    .mus-medium { color: orange; }")
    html.append("    .mus-low { color: red; }")
    html.append("  </style>")
    html.append("</head>")
    html.append("<body>")

    # Page header
    html.append(f"  <h1>Memory Visualization for Agent {agent_id}</h1>")
    html.append(
        "  <p>Hierarchical structure showing L2 (chapter) summaries and their constituent "
        "L1 (session) summaries.</p>"
    )

    # L2 Summaries section
    html.append("  <h2>Hierarchical Memory Structure</h2>")

    # Sort L2 summaries by step
    sorted_l2s = sorted(l2_summaries, key=lambda x: x.get("step", 0))

    for l2 in sorted_l2s:
        # Get L1 summaries for this L2
        l1s = l2_to_l1s.get(l2["id"], [])

        # L2 Summary section
        html.append("  <div class='l2-summary'>")

        # L2 Header
        l2_id = l2["id"][-8:] if l2.get("id") else "unknown"
        step = l2.get("step", "unknown")
        consolidation_period = l2.get("consolidation_period", "unknown")
        mus = l2.get("mus", 0.0)
        mus_class = "mus-high" if mus >= 0.7 else "mus-medium" if mus >= 0.3 else "mus-low"
        retrieval_count = l2.get("retrieval_count", 0)

        html.append("    <div class='l2-header'>")
        html.append(f"      <h3>L2 Summary {l2_id}</h3>")
        html.append("      <div class='metadata'>")
        html.append(f"        <p>Step: {step} | Consolidation Period: {consolidation_period} | ")
        html.append(f"        <span class='{mus_class}'>MUS: {mus:.2f}</span> | ")
        html.append(
            f"        Retrievals: {retrieval_count} | Consolidates: {len(l1s)} L1 summaries</p>"
        )
        html.append("      </div>")
        html.append("    </div>")

        # L2 Content
        content = shorten_text(l2.get("content", ""), max_length)
        html.append(f"    <div class='content'>{content}</div>")

        # L1 Summaries under this L2
        if l1s:
            html.append("    <h4>Constituent L1 Summaries:</h4>")

            for l1 in sorted(l1s, key=lambda x: x.get("step", 0)):
                l1_id = l1["id"][-8:] if l1.get("id") else "unknown"
                step = l1.get("step", "unknown")
                mus = l1.get("mus", 0.0)
                mus_class = "mus-high" if mus >= 0.7 else "mus-medium" if mus >= 0.3 else "mus-low"
                retrieval_count = l1.get("retrieval_count", 0)
                last_retrieved = format_timestamp(l1.get("last_retrieved_timestamp", ""))

                html.append("    <div class='l1-summary'>")
                html.append("      <div class='l1-header'>")
                html.append(f"        <h4>L1 Summary {l1_id}</h4>")
                html.append("        <div class='metadata'>")
                html.append(f"<p>Step: {step} | <span class='{mus_class}'>MUS: {mus:.2f}</span> |")
                html.append(
                    f"          Retrievals: {retrieval_count} | Last Retrieved: {last_retrieved}"
                )
                html.append("</p>")
                html.append("        </div>")
                html.append("      </div>")

                content = shorten_text(l1.get("content", ""), max_length)
                html.append(f"      <div class='content'>{content}</div>")
                html.append("    </div>")

        html.append("  </div>")

    # Unassigned L1 Summaries
    if unassigned_l1s:
        html.append("  <h2>Unassigned L1 Summaries</h2>")
        html.append("  <p>These L1 summaries are not associated with any L2 summary yet.</p>")

        for l1 in sorted(unassigned_l1s, key=lambda x: x.get("step", 0)):
            l1_id = l1["id"][-8:] if l1.get("id") else "unknown"
            step = l1.get("step", "unknown")
            mus = l1.get("mus", 0.0)
            mus_class = "mus-high" if mus >= 0.7 else "mus-medium" if mus >= 0.3 else "mus-low"
            retrieval_count = l1.get("retrieval_count", 0)
            last_retrieved = format_timestamp(l1.get("last_retrieved_timestamp", ""))

            html.append("  <div class='l1-summary unassigned'>")
            html.append("    <div class='l1-header'>")
            html.append(f"      <h3>L1 Summary {l1_id}</h3>")
            html.append("      <div class='metadata'>")
            html.append(f"<p>Step: {step} | <span class='{mus_class}'>MUS: {mus:.2f}</span> |")
            html.append(
                f"        Retrievals: {retrieval_count} | Last Retrieved: {last_retrieved}</p>"
            )
            html.append("      </div>")
            html.append("    </div>")

            content = shorten_text(l1.get("content", ""), max_length)
            html.append(f"    <div class='content'>{content}</div>")
            html.append("  </div>")

    # Statistics
    html.append("  <div class='stats'>")
    html.append("    <h2>Statistics</h2>")
    html.append(f"    <p>Total L2 Summaries: {len(l2_summaries)}</p>")
    assigned_l1_count = sum(len(l1s) for l1s in l2_to_l1s.values())
    html.append(f"    <p>Total L1 Summaries: {len(unassigned_l1s) + assigned_l1_count}</p>")
    html.append("    <ul>")
    html.append(f"      <li>Assigned to L2s: {assigned_l1_count}</li>")
    html.append(f"      <li>Unassigned: {len(unassigned_l1s)}</li>")
    html.append("    </ul>")
    html.append("  </div>")

    # HTML footer
    html.append("</body>")
    html.append("</html>")

    return "\n".join(html)


def main() -> int:
    """Main function to run the visualization."""
    args = parse_args()
    agent_id = args.agent_id
    output_format = args.output_format
    max_length = args.max_length
    chroma_dir = args.chroma_dir

    logger.info(f"Visualizing memory for agent {agent_id}")
    logger.info(f"Output format: {output_format}")
    logger.info(f"Max content length: {max_length}")
    logger.info(f"ChromaDB directory: {chroma_dir}")

    # Initialize ChromaDB vector store
    try:
        vector_store = ChromaVectorStoreManager(persist_directory=chroma_dir)
        logger.info("Successfully connected to ChromaDB")
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return 1

    # Get all L2 and L1 summaries for the agent
    l2_summaries = get_all_l2_summaries(vector_store, agent_id)
    l1_summaries = get_all_l1_summaries(vector_store, agent_id)

    if not l2_summaries and not l1_summaries:
        logger.error(f"No memories found for agent {agent_id}")
        return 1

    # Determine relationships between L1 and L2 summaries
    l2_to_l1s, unassigned_l1s = determine_l1_l2_relationships(l1_summaries, l2_summaries)

    # Generate visualization based on chosen format
    if output_format == "text":
        visualization = generate_text_visualization(
            l2_summaries, l2_to_l1s, unassigned_l1s, max_length
        )
        print(visualization)

        # Optionally save to file
        output_file = f"agent_{agent_id}_memory_visualization.txt"
        with open(output_file, "w") as f:
            f.write(visualization)
        logger.info(f"Text visualization saved to {output_file}")

    elif output_format == "html":
        visualization = generate_html_visualization(
            agent_id, l2_summaries, l2_to_l1s, unassigned_l1s, max_length
        )

        # Save HTML to file
        output_file = f"agent_{agent_id}_memory_visualization.html"
        with open(output_file, "w") as f:
            f.write(visualization)
        logger.info(f"HTML visualization saved to {output_file}")
        print(f"HTML visualization saved to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
