#!/usr/bin/env python
"""
Script to generate a CSV template for RAG assessment scoring.
"""

import csv
import os

# All scenario IDs
scenarios = [
    "baseline",
    "mus_very_low",
    "mus_low",
    "mus_medium_low",
    "mus_medium",
    "mus_medium_high",
    "mus_high",
    "mus_very_high",
    "l1_low_l2_medium",
    "l1_medium_l2_low",
    "l1_only",
    "l2_only",
    "age_based_only",
]

# Output path
output_file = "request_to_operator/rag_assessment_scores.csv"

# Ensure directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Setup CSV file
with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Scenario ID",
            "Agent ID",
            "Query Number",
            "Relevance (1-5)",
            "Coherence (1-5)",
            "Completeness (1-5)",
            "Accuracy (1-5)",
            "Average Score",
            "Comments",
        ]
    )

    # Create a row for each scenario, agent, and query combo
    for scenario in scenarios:
        for agent_id in range(1, 5):  # 4 agents
            for query_num in range(3):  # 3 queries
                writer.writerow(
                    [
                        scenario,
                        f"agent_{agent_id}",
                        query_num,
                        "",
                        "",
                        "",
                        "",
                        "=AVERAGE(D{row}:G{row})".format(row=file.tell()),
                        "",
                    ]
                )

print(f"CSV template created at: {output_file}")
