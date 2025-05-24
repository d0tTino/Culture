#!/usr/bin/env python
"""
Script to integrate operator-provided RAG assessment scores into the JSON files.

This script reads the CSV file with operator scores and updates the corresponding
RAG assessment JSON files for each scenario.
"""

import argparse
import csv
import json
import os


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Integrate RAG assessment scores")
    parser.add_argument(
        "--csv",
        type=str,
        default="request_to_operator/rag_assessment_scores.csv",
        help="Path to the CSV file with operator scores",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="benchmarks/tuning_results",
        help="Path to the tuning results directory",
    )
    return parser.parse_args()


def main():
    """Main function to integrate the scores."""
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found at {args.csv}")
        return 1

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found at {args.results_dir}")
        return 1

    # Load all scores from the CSV
    scores = []
    with open(args.csv, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Skip rows with missing scores
            if not all(
                row.get(field, "").strip()
                for field in [
                    "Relevance (1-5)",
                    "Coherence (1-5)",
                    "Completeness (1-5)",
                    "Accuracy (1-5)",
                ]
            ):
                continue

            # Validate row data
            try:
                relevance = float(row["Relevance (1-5)"])
                coherence = float(row["Coherence (1-5)"])
                completeness = float(row["Completeness (1-5)"])
                accuracy = float(row["Accuracy (1-5)"])

                # Check score ranges
                if not all(
                    1 <= score <= 5 for score in [relevance, coherence, completeness, accuracy]
                ):
                    print(f"Warning: Invalid score range in row: {row}")
                    continue

                # Calculate average if not provided
                avg_score = row.get("Average Score", "")
                if not avg_score:
                    avg_score = (relevance + coherence + completeness + accuracy) / 4
                else:
                    try:
                        avg_score = float(avg_score)
                    except ValueError:
                        avg_score = (relevance + coherence + completeness + accuracy) / 4

                scores.append(
                    {
                        "scenario_id": row["Scenario ID"],
                        "agent_id": row["Agent ID"],
                        "query_num": int(row["Query Number"]),
                        "relevance": relevance,
                        "coherence": coherence,
                        "completeness": completeness,
                        "accuracy": accuracy,
                        "average_score": avg_score,
                        "comments": row.get("Comments", ""),
                    }
                )
            except (ValueError, KeyError) as e:
                print(f"Warning: Error processing row: {row}. Error: {e}")
                continue

    if not scores:
        print("Error: No valid scores found in the CSV file")
        return 1

    print(f"Loaded {len(scores)} valid assessment scores")

    # Group scores by scenario
    scenario_scores = {}
    for score in scores:
        scenario_id = score["scenario_id"]
        if scenario_id not in scenario_scores:
            scenario_scores[scenario_id] = []
        scenario_scores[scenario_id].append(score)

    # Update the JSON files for each scenario
    for scenario_id, scenario_score_list in scenario_scores.items():
        scenario_dir = os.path.join(args.results_dir, scenario_id)
        rag_assessment_dir = os.path.join(scenario_dir, "rag_assessment")

        if not os.path.exists(rag_assessment_dir):
            print(f"Warning: RAG assessment directory not found for scenario {scenario_id}")
            continue

        # Find the JSON file
        json_file = os.path.join(rag_assessment_dir, f"{scenario_id}_rag_assessment.json")
        if not os.path.exists(json_file):
            print(f"Warning: RAG assessment JSON file not found for scenario {scenario_id}")
            continue

        # Load the JSON data
        try:
            with open(json_file) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file {json_file}")
            continue

        # Update the JSON data with the scores
        updated = False
        for score in scenario_score_list:
            agent_id = score["agent_id"]
            query_num = score["query_num"]

            if agent_id in data.get("agents", {}) and "queries" in data["agents"][agent_id]:
                queries = data["agents"][agent_id]["queries"]
                if 0 <= query_num < len(queries):
                    # Update the human evaluation score
                    queries[query_num]["human_evaluation_score"] = {
                        "relevance": score["relevance"],
                        "coherence": score["coherence"],
                        "completeness": score["completeness"],
                        "accuracy": score["accuracy"],
                        "average": score["average_score"],
                        "comments": score["comments"],
                    }
                    updated = True
                else:
                    print(
                        f"Warning: Query number {query_num} out of range for agent {agent_id} in scenario {scenario_id}"
                    )
            else:
                print(f"Warning: Agent {agent_id} not found in scenario {scenario_id}")

        if updated:
            # Add overall RAG metrics to data
            calculate_rag_metrics(data)

            # Save the updated JSON
            with open(json_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Updated RAG assessment JSON for scenario {scenario_id}")
        else:
            print(f"No updates made to scenario {scenario_id}")

    print("Integration complete")
    return 0


def calculate_rag_metrics(data):
    """Calculate overall RAG metrics for the scenario."""
    total_score = 0
    total_count = 0

    for agent_id, agent_data in data.get("agents", {}).items():
        agent_total = 0
        agent_count = 0

        for query in agent_data.get("queries", []):
            if "human_evaluation_score" in query and "average" in query["human_evaluation_score"]:
                score = query["human_evaluation_score"]["average"]
                agent_total += score
                agent_count += 1
                total_score += score
                total_count += 1

        if agent_count > 0:
            agent_data["average_rag_score"] = agent_total / agent_count

    if total_count > 0:
        # Add overall metrics to the scenario data
        if "rag_assessment" not in data:
            data["rag_assessment"] = {}

        data["rag_assessment"]["avg_score"] = total_score / total_count
        data["rag_assessment"]["query_count"] = total_count

        # Calculate success rate (scores >= 3.5 considered successful)
        success_count = 0
        for agent_id, agent_data in data.get("agents", {}).items():
            for query in agent_data.get("queries", []):
                if (
                    "human_evaluation_score" in query
                    and "average" in query["human_evaluation_score"]
                ):
                    if query["human_evaluation_score"]["average"] >= 3.5:
                        success_count += 1

        data["rag_assessment"]["success_rate"] = (
            success_count / total_count if total_count > 0 else 0
        )


if __name__ == "__main__":
    exit(main())
