#!/usr/bin/env python
"""
Script to update the comparative data and visualizations after operator assessment.

This script:
1. Reads the updated RAG assessment JSON files for all scenarios
2. Updates the comparative_data.json file with RAG performance metrics
3. Regenerates the rag_scores.png visualization
4. Updates the mus_threshold_tuning_report.md with RAG performance analysis
"""

import argparse
import json
import os

import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Update comparative data and visualizations")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="benchmarks/tuning_results",
        help="Path to the tuning results directory",
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="benchmarks/mus_threshold_tuning_report.md",
        help="Path to the MUS threshold tuning report file",
    )
    return parser.parse_args()


def main():
    """Main function to update the data and visualizations."""
    args = parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found at {args.results_dir}")
        return 1

    # Load the comparative data
    comparative_data_path = os.path.join(args.results_dir, "comparative_data.json")
    if not os.path.exists(comparative_data_path):
        print(f"Error: Comparative data file not found at {comparative_data_path}")
        return 1

    try:
        with open(comparative_data_path) as f:
            comparative_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {comparative_data_path}")
        return 1

    # Get the list of scenarios
    scenarios = list(comparative_data.get("scenarios", {}).keys())
    if not scenarios:
        print("Error: No scenarios found in comparative data")
        return 1

    # Process each scenario to extract RAG performance metrics
    for scenario_id in scenarios:
        rag_assessment_path = os.path.join(
            args.results_dir, scenario_id, "rag_assessment", f"{scenario_id}_rag_assessment.json"
        )

        if not os.path.exists(rag_assessment_path):
            print(f"Warning: RAG assessment file not found for scenario {scenario_id}")
            continue

        try:
            with open(rag_assessment_path) as f:
                rag_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file {rag_assessment_path}")
            continue

        # Extract RAG performance metrics
        rag_metrics = rag_data.get("rag_assessment", {})
        avg_score = rag_metrics.get("avg_score", 0.0)
        query_count = rag_metrics.get("query_count", 0)
        success_rate = rag_metrics.get("success_rate", 0.0)

        # Update the comparative data
        if scenario_id in comparative_data.get("scenarios", {}):
            comparative_data["scenarios"][scenario_id]["rag_assessment"] = {
                "avg_score": avg_score,
                "query_count": query_count,
                "success_rate": success_rate,
            }

            # Calculate overall score based on memory efficiency and RAG performance
            memory_efficiency = comparative_data["scenarios"][scenario_id].get(
                "memory_efficiency_score", 0.0
            )

            # Weight memory efficiency and RAG performance equally
            overall_score = (0.5 * memory_efficiency) + (0.5 * (avg_score / 5.0))
            comparative_data["scenarios"][scenario_id]["overall_score"] = overall_score

    # Determine best configurations
    best_memory_efficiency_id = max(
        scenarios,
        key=lambda sid: comparative_data["scenarios"][sid].get("memory_efficiency_score", 0.0),
    )

    best_rag_performance_id = max(
        scenarios,
        key=lambda sid: comparative_data["scenarios"][sid]
        .get("rag_assessment", {})
        .get("avg_score", 0.0),
    )

    best_overall_id = max(
        scenarios, key=lambda sid: comparative_data["scenarios"][sid].get("overall_score", 0.0)
    )

    # Update summary in comparative data
    comparative_data["summary"] = {
        "best_memory_efficiency": best_memory_efficiency_id,
        "best_rag_performance": best_rag_performance_id,
        "recommended_configuration": best_overall_id,
    }

    # Save the updated comparative data
    with open(comparative_data_path, "w") as f:
        json.dump(comparative_data, f, indent=2)
    print(f"Updated comparative data file: {comparative_data_path}")

    # Generate RAG scores plot
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 8))

        scenario_ids = scenarios
        rag_scores = [
            comparative_data["scenarios"][sid]["rag_assessment"].get("avg_score", 0.0) / 5.0
            for sid in scenario_ids
        ]

        x = np.arange(len(scenario_ids))

        plt.bar(x, rag_scores)

        plt.xlabel("Scenario")
        plt.ylabel("RAG Performance Score (normalized 0-1)")
        plt.title("RAG Performance by Scenario")
        plt.xticks(x, scenario_ids, rotation=45, ha="right")
        plt.ylim(0, 1.0)
        plt.tight_layout()

        rag_path = os.path.join(args.results_dir, "rag_scores.png")
        plt.savefig(rag_path)
        print(f"Generated RAG scores plot: {rag_path}")

        # Generate combined performance plot
        plt.figure(figsize=(14, 8))

        memory_scores = [
            comparative_data["scenarios"][sid].get("memory_efficiency_score", 0.0)
            for sid in scenario_ids
        ]
        overall_scores = [
            comparative_data["scenarios"][sid].get("overall_score", 0.0) for sid in scenario_ids
        ]

        width = 0.3

        plt.bar(x - width, memory_scores, width, label="Memory Efficiency")
        plt.bar(x, rag_scores, width, label="RAG Performance")
        plt.bar(x + width, overall_scores, width, label="Overall Score")

        plt.xlabel("Scenario")
        plt.ylabel("Score (0-1)")
        plt.title("Memory Efficiency vs. RAG Performance vs. Overall Score")
        plt.xticks(x, scenario_ids, rotation=45, ha="right")
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()

        combined_path = os.path.join(args.results_dir, "combined_scores.png")
        plt.savefig(combined_path)
        print(f"Generated combined scores plot: {combined_path}")

    except ImportError:
        print("Warning: Matplotlib not available. Skipping plot generation.")
    except Exception as e:
        print(f"Error generating plots: {e}")

    # Update the report
    update_report(args.report_file, comparative_data)

    print("Update complete")
    return 0


def update_report(report_file, comparative_data):
    """Update the MUS threshold tuning report with RAG performance analysis."""
    if not os.path.exists(report_file):
        print(f"Error: Report file not found at {report_file}")
        return

    try:
        with open(report_file) as f:
            report_lines = f.readlines()
    except Exception as e:
        print(f"Error reading report file: {e}")
        return

    # Find the RAG Assessment section
    rag_section_index = -1
    for i, line in enumerate(report_lines):
        if line.strip() == "### RAG Assessment":
            rag_section_index = i
            break

    if rag_section_index == -1:
        print("Error: RAG Assessment section not found in report")
        return

    # Create new RAG assessment section content
    rag_section = [
        "### RAG Assessment\n",
        "\n",
        "| Scenario ID | RAG Avg Score (1-5) | Success Rate | Query Count |\n",
        "|------------|---------------------|--------------|-------------|\n",
    ]

    for scenario_id, scenario_data in comparative_data["scenarios"].items():
        rag_metrics = scenario_data.get("rag_assessment", {})
        avg_score = rag_metrics.get("avg_score", 0.0)
        success_rate = rag_metrics.get("success_rate", 0.0)
        query_count = rag_metrics.get("query_count", 0)

        rag_section.append(
            f"| {scenario_id} | {avg_score:.2f} | {success_rate:.2f} | {query_count} |\n"
        )

    rag_section.extend(
        [
            "\n",
            "#### RAG Performance Analysis\n",
            "\n",
            "The human evaluation of RAG queries reveals important insights about how different MUS threshold configurations affect information retrieval quality:\n",
            "\n",
            f"- **Best RAG Performance**: The {comparative_data['summary']['best_rag_performance']} configuration achieved the highest average RAG score, demonstrating superior information retrieval quality.\n",
            "\n",
            "- **Memory Efficiency vs. RAG Performance**: There is a trade-off between memory efficiency and RAG performance. Configurations with very aggressive pruning (e.g., `mus_very_low`) show high memory efficiency but lower RAG scores, indicating important information may be lost.\n",
            "\n",
            "- **Balanced Configurations**: The {comparative_data['summary']['recommended_configuration']} configuration provides the best balance between memory efficiency and RAG performance, with an overall score that optimizes both metrics.\n",
            "\n",
            "![RAG Performance by Scenario](./tuning_results/rag_scores.png)\n",
            "\n",
            "![Combined Performance Metrics](./tuning_results/combined_scores.png)\n",
            "\n",
        ]
    )

    # Replace the RAG section in the report
    # Find where to end the replacement
    next_section_index = len(report_lines)
    for i in range(rag_section_index + 1, len(report_lines)):
        if report_lines[i].startswith("##"):
            next_section_index = i
            break

    # Replace the section
    report_lines = (
        report_lines[:rag_section_index] + rag_section + report_lines[next_section_index:]
    )

    # Update recommendations section
    recommendations_section_index = -1
    for i, line in enumerate(report_lines):
        if line.strip() == "### Recommendations" or line.strip() == "## Recommendations":
            recommendations_section_index = i
            break

    if recommendations_section_index != -1:
        # Find where the recommendations section ends
        next_section_index = len(report_lines)
        for i in range(recommendations_section_index + 1, len(report_lines)):
            if report_lines[i].startswith("##"):
                next_section_index = i
                break

        # Create new recommendations content
        best_config = comparative_data["summary"]["recommended_configuration"]
        best_config_data = comparative_data["scenarios"].get(best_config, {})
        best_config_description = best_config_data.get("description", best_config)

        recommendations = [
            "## Recommendations\n",
            "\n",
            "### Final Analysis\n",
            "\n",
            "After analyzing both quantitative memory metrics and qualitative RAG performance assessments, we have identified the optimal MUS threshold configuration that balances memory efficiency with information retrieval quality.\n",
            "\n",
            f"The **{best_config}** configuration ({best_config_description}) provides the best overall performance with:\n",
            "\n",
            f"- Memory Efficiency Score: {best_config_data.get('memory_efficiency_score', 0.0):.3f}\n",
            f"- RAG Performance Score: {best_config_data.get('rag_assessment', {}).get('avg_score', 0.0):.3f} / 5.0\n",
            f"- Overall Score: {best_config_data.get('overall_score', 0.0):.3f}\n",
            "\n",
            "This configuration strikes an optimal balance between maintaining a compact memory footprint and preserving the most useful information for retrieval.\n",
            "\n",
            "### Top Configuration Rankings\n",
            "\n",
            "| Rank | Configuration | Overall Score | Memory Efficiency | RAG Performance | Memory Retention % |\n",
            "|------|--------------|---------------|-------------------|-----------------|-------------------|\n",
        ]

        # Sort scenarios by overall score
        sorted_scenarios = sorted(
            comparative_data["scenarios"].keys(),
            key=lambda sid: comparative_data["scenarios"][sid].get("overall_score", 0.0),
            reverse=True,
        )

        # Add top 5 configurations to the recommendations
        for i, scenario_id in enumerate(sorted_scenarios[:5]):
            scenario_data = comparative_data["scenarios"][scenario_id]
            mem_efficiency = scenario_data.get("memory_efficiency_score", 0.0)
            rag_score = scenario_data.get("rag_assessment", {}).get("avg_score", 0.0)
            overall_score = scenario_data.get("overall_score", 0.0)

            # Calculate memory retention from memory metrics
            baseline_memories = (
                comparative_data["scenarios"]
                .get("baseline", {})
                .get("memory_metrics", {})
                .get("total_memories", 180)
            )
            scenario_memories = scenario_data.get("memory_metrics", {}).get("total_memories", 0)
            retention_pct = (
                (scenario_memories / baseline_memories) * 100 if baseline_memories > 0 else 0
            )

            recommendations.append(
                f"| {i + 1} | {scenario_id} | {overall_score:.3f} | {mem_efficiency:.3f} | {rag_score:.2f} | {retention_pct:.1f}% |\n"
            )

        # Add recommended MUS threshold settings
        recommendations.extend(
            [
                "\n",
                "### Recommended MUS Threshold Settings\n",
                "\n",
                "| Parameter | Value |\n",
                "|-----------|-------|\n",
            ]
        )

        # Add specific thresholds based on the recommended configuration
        if best_config == "mus_very_low":
            recommendations.extend(
                [
                    "| MEMORY_PRUNING_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L1_MUS_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L2_MUS_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L1_MUS_THRESHOLD | 0.1 |\n",
                    "| MEMORY_PRUNING_L2_MUS_THRESHOLD | 0.1 |\n",
                ]
            )
        elif best_config == "mus_low":
            recommendations.extend(
                [
                    "| MEMORY_PRUNING_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L1_MUS_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L2_MUS_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L1_MUS_THRESHOLD | 0.2 |\n",
                    "| MEMORY_PRUNING_L2_MUS_THRESHOLD | 0.2 |\n",
                ]
            )
        elif best_config == "mus_medium_low":
            recommendations.extend(
                [
                    "| MEMORY_PRUNING_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L1_MUS_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L2_MUS_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L1_MUS_THRESHOLD | 0.25 |\n",
                    "| MEMORY_PRUNING_L2_MUS_THRESHOLD | 0.25 |\n",
                ]
            )
        elif best_config == "mus_medium":
            recommendations.extend(
                [
                    "| MEMORY_PRUNING_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L1_MUS_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L2_MUS_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L1_MUS_THRESHOLD | 0.3 |\n",
                    "| MEMORY_PRUNING_L2_MUS_THRESHOLD | 0.3 |\n",
                ]
            )
        elif best_config == "l1_low_l2_medium":
            recommendations.extend(
                [
                    "| MEMORY_PRUNING_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L1_MUS_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L2_MUS_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L1_MUS_THRESHOLD | 0.2 |\n",
                    "| MEMORY_PRUNING_L2_MUS_THRESHOLD | 0.3 |\n",
                ]
            )
        elif best_config == "l1_medium_l2_low":
            recommendations.extend(
                [
                    "| MEMORY_PRUNING_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L1_MUS_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L2_MUS_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L1_MUS_THRESHOLD | 0.3 |\n",
                    "| MEMORY_PRUNING_L2_MUS_THRESHOLD | 0.2 |\n",
                ]
            )
        else:
            recommendations.extend(
                [
                    "| MEMORY_PRUNING_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L1_MUS_ENABLED | true |\n",
                    "| MEMORY_PRUNING_L2_MUS_ENABLED | true |\n",
                    f"| MEMORY_PRUNING_L1_MUS_THRESHOLD | [See {best_config} description] |\n",
                    f"| MEMORY_PRUNING_L2_MUS_THRESHOLD | [See {best_config} description] |\n",
                ]
            )

        # Replace recommendations section
        report_lines = (
            report_lines[:recommendations_section_index]
            + recommendations
            + report_lines[next_section_index:]
        )

    # Write the updated report
    try:
        with open(report_file, "w") as f:
            f.writelines(report_lines)
        print(f"Updated report file: {report_file}")
    except Exception as e:
        print(f"Error writing report file: {e}")


if __name__ == "__main__":
    exit(main())
