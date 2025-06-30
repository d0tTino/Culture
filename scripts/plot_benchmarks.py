import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_benchmarks() -> None:
    """
    Reads all benchmark JSON files, creates a plot of DU deltas over time,
    and saves it as a PNG image.
    """
    benchmarks_dir = Path("benchmarks")
    if not benchmarks_dir.exists():
        print("No benchmarks directory found.")
        return

    benchmark_files = list(benchmarks_dir.glob("benchmark_*.json"))
    if not benchmark_files:
        print("No benchmark files found.")
        return

    data = []
    for file in benchmark_files:
        with open(file) as f:
            data.append(json.load(f))

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp")

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        x="timestamp",
        y="total_du_delta",
        data=df,
        marker="o",
        label="Total DU Delta",
    )

    plt.title("Benchmark: Total DU Delta Over Time")
    plt.xlabel("Time")
    plt.ylabel("Total DU Delta")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_file = "benchmarks.png"
    plt.savefig(output_file)
    print(f"Benchmark plot saved to {output_file}")


def plot_performance_over_time(df: pd.DataFrame, output_path: Path) -> None:
    if "total_du_delta" not in df.columns:
        print("Skipping plot: 'total_du_delta' column not found.")
        return
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Benchmark plot saved to {output_path}")


if __name__ == "__main__":
    plot_benchmarks()
