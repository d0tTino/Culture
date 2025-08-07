#!/usr/bin/env python
"""Simple benchmark comparing vLLM and Ollama latency and throughput."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable

# Add project root so we can import llm_client directly when executed from scripts/
sys.path.append(str(os.path.dirname(os.path.dirname(__file__))))

from src.infra import config as infra_config
from src.infra import llm_client


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM vs Ollama latency and throughput"
    )
    parser.add_argument("prompt", type=str, nargs="?", default="Hello", help="Prompt text")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per backend")
    parser.add_argument("--model", type=str, default="mistral:latest", help="Model name")
    parser.add_argument(
        "--vllm_base",
        type=str,
        default=os.environ.get("VLLM_API_BASE", "http://localhost:8000"),
        help="Base URL of the vLLM server",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write benchmark results as JSON",
    )
    return parser.parse_args()


def time_calls(func: Callable[[], None], runs: int) -> list[float]:
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        try:
            func()
        except Exception as exc:
            print(f"Call failed: {exc}")
            return []
        times.append(time.perf_counter() - start)
    return times


def benchmark(prompt: str, model: str, runs: int, vllm_base: str, output: str | None) -> None:
    # Benchmark vLLM first
    os.environ["VLLM_API_BASE"] = vllm_base
    infra_config.load_config(validate_required=False)
    try:
        llm_client.get_llm_client()  # refresh to use vLLM
    except llm_client.LLMClientInitError as exc:
        print(f"Failed to initialize vLLM client: {exc}")
        return

    def vllm_call() -> None:
        llm_client.generate_text(prompt, model=model)

    vllm_times = time_calls(vllm_call, runs)
    if not vllm_times:
        print("vLLM calls failed; aborting benchmark")
        return

    # Benchmark Ollama as fallback
    os.environ.pop("VLLM_API_BASE", None)
    infra_config.load_config(validate_required=False)
    try:
        llm_client.get_llm_client()  # switch to Ollama
    except llm_client.LLMClientInitError as exc:
        print(f"Failed to initialize Ollama client: {exc}")
        return

    def ollama_call() -> None:
        llm_client.generate_text(prompt, model=model)

    ollama_times = time_calls(ollama_call, runs)
    if not ollama_times:
        print("Ollama calls failed; aborting benchmark")
        return

    def avg(values: list[float]) -> float:
        return sum(values) / len(values)

    def throughput(values: list[float]) -> float:
        return runs / sum(values)

    results = {
        "vllm": {"avg_latency": avg(vllm_times), "throughput": throughput(vllm_times)},
        "ollama": {"avg_latency": avg(ollama_times), "throughput": throughput(ollama_times)},
    }

    print("| Backend | Avg latency (s) | Throughput (req/s) |")
    print("|---------|----------------:|--------------------:|")
    print(f"| vLLM | {results['vllm']['avg_latency']:.2f} | {results['vllm']['throughput']:.2f} |")
    print(f"| Ollama | {results['ollama']['avg_latency']:.2f} | {results['ollama']['throughput']:.2f} |")

    if output:
        import json

        with open(output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"Results written to {output}")


def main() -> None:
    args = parse_args()
    benchmark(args.prompt, args.model, args.runs, args.vllm_base, args.output)


if __name__ == "__main__":
    main()
