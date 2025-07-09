#!/usr/bin/env python
"""Simple latency benchmark comparing Ollama and vLLM."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable

# Add project root so we can import llm_client directly when executed from scripts/
sys.path.append(str(os.path.dirname(os.path.dirname(__file__))))

from src.infra import llm_client


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark Ollama vs vLLM latency")
    parser.add_argument("prompt", type=str, nargs="?", default="Hello", help="Prompt text")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per backend")
    parser.add_argument("--model", type=str, default="mistral:latest", help="Model name")
    parser.add_argument(
        "--vllm_base",
        type=str,
        default=os.environ.get("VLLM_API_BASE", "http://localhost:8001"),
        help="Base URL of the vLLM server",
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


def benchmark(prompt: str, model: str, runs: int, vllm_base: str) -> None:
    # Benchmark Ollama
    os.environ.pop("VLLM_API_BASE", None)
    llm_client.get_ollama_client()  # refresh client

    def ollama_call() -> None:
        llm_client.generate_text(prompt, model=model)

    ollama_times = time_calls(ollama_call, runs)
    if not ollama_times:
        print("Ollama calls failed; aborting benchmark")
        return

    # Benchmark vLLM
    os.environ["VLLM_API_BASE"] = vllm_base
    llm_client.get_ollama_client()  # switch to vLLM

    def vllm_call() -> None:
        llm_client.generate_text(prompt, model=model)

    vllm_times = time_calls(vllm_call, runs)
    if not vllm_times:
        print("vLLM calls failed; aborting benchmark")
        return

    def avg(values: list[float]) -> float:
        return sum(values) / len(values)

    print(f"Ollama avg latency: {avg(ollama_times):.2f}s over {runs} runs")
    print(f"vLLM  avg latency: {avg(vllm_times):.2f}s over {runs} runs")


def main() -> None:
    args = parse_args()
    benchmark(args.prompt, args.model, args.runs, args.vllm_base)


if __name__ == "__main__":
    main()
