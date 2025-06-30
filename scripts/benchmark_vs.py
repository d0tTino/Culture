import asyncio
import time
import json
from pathlib import Path
import sys

# Add project root to path to allow importing src modules
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.app import create_simulation

def run_benchmark():
    """
    Runs the vertical slice simulation, measures performance, and logs the results.
    """
    start_time = time.monotonic()

    sim = create_simulation(num_agents=3, steps=3, use_vector_store=True)
    asyncio.run(sim.async_run(sim.steps_to_run))

    end_time = time.monotonic()
    duration = end_time - start_time

    # Collect results
    du_usage = {agent.agent_id: agent.state.du for agent in sim.agents}
    results = {
        "timestamp": time.time(),
        "duration_seconds": duration,
        "du_usage": du_usage,
        "total_du_used": sum(du_usage.values()),
    }

    # Write results to a file
    benchmarks_dir = Path("benchmarks")
    benchmarks_dir.mkdir(exist_ok=True)
    results_file = benchmarks_dir / f"benchmark_{int(time.time())}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Benchmark complete. Results saved to {results_file}")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    run_benchmark() 