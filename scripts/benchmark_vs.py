import asyncio
import json
import sys
import time
from pathlib import Path

from src.app import create_simulation

# Add project root to path to allow importing src modules
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_benchmark() -> None:
    """
    Runs the vertical slice simulation, measures performance, and logs the results.
    """
    sim = create_simulation(num_agents=3, steps=3, use_vector_store=True)

    initial_du = {agent.agent_id: agent.state.du for agent in sim.agents}

    start_time = time.monotonic()
    asyncio.run(sim.async_run(sim.steps_to_run))
    end_time = time.monotonic()

    duration = end_time - start_time

    final_du = {agent.agent_id: agent.state.du for agent in sim.agents}
    du_deltas = {
        agent_id: final_du[agent_id] - initial_du.get(agent_id, 0)
        for agent_id in final_du
    }

    # Collect results
    results = {
        "timestamp": time.time(),
        "duration_seconds": duration,
        "initial_du": initial_du,
        "final_du": final_du,
        "du_deltas": du_deltas,
        "total_du_delta": sum(du_deltas.values()),
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
