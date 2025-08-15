import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.app import create_simulation
from src.infra import event_log
from src.infra.snapshot import compute_trace_hash, load_snapshot, save_snapshot
from src.sim.simulation import Simulation
from tests.utils.mock_llm import MockLLM


def _snapshot_from_sim(sim: Simulation) -> dict:
    snapshot = {
        "step": sim.current_step,
        "collective_ip": sim.collective_ip,
        "collective_du": sim.collective_du,
        "knowledge_board": sim.knowledge_board.to_dict(),
        "world_map": sim.world_map.to_dict(),
        "agents": [
            {
                "agent_id": ag.agent_id,
                "ip": ag.state.ip,
                "du": ag.state.du,
            }
            for ag in sim.agents
        ],
    }
    snapshot_no_vector = {
        **{k: v for k, v in snapshot.items() if k != "trace_hash"},
        "knowledge_board": {k: v for k, v in snapshot["knowledge_board"].items() if k != "vector"},
        "world_map": {k: v for k, v in snapshot["world_map"].items() if k != "vector"},
    }
    snapshot["trace_hash"] = compute_trace_hash(snapshot_no_vector)
    return snapshot


def _hash_without_mood(snap: dict) -> str:
    data = {
        **{k: v for k, v in snap.items() if k not in {"trace_hash", "seed", "rng_state"}},
        "knowledge_board": {
            k: v for k, v in snap.get("knowledge_board", {}).items() if k != "vector"
        },
        "world_map": {k: v for k, v in snap.get("world_map", {}).items() if k != "vector"},
        "agents": [{k: v for k, v in ag.items() if k != "mood"} for ag in snap.get("agents", [])],
    }
    return compute_trace_hash(data)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_replay_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_REDPANDA", "0")
    monkeypatch.setenv("SNAPSHOT_INTERVAL_STEPS", "1")
    log_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(log_path))
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)
    monkeypatch.setattr(event_log, "_producer", None, raising=False)
    monkeypatch.setattr("src.sim.simulation.upload_snapshot", lambda *a, **k: None)

    def save(step: int, data: dict, directory: Path = tmp_path) -> None:
        from src.infra.snapshot import save_snapshot as real_save

        real_save(step, data, directory)

    def load(step: int | str | Path, directory: Path = tmp_path):
        from src.infra.snapshot import load_snapshot as real_load

        return real_load(step, directory=directory)

    monkeypatch.setattr("src.sim.simulation.save_snapshot", save)
    monkeypatch.setattr("src.sim.simulation.load_snapshot", load)

    with MockLLM():
        sim = create_simulation(num_agents=1, steps=1, scenario="test")
        sim.evaluation_hooks = []
        snap0 = _snapshot_from_sim(sim)
        save_snapshot(0, snap0, directory=tmp_path)
        for _ in range(10):
            await sim.run_step()

    originals: dict[int, dict] = {}
    for step in range(5, 11):
        originals[step] = load_snapshot(tmp_path / f"snapshot_{step}.json")

    stub_pkg = tmp_path / "matplotlib"
    stub_pkg.mkdir()
    (stub_pkg / "__init__.py").write_text("")
    (stub_pkg / "pyplot.py").write_text(
        """
fig = figure = lambda *a, **k: None
plot = xlabel = ylabel = title = tight_layout = savefig = close = lambda *a, **k: None
"""
    )

    runner = tmp_path / "export_runner.py"
    runner.write_text(
        "import tools.export_traces as m, sys\n"
        "m.scenario_metrics = {}\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(m.main())\n"
    )

    env = {**os.environ, "PYTHONPATH": f"{tmp_path}:{os.environ.get('PYTHONPATH', '')}"}
    subprocess.run(
        [
            sys.executable,
            str(runner),
            str(log_path),
            "--start",
            "5",
            "--end",
            "10",
        ],
        check=True,
        env=env,
    )

    sliced_log = tmp_path / "logs.jsonl"
    with log_path.open("r", encoding="utf-8") as src, sliced_log.open(
        "w", encoding="utf-8"
    ) as out:
        for line in src:
            obj = json.loads(line)
            tick = int(obj.get("tick", obj.get("step", 0)))
            if 5 <= tick <= 10:
                out.write(json.dumps(obj) + "\n")
    monkeypatch.setenv("EVENT_LOG_PATH", str(sliced_log))

    for step in range(5, 11):
        replay = Simulation.replay_from_snapshot(
            tmp_path / f"snapshot_{step-1}.json", start_step=step, end_step=step
        )
        replay_snap = _snapshot_from_sim(replay)
        expected_hash = _hash_without_mood(originals[step])
        assert replay_snap["trace_hash"] == expected_hash
