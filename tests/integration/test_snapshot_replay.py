import json
from pathlib import Path

import pytest

from src.app import create_simulation
from src.infra import config, event_log
from src.infra.checkpoint import (
    load_checkpoint,
    restore_environment,
    restore_rng_state,
    save_checkpoint,
)
from src.infra.snapshot import compute_trace_hash
from tests.utils.mock_llm import MockLLM


@pytest.mark.asyncio
@pytest.mark.integration
async def test_snapshot_replay_trace_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNAPSHOT_INTERVAL_STEPS", "1")
    config.load_config(validate_required=False)

    monkeypatch.setattr(event_log, "log_event", lambda e: e)
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)
    monkeypatch.setattr("src.sim.simulation.upload_snapshot", lambda *a, **k: None)

    snap_dir1 = tmp_path / "run1"
    snap_dir2 = tmp_path / "run2"
    snap_dir1.mkdir()
    snap_dir2.mkdir()

    def save1(step: int, data: dict, directory: Path = snap_dir1) -> None:
        from src.infra.snapshot import save_snapshot as real_save

        real_save(step, data, directory)

    def save2(step: int, data: dict, directory: Path = snap_dir2) -> None:
        from src.infra.snapshot import save_snapshot as real_save

        real_save(step, data, directory)

    monkeypatch.setattr("src.sim.simulation.save_snapshot", save1)

    with MockLLM():
        sim = create_simulation(num_agents=1, steps=1, scenario="test")
        chk = tmp_path / "sim.pkl"
        save_checkpoint(sim, chk)
        await sim.run_step()

    snap1_path = snap_dir1 / "snapshot_1.json"
    with snap1_path.open() as f:
        snap1 = json.load(f)
    snap1_no_vector = {
        **{k: v for k, v in snap1.items() if k != "trace_hash"},
        "knowledge_board": {k: v for k, v in snap1["knowledge_board"].items() if k != "vector"},
        "world_map": {k: v for k, v in snap1["world_map"].items() if k != "vector"},
    }
    expected_hash = compute_trace_hash(snap1_no_vector)
    assert snap1["trace_hash"] == expected_hash

    loaded, meta = load_checkpoint(chk, replay=True)
    restore_rng_state(meta.get("rng_state"))
    restore_environment(meta.get("environment"))

    monkeypatch.setattr("src.sim.simulation.save_snapshot", save2)

    with MockLLM():
        await loaded.run_step()

    snap2_path = snap_dir2 / "snapshot_1.json"
    with snap2_path.open() as f:
        snap2 = json.load(f)
    snap2_no_vector = {
        **{k: v for k, v in snap2.items() if k != "trace_hash"},
        "knowledge_board": {k: v for k, v in snap2["knowledge_board"].items() if k != "vector"},
        "world_map": {k: v for k, v in snap2["world_map"].items() if k != "vector"},
    }
    replay_hash = compute_trace_hash(snap2_no_vector)

    assert replay_hash == expected_hash
