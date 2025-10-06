import json
import sys
import types
import zipfile
from itertools import count, cycle
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.infra import metrics as metrics_module


def _ensure_record_du_budget_stub() -> None:
    """Provide a lightweight ``record_du_budget`` helper if missing."""

    if hasattr(metrics_module, "record_du_budget"):
        return

    def record_du_budget(agent_id: str, value: float) -> None:
        remaining = float(value)
        metrics_module._agent_du_budget[agent_id] = remaining
        gauge = getattr(metrics_module.prom_metrics, "AGENT_REMAINING_DU", None)
        if hasattr(gauge, "labels"):
            try:
                gauge.labels(agent_id=agent_id).set(remaining)
            except Exception:
                try:
                    gauge.set(remaining)
                except Exception:
                    pass
        else:
            try:
                gauge.set(remaining)
            except Exception:
                pass
        hook = getattr(metrics_module.ledger, "record_du_budget", None)
        if callable(hook):
            try:
                hook(agent_id, remaining)
            except Exception:
                pass

    metrics_module.record_du_budget = record_du_budget  # type: ignore[attr-defined]


_ensure_record_du_budget_stub()

from scripts import run_signature_demo
from src.agents.graphs.basic_agent_types import AgentActionOutput
from src.app import create_simulation, load_scenario
from src.infra import event_log as event_log_module
from src.infra.ledger import Ledger
from tests.utils.mock_llm import MockLLM
from tests.utils.mock_s3 import setup_mock_s3


@pytest.mark.asyncio
@pytest.mark.integration
async def test_run_signature_demo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Execute ``run_signature_demo`` and verify produced artifacts."""

    # Redirect results directory to an isolated temporary location.
    result_dir = tmp_path / "signature_demo"
    monkeypatch.setattr(run_signature_demo, "RESULT_DIR", result_dir)
    monkeypatch.setattr(run_signature_demo, "README_PATH", result_dir / "README.md")
    scenario_path = Path(__file__).resolve().parents[3] / "scenarios" / "signature_demo.yaml"
    monkeypatch.setattr(run_signature_demo, "SCENARIO_PATH", scenario_path)

    # Ensure simulation snapshots produced during the run stay inside the tmp path.
    auto_snapshot_dir = tmp_path / "auto_snapshots"
    auto_snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save(step: int, data: dict, directory: Path = auto_snapshot_dir) -> None:
        from src.infra.snapshot import save_snapshot as real_save

        real_save(step, data, directory)

    monkeypatch.setattr("src.sim.simulation.save_snapshot", save)

    # Provide a lightweight matplotlib stub so plot generation succeeds without the real package.
    pyplot_stub = types.SimpleNamespace(
        figure=lambda: None,
        plot=lambda *a, **k: None,
        xlabel=lambda *a, **k: None,
        ylabel=lambda *a, **k: None,
        title=lambda *a, **k: None,
        tight_layout=lambda: None,
        savefig=lambda path: Path(path).write_text("dummy plot"),
        close=lambda *a, **k: None,
    )
    matplotlib_stub = types.SimpleNamespace(pyplot=pyplot_stub)
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_stub)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot_stub)

    # Install the in-repo boto3 stub and route S3 writes to the tmp directory.
    s3_client = setup_mock_s3()
    s3_client.base_dir = tmp_path / "mock_s3"
    monkeypatch.setattr("src.infra.snapshot.boto3", sys.modules["boto3"], raising=False)
    monkeypatch.setattr("src.infra.snapshot._s3_client", None, raising=False)
    monkeypatch.setattr("src.infra.snapshot.S3_BUCKET", "test-bucket", raising=False)
    monkeypatch.setattr("src.infra.snapshot.S3_PREFIX", "runs", raising=False)
    monkeypatch.setattr("src.infra.snapshot.SNAPSHOT_COMPRESS", False, raising=False)

    # Isolate the ledger database so the demo run does not touch the shared file.
    ledger_instance = Ledger(tmp_path / "ledger.sqlite3")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger_instance)
    request.addfinalizer(ledger_instance.conn.close)

    # Reset the event log state and use a deterministic seed for reproducible headers.
    monkeypatch.setattr(event_log_module, "_get_producer", lambda: None)
    monkeypatch.setattr(event_log_module, "_producer", None, raising=False)
    monkeypatch.setattr(event_log_module, "_last_hash", None, raising=False)
    monkeypatch.setattr(event_log_module, "_seed", None, raising=False)
    monkeypatch.setattr(event_log_module, "_seed_cache", {}, raising=False)
    monkeypatch.setattr(event_log_module, "_header_written", set(), raising=False)
    event_log_module.set_seed(42)

    # Build deterministic agent outputs tied to the scenario beats.
    (
        _,
        _,
        _,
        beats,
        hook_names,
        evaluation_targets,
        success_metrics,
    ) = load_scenario(str(scenario_path))
    beat_cycle = cycle(beats or ["default"])
    message_counter = count(1)

    def next_structured_output() -> AgentActionOutput:
        beat = next(beat_cycle)
        idx = next(message_counter)
        return AgentActionOutput(
            thought=f"thought-{idx}",
            message_content=f"{beat} message {idx}",
            message_recipient_id=None,
            action_intent="continue_collaboration",
            requested_role_change=None,
            project_name_to_create=None,
            project_description_for_creation=None,
            project_id_to_join_or_leave=None,
        )

    created_sims: list = []

    def create_simulation_stub(**kwargs):
        sim = create_simulation(**kwargs)
        created_sims.append(sim)
        return sim

    monkeypatch.setattr(run_signature_demo, "create_simulation", create_simulation_stub)

    responses = {"default": "text"}
    with MockLLM(responses, strict_mode=False):
        monkeypatch.setattr(
            "src.infra.llm_client.generate_structured_output",
            lambda *a, **k: next_structured_output(),
        )
        monkeypatch.setattr(
            "src.infra.llm_client.async_generate_structured_output",
            AsyncMock(side_effect=lambda *a, **k: next_structured_output()),
        )
        await run_signature_demo.main()

    assert created_sims, "Simulation was not constructed"
    sim = created_sims[0]
    assert sim.evaluation_hook_names == hook_names
    assert sim.evaluation_targets == evaluation_targets
    assert sim.success_metrics == success_metrics

    event_log_path = result_dir / "event_log.jsonl"
    metrics_path = result_dir / "metrics.json"
    traces_path = result_dir / "traces.jsonl"
    bundle_path = result_dir / "signature_demo_bundle.zip"
    snapshots_dir = result_dir / "snapshots"

    assert event_log_path.exists(), "Event log was not generated"
    assert metrics_path.exists(), "Metrics JSON was not generated"
    assert traces_path.exists(), "Trace dataset was not generated"
    assert bundle_path.exists(), "Bundle archive was not generated"
    assert snapshots_dir.is_dir(), "Snapshot directory missing"

    log_lines = event_log_path.read_text(encoding="utf-8").splitlines()
    assert log_lines, "Event log is empty"
    header = json.loads(log_lines[0])
    assert header["type"] == "header"
    assert header["seed"] == 42

    events = [json.loads(line) for line in log_lines[1:] if line.strip()]
    assert events, "Event log contains no events"
    max_step = max(ev.get("step", 0) for ev in events)
    assert max_step > 0

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in ("coalitions", "sentiment", "collective_du", "collective_ip"):
        assert key in metrics
        assert metrics[key], f"Metric '{key}' has no data points"
        step, value = metrics[key][0]
        assert isinstance(step, int)
        assert isinstance(value, float)
    target_summary = metrics.get("_target_summary")
    assert isinstance(target_summary, dict)
    for key in ("coalitions", "sentiment", "collective_du", "collective_ip"):
        assert key in target_summary
        entry = target_summary[key]
        assert isinstance(entry, dict)
        assert "status" in entry

    replay_files = sorted(snapshots_dir.glob("replay_*.jsonl"))
    assert replay_files, "Replay slice not created"
    replay_path = replay_files[0]
    replay_lines = replay_path.read_text(encoding="utf-8").splitlines()
    assert replay_lines, "Replay slice is empty"
    replay_header = json.loads(replay_lines[0])
    assert replay_header["type"] == "header"
    assert replay_header["seed"] == 42
    assert replay_path.name.endswith(f"_{max_step}.jsonl")

    snapshot_files = sorted(p for p in snapshots_dir.glob("snapshot_*.json"))
    assert snapshot_files, "Snapshot JSON not written"
    snapshot = json.loads(snapshot_files[-1].read_text(encoding="utf-8"))
    assert snapshot.get("seed") == 42
    assert snapshot.get("trace_hash")

    with zipfile.ZipFile(bundle_path) as zf:
        names = set(zf.namelist())
        assert "traces.jsonl" in names
        assert "metrics.json" in names
        assert any(name.startswith("snapshots/") for name in names)

    readme_text = (result_dir / "README.md").read_text(encoding="utf-8")
    assert "Evaluation Target Summary" in readme_text
