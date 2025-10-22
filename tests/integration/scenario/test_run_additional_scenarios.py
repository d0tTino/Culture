import json
import sys
import types
import zipfile
from itertools import count, cycle
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from scripts import run_signature_demo
from src.agents.graphs.basic_agent_types import AgentActionOutput
from src.app import create_simulation, load_scenario
from src.infra import event_log as event_log_module
from src.infra.ledger import Ledger
from tests.integration.scenario.test_run_signature_demo import (
    _ensure_record_du_budget_stub,
)
from tests.utils.mock_llm import MockLLM
from tests.utils.mock_s3 import setup_mock_s3


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("slug,extra_metric", [
    ("crisis_response", "response_alignment"),
    ("research_sprint", "insight_velocity"),
])
async def test_run_additional_scenarios(
    slug: str,
    extra_metric: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Run additional scenarios through the signature runner and validate artifacts."""

    _ensure_record_du_budget_stub()

    scenario_path = Path(__file__).resolve().parents[3] / "scenarios" / f"{slug}.yaml"
    result_dir = tmp_path / slug

    monkeypatch.setattr(run_signature_demo, "SCENARIO_SLUG", slug, raising=False)
    monkeypatch.setattr(
        run_signature_demo, "SCENARIO_TITLE", slug.replace("_", " ").title(), raising=False
    )
    monkeypatch.setattr(run_signature_demo, "SCENARIO_PATH", scenario_path, raising=False)
    monkeypatch.setattr(run_signature_demo, "RESULT_DIR", result_dir, raising=False)
    monkeypatch.setattr(
        run_signature_demo, "README_PATH", result_dir / "README.md", raising=False
    )
    monkeypatch.setattr(
        run_signature_demo, "BUNDLE_NAME", f"{slug}_bundle.zip", raising=False
    )

    auto_snapshot_dir = tmp_path / f"{slug}_auto_snapshots"
    auto_snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save(step: int, data: dict, directory: Path = auto_snapshot_dir) -> None:
        from src.infra.snapshot import save_snapshot as real_save

        real_save(step, data, directory)

    monkeypatch.setattr("src.sim.simulation.save_snapshot", save)

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

    s3_client = setup_mock_s3()
    s3_client.base_dir = tmp_path / f"mock_s3_{slug}"
    monkeypatch.setattr("src.infra.snapshot.boto3", sys.modules["boto3"], raising=False)
    monkeypatch.setattr("src.infra.snapshot._s3_client", None, raising=False)
    monkeypatch.setattr("src.infra.snapshot.S3_BUCKET", "test-bucket", raising=False)
    monkeypatch.setattr("src.infra.snapshot.S3_PREFIX", slug, raising=False)
    monkeypatch.setattr("src.infra.snapshot.SNAPSHOT_COMPRESS", False, raising=False)

    ledger_instance = Ledger(tmp_path / f"ledger_{slug}.sqlite3")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger_instance)
    request.addfinalizer(ledger_instance.conn.close)

    monkeypatch.setattr(event_log_module, "_get_producer", lambda: None)
    monkeypatch.setattr(event_log_module, "_producer", None, raising=False)
    monkeypatch.setattr(event_log_module, "_last_hash", None, raising=False)
    monkeypatch.setattr(event_log_module, "_seed", None, raising=False)
    monkeypatch.setattr(event_log_module, "_seed_cache", {}, raising=False)
    monkeypatch.setattr(event_log_module, "_header_written", set(), raising=False)
    event_log_module.set_seed(21)

    (
        _desc,
        scenario_steps_override,
        scenario_agents_override,
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
    create_kwargs: list[dict[str, object]] = []

    def create_simulation_stub(**kwargs):
        create_kwargs.append(dict(kwargs))
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

        await run_signature_demo.main(seed=21)

    assert created_sims, "Simulation was not constructed"
    default_kwargs = create_kwargs[0]
    expected_agents = scenario_agents_override or 3
    expected_steps = scenario_steps_override or 0

    assert default_kwargs["num_agents"] == expected_agents
    assert default_kwargs["steps"] == expected_steps

    default_sim = created_sims[0]
    assert default_sim.evaluation_hook_names == hook_names
    assert default_sim.evaluation_targets == evaluation_targets
    assert default_sim.success_metrics == success_metrics

    event_log_path = result_dir / "event_log.jsonl"
    metrics_path = result_dir / "metrics.json"
    traces_path = result_dir / "traces.jsonl"
    bundle_path = result_dir / f"{slug}_bundle.zip"
    snapshots_dir = result_dir / "snapshots"

    assert event_log_path.exists()
    assert metrics_path.exists()
    assert traces_path.exists()
    assert bundle_path.exists()
    assert snapshots_dir.is_dir()

    log_lines = event_log_path.read_text(encoding="utf-8").splitlines()
    assert log_lines, "Event log is empty"
    header = json.loads(log_lines[0])
    assert header["type"] == "header"
    assert header["seed"] == 21

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in ("coalitions", "sentiment", "collective_du", "collective_ip"):
        assert key in metrics
        assert metrics[key], f"Metric '{key}' has no samples"

    target_summary = metrics.get("_target_summary")
    assert isinstance(target_summary, dict)
    for key in ("coalitions", "sentiment", "collective_du", "collective_ip"):
        assert key in target_summary

    success_guidance = metrics.get("_success_metrics_guidance")
    assert isinstance(success_guidance, dict)
    assert extra_metric in success_guidance

    replay_files = sorted(snapshots_dir.glob("replay_*.jsonl"))
    assert replay_files
    replay_lines = replay_files[0].read_text(encoding="utf-8").splitlines()
    assert replay_lines
    replay_header = json.loads(replay_lines[0])
    assert replay_header["type"] == "header"

    snapshot_files = sorted(snapshots_dir.glob("snapshot_*.json"))
    assert snapshot_files
    snapshot = json.loads(snapshot_files[-1].read_text(encoding="utf-8"))
    assert snapshot.get("seed") == 21
    assert snapshot.get("trace_hash")

    with zipfile.ZipFile(bundle_path) as zf:
        names = set(zf.namelist())
        assert "traces.jsonl" in names
        assert "metrics.json" in names
        assert any(name.startswith("snapshots/") for name in names)

    readme_text = (result_dir / "README.md").read_text(encoding="utf-8")
    assert slug.replace("_", " ").title() in readme_text
    assert f"`{slug}`" in readme_text
