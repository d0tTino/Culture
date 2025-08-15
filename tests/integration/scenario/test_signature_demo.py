import json
from pathlib import Path

import pytest

from scripts import export_traces
from src.agents.graphs.basic_agent_types import AgentActionOutput
from src.app import create_simulation
from src.infra import event_log
from src.infra import snapshot as snap
from src.sim.simulation import Simulation
from tests.utils.mock_llm import MockLLM


@pytest.mark.asyncio
@pytest.mark.integration
async def test_signature_demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the signature_demo scenario and verify evaluation metrics."""
    monkeypatch.setenv("SNAPSHOT_INTERVAL_STEPS", "1")
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)
    monkeypatch.setattr(event_log, "_producer", None, raising=False)
    monkeypatch.setattr("src.sim.simulation.upload_snapshot", lambda *a, **k: None)

    def save(step: int, data: dict, directory: Path = tmp_path) -> None:
        from src.infra.snapshot import save_snapshot as real_save

        real_save(step, data, directory)

    monkeypatch.setattr("src.sim.simulation.save_snapshot", save)

    def load_no_verify(
        step: int | str | Path, directory: Path = tmp_path, compress: bool | None = None
    ) -> dict:
        path = Path(directory) / f"snapshot_{step}.json"
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    monkeypatch.setattr(snap, "load_snapshot", load_no_verify)
    monkeypatch.setattr(export_traces, "load_snapshot", load_no_verify)

    structured = AgentActionOutput(
        thought="t",
        message_content="hi",
        message_recipient_id=None,
        action_intent="continue_collaboration",
        requested_role_change=None,
        project_name_to_create=None,
        project_description_for_creation=None,
        project_id_to_join_or_leave=None,
    )
    responses = {"structured_output": structured, "default": "text"}

    with MockLLM(responses):
        sim = create_simulation(num_agents=3, steps=30, scenario="signature_demo.yaml")
        sim.evaluation_hooks = [Simulation._collect_metrics]
        for _ in range(30):
            await sim.run_step()

    metrics = sim.metrics
    assert len(metrics) == 30
    assert all(m["coalitions"] == 0 for m in metrics)
    sentiments = [m["sentiment"] for m in metrics]
    assert all(s == 0.0 for s in sentiments)
    ip_values = [m["collective_ip"] for m in metrics]
    du_values = [m["collective_du"] for m in metrics]
    assert ip_values == [ip_values[0]] * len(ip_values)
    assert du_values == [du_values[0]] * len(du_values)

    out = tmp_path / "traces.jsonl"
    ret = export_traces.main(["--snapshots", str(tmp_path), "-o", str(out)])
    assert ret == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 30
