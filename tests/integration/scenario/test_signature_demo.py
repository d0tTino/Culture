import json
from itertools import cycle
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from scripts import export_traces
from src.agents.graphs.basic_agent_types import AgentActionOutput
from src.app import create_simulation, load_scenario
from src.infra import event_log
from src.infra import snapshot as snap
from tests.utils.mock_llm import MockLLM


@pytest.mark.asyncio
@pytest.mark.integration
async def test_signature_demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the signature_demo scenario and verify evaluation metrics."""
    monkeypatch.setenv("SNAPSHOT_INTERVAL_STEPS", "1")
    monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "event_log.jsonl"))
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)
    monkeypatch.setattr(event_log, "_producer", None, raising=False)
    monkeypatch.setattr(event_log, "_last_hash", None, raising=False)
    monkeypatch.setattr(event_log, "_header_written", False, raising=False)
    event_log.set_seed(42)
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

    (
        scenario_desc,
        _,
        _,
        beats,
        hook_names,
        evaluation_targets,
        success_metrics,
    ) = load_scenario("scenarios/signature_demo.yaml")
    beat_cycle = cycle(beats)

    def next_structured_output() -> AgentActionOutput:
        beat = next(beat_cycle)
        return AgentActionOutput(
            thought="t",
            message_content=f"{beat} message",
            message_recipient_id=None,
            action_intent="continue_collaboration",
            requested_role_change=None,
            project_name_to_create=None,
            project_description_for_creation=None,
            project_id_to_join_or_leave=None,
        )

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
        sim = create_simulation(
            num_agents=3,
            steps=30,
            scenario=scenario_desc,
            seed=42,
            beats=beats,
            evaluation_hook_names=hook_names,
            evaluation_targets=evaluation_targets,
            success_metrics=success_metrics,
        )
        sim._beat_interval = max(1, sim.steps_to_run // len(beats))
        assert sim.evaluation_hook_names == hook_names
        assert sim.evaluation_targets == evaluation_targets
        assert sim.success_metrics == success_metrics
        for _ in range(30):
            await sim.run_step()

    metrics = sim.metrics
    assert len(metrics) == sim.steps_to_run + len(beats)
    assert all(m["coalitions"] == 0 for m in metrics)
    sentiments = [m["sentiment"] for m in metrics]
    assert all(s == 0.0 for s in sentiments)
    ip_values = [m["collective_ip"] for m in metrics]
    du_values = [m["collective_du"] for m in metrics]
    assert ip_values == [ip_values[0]] * len(metrics)
    assert du_values == [du_values[0]] * len(metrics)

    out = tmp_path / "traces.jsonl"
    ret = export_traces.main(["--snapshots", str(tmp_path), "-o", str(out)])
    assert ret == 0
    lines = out.read_text().splitlines()
    assert len(lines) == sim.steps_to_run

    log_path = tmp_path / "event_log.jsonl"
    events = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line.strip() and "beat" in line
    ]
    beat_events = [ev["beat"] for ev in events if ev.get("type") == "evaluation"]
    assert beat_events == beats
