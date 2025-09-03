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
from src.sim.simulation import Simulation
from tests.utils.mock_llm import MockLLM


@pytest.mark.asyncio
@pytest.mark.integration
async def test_signature_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run signature_demo and verify evaluation metrics are exported."""
    monkeypatch.setenv("SNAPSHOT_INTERVAL_STEPS", "1")
    log_path = tmp_path / "event_log.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(log_path))
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

    scenario_desc, _, _, beats = load_scenario("scenarios/signature_demo.yaml")
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
        )
        sim._beat_interval = max(1, sim.steps_to_run // len(beats))
        sim.evaluation_hooks = [Simulation._collect_metrics]
        for _ in range(30):
            await sim.run_step()

    metrics = sim.metrics
    assert metrics
    assert all("coalitions" in m for m in metrics)
    assert all("sentiment" in m for m in metrics)
    assert all("collective_du" in m for m in metrics)
    assert all("collective_ip" in m for m in metrics)

    log_lines = log_path.read_text().splitlines()
    assert any('"type": "evaluation"' in line for line in log_lines)

    traces = tmp_path / "traces.jsonl"
    ret = export_traces.main(["--events", str(log_path), "-o", str(traces)])
    assert ret == 0

    data = export_traces.load_metrics(log_path)
    assert {"coalitions", "sentiment", "collective_du", "collective_ip"} <= data.keys()
    assert all(data[key] for key in ["coalitions", "sentiment", "collective_du", "collective_ip"])
