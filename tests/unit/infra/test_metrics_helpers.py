from collections import deque
from types import SimpleNamespace

import pytest

import src.sim.resource_manager as resource_manager_mod
from src.infra import metrics as infra_metrics


class GaugeChildStub:
    def __init__(self, value: float) -> None:
        self._value = SimpleNamespace(get=lambda: value)


class GaugeStub:
    def __init__(self, mapping: dict[tuple[str, ...], float]) -> None:
        self._metrics = {
            key: GaugeChildStub(val) for key, val in mapping.items()
        }
        self._labelnames = ("agent_id",)

    def labels(self, **labels: str) -> GaugeChildStub:
        key = tuple(labels.get(name) for name in self._labelnames)
        if key not in self._metrics:
            self._metrics[key] = GaugeChildStub(0.0)
        return self._metrics[key]


@pytest.mark.unit
def test_record_du_budget_updates_backends_once(monkeypatch: pytest.MonkeyPatch) -> None:
    child_calls: list[float] = []

    class GaugeWithLabels:
        def __init__(self) -> None:
            self.label_calls: list[dict[str, str]] = []
            self.set_calls: list[float] = []

        def labels(self, **labels: str):
            self.label_calls.append(labels)

            class Child:
                def set(self_inner, value: float) -> None:
                    child_calls.append(value)

            return Child()

        def set(self, value: float) -> None:
            self.set_calls.append(value)

    gauge = GaugeWithLabels()
    ledger_calls: list[tuple[str, float]] = []

    def ledger_hook(agent_id: str, remaining: float) -> None:
        ledger_calls.append((agent_id, remaining))

    monkeypatch.setattr(infra_metrics, "_agent_du_budget", {}, raising=False)
    monkeypatch.setattr(infra_metrics.prom_metrics, "AGENT_REMAINING_DU", gauge, raising=False)
    monkeypatch.setattr(
        infra_metrics,
        "ledger",
        SimpleNamespace(record_du_budget=ledger_hook),
    )

    infra_metrics.record_du_budget("agent-42", 12.5)

    assert infra_metrics._agent_du_budget["agent-42"] == pytest.approx(12.5)
    assert child_calls == [12.5]
    assert gauge.set_calls == []
    assert ledger_calls == [("agent-42", 12.5)]


@pytest.mark.unit
def test_get_agent_du_budget_prefers_resource_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyManager:
        def get_du_budget(self, agent_id: str) -> float:
            assert agent_id == "agent-1"
            return 42.5

    manager = DummyManager()
    monkeypatch.setattr(resource_manager_mod, "_resource_manager", manager)
    monkeypatch.setattr(resource_manager_mod, "get_resource_manager", lambda: manager)
    monkeypatch.setattr(
        infra_metrics.prom_metrics,
        "AGENT_REMAINING_DU",
        GaugeStub({("agent-1",): 7.0}),
        raising=False,
    )

    assert infra_metrics.get_agent_du_budget("agent-1") == pytest.approx(42.5)


@pytest.mark.unit
def test_get_agent_du_budget_falls_back_to_gauge(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> resource_manager_mod.ResourceManager:
        raise RuntimeError("no manager")

    monkeypatch.setattr(resource_manager_mod, "_resource_manager", None)
    monkeypatch.setattr(resource_manager_mod, "get_resource_manager", boom)
    monkeypatch.setattr(
        infra_metrics.prom_metrics,
        "AGENT_REMAINING_DU",
        GaugeStub({("agent-2",): 13.37}),
        raising=False,
    )

    assert infra_metrics.get_agent_du_budget("agent-2") == pytest.approx(13.37)


@pytest.mark.unit
def test_get_agent_llm_latency_p95_prefers_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        infra_metrics,
        "_LATENCY_SAMPLES_PER_AGENT",
        {"agent-3": deque([10.0, 20.0, 30.0], maxlen=100)},
        raising=False,
    )
    monkeypatch.setattr(
        infra_metrics.prom_metrics,
        "AGENT_LLM_LATENCY_P95_MS",
        GaugeStub({("agent-3",): 99.0}),
        raising=False,
    )

    assert infra_metrics.get_agent_llm_latency_p95("agent-3") == pytest.approx(20.0)


@pytest.mark.unit
def test_get_agent_llm_latency_p95_falls_back_to_gauge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        infra_metrics,
        "_LATENCY_SAMPLES_PER_AGENT",
        {},
        raising=False,
    )
    monkeypatch.setattr(
        infra_metrics.prom_metrics,
        "AGENT_LLM_LATENCY_P95_MS",
        GaugeStub({("agent-4",): 55.5}),
        raising=False,
    )

    assert infra_metrics.get_agent_llm_latency_p95("agent-4") == pytest.approx(55.5)
