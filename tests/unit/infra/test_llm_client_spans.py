import asyncio
import types

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from src.infra import llm_client
from src.interfaces import metrics


class _InMemoryExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list = []

    def export(self, spans):  # type: ignore[override]
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover - no action needed
        return None


_EXPORTER = _InMemoryExporter()
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(_EXPORTER))


def _setup_tracer() -> _InMemoryExporter:
    _EXPORTER.spans.clear()
    return _EXPORTER


@pytest.mark.unit
def test_vllm_client_span_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = _setup_tracer()

    async def mock_post(self: httpx.AsyncClient, url: str, json: object, timeout: float):
        class Resp:
            status_code = 200
            text = (
                '{"usage":{"prompt_tokens":2,"completion_tokens":3},'
                '"choices":[{"message":{"role":"assistant","content":"ok"}}]}'
            )

            def raise_for_status(self) -> None:
                return None

        return Resp()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    monkeypatch.setattr(
        llm_client,
        "get_config",
        lambda name: {"GAS_PRICE_PER_CALL": 1, "GAS_PRICE_PER_TOKEN": 0.5}.get(name),
    )

    client = llm_client._create_vllm_client()
    before = metrics.LLM_CALLS_TOTAL._value.get()
    asyncio.run(client.async_chat("m", [], {}))
    after = metrics.LLM_CALLS_TOTAL._value.get()
    assert after == before + 1
    span = exporter.spans[-1]
    assert span.attributes["llm.tokens.total"] == 5
    assert span.attributes["llm.du.cost"] == pytest.approx(1 + 0.5 * 5)
    assert metrics.LLM_LATENCY_MS._value.get() > 0


@pytest.mark.unit
def test_ollama_client_span_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = _setup_tracer()

    class DummyClient:
        def chat(self, model: str, messages: list, options: dict | None = None):
            return {
                "message": {"role": "assistant", "content": "ok"},
                "prompt_eval_count": 2,
                "eval_count": 3,
            }

    monkeypatch.setattr(
        llm_client, "ollama", types.SimpleNamespace(Client=lambda host=None: DummyClient())
    )
    monkeypatch.setattr(
        llm_client,
        "get_config",
        lambda name: {"GAS_PRICE_PER_CALL": 1, "GAS_PRICE_PER_TOKEN": 0.5}.get(name),
    )

    client = llm_client._create_ollama_client()
    before = metrics.LLM_CALLS_TOTAL._value.get()
    client.chat("m", [])
    after = metrics.LLM_CALLS_TOTAL._value.get()
    assert after == before + 1
    span = exporter.spans[-1]
    assert span.attributes["llm.tokens.total"] == 5
    assert span.attributes["llm.du.cost"] == pytest.approx(1 + 0.5 * 5)
    assert metrics.LLM_LATENCY_MS._value.get() > 0
