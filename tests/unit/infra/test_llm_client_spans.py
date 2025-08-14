import pytest

from src.infra.llm_client import _create_vllm_client


class MockSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class MockTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name):
        span = MockSpan(name)
        self.spans.append(span)
        return span


class MockAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json, timeout):
        json_module = __import__("json")

        class Resp:
            text = json_module.dumps(
                {
                    "responses": [
                        {
                            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                        }
                    ]
                }
            )

            def raise_for_status(self):
                return None

        return Resp()


@pytest.mark.asyncio
async def test_async_chat_batch_tracing(monkeypatch):
    tracer = MockTracer()
    monkeypatch.setattr("src.infra.llm_client.tracer", tracer)
    monkeypatch.setattr("src.infra.llm_client.httpx.AsyncClient", MockAsyncClient)
    monkeypatch.setattr("src.infra.llm_client.VLLM_API_BASE", "http://test")
    monkeypatch.setattr("src.infra.llm_client.LLM_API_BASE", "http://test")

    client = _create_vllm_client()
    batch = [("model", [{"role": "user", "content": "hi"}], None)]
    result = await client.async_chat_batch(batch)

    assert result[0]["message"]["role"] == "assistant"
    assert tracer.spans[0].name == "llm.batch"
    span = tracer.spans[0]
    assert span.attributes["llm.tokens.prompt"] == 1
    assert span.attributes["llm.tokens.completion"] == 2
    assert span.attributes["llm.tokens.total"] == 3
    assert "llm.latency_ms" in span.attributes
