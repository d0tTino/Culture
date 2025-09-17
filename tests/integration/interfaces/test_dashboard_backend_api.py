import json

import pytest


class DummyRequest:
    async def is_disconnected(self) -> bool:  # pragma: no cover - simple stub
        return False


class DummyWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.sent: list[str] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, text: str) -> None:
        self.sent.append(text)


def load_dashboard_backend():
    import importlib
    import sys
    import types

    if "fastapi" in sys.modules:
        fastapi_mod = sys.modules["fastapi"]
    else:
        fastapi_mod = types.ModuleType("fastapi")
        sys.modules["fastapi"] = fastapi_mod
    if not hasattr(fastapi_mod, "FastAPI") or not hasattr(getattr(fastapi_mod, "FastAPI"), "post"):

        class _FastAPI:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def get(self, *args: object, **kwargs: object):
                def dec(fn):
                    return fn

                return dec

            def post(self, *args: object, **kwargs: object):
                def dec(fn):
                    return fn

                return dec

            def websocket(self, *args: object, **kwargs: object):
                def dec(fn):
                    return fn

                return dec

        fastapi_mod.FastAPI = _FastAPI
        fastapi_mod.Request = object
        fastapi_mod.Response = object
        fastapi_mod.WebSocket = object
        fastapi_mod.WebSocketDisconnect = Exception

        class _JSONResponse:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.body = json.dumps(args[0]).encode() if args else b""

        responses_mod = types.ModuleType("fastapi.responses")
        responses_mod.JSONResponse = _JSONResponse
        sys.modules["fastapi.responses"] = responses_mod
    if "src.interfaces.dashboard_backend" in sys.modules:
        del sys.modules["src.interfaces.dashboard_backend"]
    return importlib.import_module("src.interfaces.dashboard_backend")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_events_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    if "fastapi" in sys.modules:
        fastapi_mod = sys.modules["fastapi"]
    else:
        fastapi_mod = types.ModuleType("fastapi")
        sys.modules["fastapi"] = fastapi_mod
    if not hasattr(fastapi_mod, "WebSocket"):

        class _WS:
            async def accept(self) -> None:
                pass

            async def send_text(self, text: str) -> None:
                pass

        class _FastAPI:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def get(self, *args: object, **kwargs: object):
                def dec(fn):
                    return fn

                return dec

            def post(self, *args: object, **kwargs: object):
                def dec(fn):
                    return fn

                return dec

            def websocket(self, *args: object, **kwargs: object):
                def dec(fn):
                    return fn

                return dec

        fastapi_mod.FastAPI = _FastAPI
        fastapi_mod.Request = DummyRequest
        fastapi_mod.Response = object
        fastapi_mod.WebSocket = _WS
        fastapi_mod.WebSocketDisconnect = Exception
    from src import http_app
    from src.interfaces import dashboard_backend as db

    captured: list[dict[str, str]] = []

    class CaptureESR:
        def __init__(self, gen: object) -> None:
            self.gen = gen

    monkeypatch.setattr(http_app, "EventSourceResponse", CaptureESR)

    queue = db.get_event_queue()
    await queue.put(db.SimulationEvent(type="tick", data={"step": 1}))
    await queue.put(None)
    resp = await http_app.stream_events(DummyRequest())
    event = await resp.gen.__anext__()
    captured.append(event)
    assert json.loads(captured[0]["data"])["data"]["step"] == 1
    with pytest.raises(StopAsyncIteration):
        await resp.gen.__anext__()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_websocket_events() -> None:
    import sys
    import types

    if "fastapi" in sys.modules:
        fastapi_mod = sys.modules["fastapi"]
    else:
        fastapi_mod = types.ModuleType("fastapi")
        sys.modules["fastapi"] = fastapi_mod
    if not hasattr(fastapi_mod, "WebSocket"):

        class _WS:
            async def accept(self) -> None:
                pass

            async def send_text(self, text: str) -> None:
                pass

        class _FastAPI:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def get(self, *args: object, **kwargs: object):
                def dec(fn):
                    return fn

                return dec

            def post(self, *args: object, **kwargs: object):
                def dec(fn):
                    return fn

                return dec

            def websocket(self, *args: object, **kwargs: object):
                def dec(fn):
                    return fn

                return dec

        fastapi_mod.FastAPI = _FastAPI
        fastapi_mod.Request = DummyRequest
        fastapi_mod.Response = object
        fastapi_mod.WebSocket = _WS
        fastapi_mod.WebSocketDisconnect = Exception
    from src.interfaces import dashboard_backend as db

    ws = DummyWebSocket()
    queue = db.get_event_queue()
    await queue.put(db.SimulationEvent(type="start", data={"step": 2}))
    await queue.put(None)
    await db.websocket_events(ws)
    payload = json.loads(ws.sent[0])
    assert payload["data"]["step"] == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_token_balances_includes_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.interfaces import dashboard_backend as db

    class _Cursor:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self._rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return list(self._rows)

    class _Conn:
        def execute(self, query: str, params: tuple[object, ...] = ()) -> _Cursor:
            if "FROM agent_balances" in query:
                return _Cursor([("agent-1", 1.0, 2.0)])
            if "FROM agent_tokens" in query:
                return _Cursor([("agent-1", "token-a", 3)])
            raise AssertionError(f"Unexpected query: {query}")

    calls: list[tuple[str, str]] = []

    def fake_budget(agent_id: str) -> float:
        calls.append(("budget", agent_id))
        return 7.5

    def fake_latency(agent_id: str) -> float:
        calls.append(("latency", agent_id))
        return 123.4

    monkeypatch.setattr(db.ledger, "conn", _Conn(), raising=False)
    monkeypatch.setattr(db.infra_metrics, "get_agent_du_budget", fake_budget)
    monkeypatch.setattr(db.infra_metrics, "get_agent_llm_latency_p95", fake_latency)

    resp = await db.api_token_balances()
    payload = json.loads(resp.body)

    agent = payload["agents"]["agent-1"]
    assert agent["remaining_du_budget"] == pytest.approx(7.5)
    assert agent["llm_latency_p95_ms"] == pytest.approx(123.4)
    assert agent["tokens"]["token-a"] == 3
    assert ("budget", "agent-1") in calls
    assert ("latency", "agent-1") in calls


@pytest.mark.integration
@pytest.mark.asyncio
async def test_message_queue_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from src.interfaces import dashboard_backend as db

    queue: asyncio.Queue[db.AgentMessage] = asyncio.Queue(maxsize=2)
    monkeypatch.setattr(db, "message_sse_queue", queue)

    msg1 = db.AgentMessage(agent_id="a", content="1", step=1)
    msg2 = db.AgentMessage(agent_id="a", content="2", step=2)
    msg3 = db.AgentMessage(agent_id="a", content="3", step=3)

    await db.enqueue_message(msg1)
    await db.enqueue_message(msg2)
    await db.enqueue_message(msg3)

    assert queue.qsize() == 2
    remaining = [queue.get_nowait().content for _ in range(queue.qsize())]
    assert remaining == ["2", "3"]
