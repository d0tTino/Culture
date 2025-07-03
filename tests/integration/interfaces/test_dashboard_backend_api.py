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
    await queue.put(db.SimulationEvent(event_type="tick", data={"step": 1}))
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
    await queue.put(db.SimulationEvent(event_type="start", data={"step": 2}))
    await queue.put(None)
    await db.websocket_events(ws)
    payload = json.loads(ws.sent[0])
    assert payload["data"]["step"] == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_control_pause_resume() -> None:
    db = load_dashboard_backend()

    db.SIM_STATE["paused"] = False
    resp = await db.control({"command": "pause"})
    data = json.loads(resp.body)
    assert data["paused"] is True
    assert db.SIM_STATE["paused"] is True

    resp = await db.control({"command": "resume"})
    data = json.loads(resp.body)
    assert data["paused"] is False
    assert db.SIM_STATE["paused"] is False


class DummyControlWS:
    def __init__(self, messages: list[str]) -> None:
        self.accepted = False
        self.sent: list[str] = []
        self.messages = messages

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        if self.messages:
            return self.messages.pop(0)
        from src.interfaces import dashboard_backend as db

        raise db.WebSocketDisconnect()

    async def send_text(self, text: str) -> None:
        self.sent.append(text)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ws_control_flow() -> None:
    db = load_dashboard_backend()
    db.SIM_STATE["paused"] = False
    ws = DummyControlWS(
        [
            "notjson",
            json.dumps({"command": "pause"}),
            json.dumps({"command": "resume"}),
        ]
    )
    await db.ws_control(ws)

    assert ws.accepted is True
    assert json.loads(ws.sent[0])["error"] == "invalid"
    assert json.loads(ws.sent[1])["paused"] is True
    assert json.loads(ws.sent[2])["paused"] is False
    assert db.SIM_STATE["paused"] is False
