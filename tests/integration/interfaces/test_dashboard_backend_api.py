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

    await db.event_queue.put(db.SimulationEvent(event_type="tick", data={"step": 1}))
    await db.event_queue.put(None)
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
    await db.event_queue.put(db.SimulationEvent(event_type="start", data={"step": 2}))
    await db.event_queue.put(None)
    await db.websocket_events(ws)
    payload = json.loads(ws.sent[0])
    assert payload["data"]["step"] == 2
