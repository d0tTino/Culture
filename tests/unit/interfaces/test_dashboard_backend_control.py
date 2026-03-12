import importlib
import json
import sys
import types

import pytest


def load_dashboard_backend():
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


class DummyRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload

    async def body(self) -> bytes:
        return json.dumps(self.payload).encode()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_control_pause_resume() -> None:
    db = load_dashboard_backend()
    resp = await db.control(DummyRequest({"command": "pause"}))
    data = json.loads(resp.body)
    assert data["paused"] is True

    resp = await db.control(DummyRequest({"command": "set_speed", "value": 2}))
    data = json.loads(resp.body)
    assert data["speed"] == 2

    resp = await db.control(DummyRequest({"command": "resume"}))
    data = json.loads(resp.body)
    assert data["paused"] is False


class DummyWS:
    def __init__(self, messages: list[str]) -> None:
        self.accepted = False
        self.sent: list[str] = []
        self.messages = messages

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        if self.messages:
            return self.messages.pop(0)
        raise load_dashboard_backend().WebSocketDisconnect()

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def close(self) -> None:
        pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ws_control() -> None:
    db = load_dashboard_backend()
    ws = DummyWS(['{"command":"pause"}'])
    await db.ws_control(ws)
    assert ws.accepted is True
    assert json.loads(ws.sent[0])["paused"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_speed_invalid_value() -> None:
    db = load_dashboard_backend()
    await db.control(DummyRequest({"command": "set_speed", "value": 2}))
    resp = await db.control(DummyRequest({"command": "set_speed", "value": "bad"}))
    data = json.loads(resp.body)
    assert data["speed"] == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dashboard_help_and_onboarding_endpoints() -> None:
    db = load_dashboard_backend()
    help_payload = json.loads((await db.api_interaction_help()).body)
    assert "observer" in help_payload["modes"]
    assert "moderator" in help_payload["modes"]

    cards_payload = json.loads((await db.api_onboarding_scenarios()).body)
    assert len(cards_payload["cards"]) >= 3
    assert {"title", "prompt", "recommended_mode"}.issubset(cards_payload["cards"][0].keys())
