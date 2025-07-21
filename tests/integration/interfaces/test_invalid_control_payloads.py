import json

import pytest

from src.interfaces import dashboard_backend as db


class DummyRequest:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def body(self) -> bytes:
        return self._payload


class DummyWebSocket:
    def __init__(self, messages):
        self.messages = messages
        self.accepted = False
        self.sent = []
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        if self.messages:
            return self.messages.pop(0)
        raise db.WebSocketDisconnect()

    async def send_text(self, text: str):
        self.sent.append(text)

    async def close(self):
        self.closed = True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_control_invalid_json() -> None:
    req = DummyRequest(b"{")
    resp = await db.control(req)
    assert json.loads(resp.body) == {"error": "invalid"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_control_invalid_structure() -> None:
    req = DummyRequest(json.dumps([]).encode())
    resp = await db.control(req)
    assert json.loads(resp.body) == {"error": "invalid"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ws_control_invalid_json() -> None:
    ws = DummyWebSocket(["{"])
    await db.ws_control(ws)
    assert json.loads(ws.sent[0]) == {"error": "invalid"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ws_control_invalid_structure() -> None:
    ws = DummyWebSocket(["[]"])
    await db.ws_control(ws)
    assert json.loads(ws.sent[0]) == {"error": "invalid"}
