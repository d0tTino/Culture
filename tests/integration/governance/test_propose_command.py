from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.interfaces import discord_bot as bot


class DummyInteraction:
    def __init__(self) -> None:
        self.response = SimpleNamespace(send_message=AsyncMock())
        self.channel = SimpleNamespace(id=123)


class DummyClient:
    called: dict | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def post(self, url: str, json: dict[str, object]) -> object:
        DummyClient.called = {"url": url, "json": json}

        class Resp:
            def json(self_inner):
                return {"approved": True}

        return Resp()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_slash_propose_uses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bot.httpx, "AsyncClient", lambda *a, **k: DummyClient())
    monkeypatch.setattr(bot.ledger, "get_balance_async", AsyncMock(return_value=(1.0, 1.0)))
    monkeypatch.setattr(bot, "active_bot", SimpleNamespace(channel_to_agent={123: "a1"}))

    await bot.slash_propose.callback(DummyInteraction(), text="hello")

    assert DummyClient.called["url"].endswith("/api/propose")
    assert DummyClient.called["json"] == {"proposer_id": "a1", "text": "hello"}
    DummyClient.called = {}
