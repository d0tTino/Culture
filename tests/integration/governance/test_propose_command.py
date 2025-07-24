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
    monkeypatch.setitem(
        bot.DEFAULT_CONTEXT.sim_state,
        "discord_bot",
        SimpleNamespace(channel_to_agent={123: "a1"}, context=bot.DEFAULT_CONTEXT),
    )

    await bot.slash_propose.callback(DummyInteraction(), text="hello")

    assert DummyClient.called["url"].endswith("/api/propose")
    assert DummyClient.called["json"] == {"proposer_id": "a1", "text": "hello"}
    DummyClient.called = {}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_slash_propose_law_uses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class LawClient:
        called: dict | None = None

        async def __aenter__(self) -> "LawClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            pass

        async def post(self, url: str, json: dict[str, object]) -> object:
            LawClient.called = {"url": url, "json": json}

            class Resp:
                def json(self_inner) -> dict[str, object]:
                    return {"approved": True}

            return Resp()

    monkeypatch.setattr(bot.httpx, "AsyncClient", lambda *a, **k: LawClient())
    monkeypatch.setattr(bot.ledger, "get_balance_async", AsyncMock(return_value=(1.0, 1.0)))
    monkeypatch.setitem(
        bot.DEFAULT_CONTEXT.sim_state,
        "discord_bot",
        SimpleNamespace(channel_to_agent={123: "a1"}, context=bot.DEFAULT_CONTEXT),
    )

    await bot.slash_propose_law.callback(DummyInteraction(), text="hello")

    assert LawClient.called["url"].endswith("/api/propose_law")
    assert LawClient.called["json"] == {"proposer_id": "a1", "text": "hello"}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_slash_vote_uses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class VoteClient:
        called: dict | None = None

        async def __aenter__(self) -> "VoteClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            pass

        async def post(self, url: str, json: dict[str, object]) -> object:
            VoteClient.called = {"url": url, "json": json}

            class Resp:
                def json(self_inner) -> dict[str, object]:
                    return {"vote": True}

            return Resp()

    monkeypatch.setattr(bot.httpx, "AsyncClient", lambda *a, **k: VoteClient())
    monkeypatch.setattr(bot.ledger, "get_balance_async", AsyncMock(return_value=(1.0, 1.0)))
    monkeypatch.setitem(
        bot.DEFAULT_CONTEXT.sim_state,
        "discord_bot",
        SimpleNamespace(channel_to_agent={123: "a1"}, context=bot.DEFAULT_CONTEXT),
    )

    await bot.slash_vote.callback(DummyInteraction(), text="hello", approve=True)

    assert VoteClient.called["url"].endswith("/api/vote")
    assert VoteClient.called["json"] == {"agent_id": "a1", "text": "hello", "approve": True}
