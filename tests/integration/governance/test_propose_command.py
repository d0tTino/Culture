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
async def test_slash_propose_law_uses_service(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict | None = None

    async def fake_propose(
        proposer: object, text: str, agents: list[object], vote_weights: object | None = None
    ) -> bool:
        nonlocal called
        called = {
            "proposer": getattr(proposer, "agent_id", None),
            "text": text,
            "agents": [getattr(a, "agent_id", None) for a in agents],
            "vote_weights": vote_weights,
        }
        return True

    monkeypatch.setattr(bot.ledger, "get_balance_async", AsyncMock(return_value=(1.0, 1.0)))
    monkeypatch.setitem(
        bot.DEFAULT_CONTEXT.sim_state,
        "discord_bot",
        SimpleNamespace(channel_to_agent={123: "a1"}, context=bot.DEFAULT_CONTEXT),
    )
    monkeypatch.setitem(
        bot.dashboard_backend.SIM_STATE,
        "simulation",
        SimpleNamespace(agents=[SimpleNamespace(agent_id="a1"), SimpleNamespace(agent_id="a2")]),
    )
    monkeypatch.setattr(bot.governance, "propose_law", fake_propose)

    await bot.slash_propose_law.callback(DummyInteraction(), text="hello")

    assert called == {
        "proposer": "a1",
        "text": "hello",
        "agents": ["a1", "a2"],
        "vote_weights": None,
    }


@pytest.mark.asyncio
@pytest.mark.integration
async def test_slash_vote_uses_service(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict | None = None

    async def fake_vote(agent: object, text: str) -> bool:
        nonlocal called
        called = {
            "agent": getattr(agent, "agent_id", None),
            "text": text,
        }
        return True

    monkeypatch.setattr(bot.ledger, "get_balance_async", AsyncMock(return_value=(1.0, 1.0)))
    monkeypatch.setitem(
        bot.DEFAULT_CONTEXT.sim_state,
        "discord_bot",
        SimpleNamespace(channel_to_agent={123: "a1"}, context=bot.DEFAULT_CONTEXT),
    )
    monkeypatch.setitem(
        bot.dashboard_backend.SIM_STATE,
        "simulation",
        SimpleNamespace(agents=[SimpleNamespace(agent_id="a1")]),
    )
    monkeypatch.setattr(bot.governance, "vote", fake_vote)

    await bot.slash_vote.callback(DummyInteraction(), text="hello", approve=True)

    assert called == {"agent": "a1", "text": "hello"}
