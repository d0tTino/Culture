from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from src.interfaces import discord_bot as bot


class DummyInteraction:
    def __init__(self) -> None:
        self.response = SimpleNamespace(send_message=AsyncMock())
        self.channel = SimpleNamespace(id=123)


@pytest.fixture
def active_discord_bot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        bot.DEFAULT_CONTEXT.sim_state,
        "discord_bot",
        SimpleNamespace(channel_to_agent={123: "a1"}, context=bot.DEFAULT_CONTEXT),
    )


@pytest.fixture
def funded_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bot.ledger, "get_balance_async", AsyncMock(return_value=(1.0, 1.0)))


@pytest.mark.asyncio
@pytest.mark.integration
async def test_slash_propose_uses_configured_api_base(
    monkeypatch: pytest.MonkeyPatch, active_discord_bot: None, funded_agent: None
) -> None:
    class DummyClient:
        called: dict[str, object] | None = None

        async def __aenter__(self) -> "DummyClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> httpx.Response:
            DummyClient.called = {"url": url, "json": json}
            request = httpx.Request("POST", url)
            return httpx.Response(200, json={"approved": True}, request=request)

    monkeypatch.setattr(bot.httpx, "AsyncClient", lambda *a, **k: DummyClient())
    monkeypatch.setattr(bot.config, "get", lambda key, default=None: "http://config-host:8765")

    await bot.slash_propose.callback(DummyInteraction(), text="hello")

    assert DummyClient.called == {
        "url": "http://config-host:8765/api/governance/propose",
        "json": {"proposer_id": "a1", "text": "hello"},
    }


@pytest.mark.asyncio
@pytest.mark.integration
async def test_slash_propose_law_fallback_uses_simulation_context(
    monkeypatch: pytest.MonkeyPatch, active_discord_bot: None, funded_agent: None
) -> None:
    sim = SimpleNamespace(propose_law=AsyncMock(return_value=True))
    monkeypatch.setitem(bot.DEFAULT_CONTEXT.sim_state, "simulation", sim)

    await bot.slash_propose_law.callback(
        DummyInteraction(), text="hello", weights='{"a2": 2, "a3": 1}'
    )

    sim.propose_law.assert_awaited_once_with("a1", "hello", {"a2": 2, "a3": 1})


@pytest.mark.asyncio
@pytest.mark.integration
async def test_slash_vote_fallback_uses_governance_service(
    monkeypatch: pytest.MonkeyPatch, active_discord_bot: None, funded_agent: None
) -> None:
    voter = SimpleNamespace(agent_id="a1")
    sim = SimpleNamespace(agents=[voter])
    monkeypatch.setitem(bot.DEFAULT_CONTEXT.sim_state, "simulation", sim)

    vote_weighted = AsyncMock(return_value=True)
    from src.governance.service import governance

    monkeypatch.setattr(governance, "vote_weighted", vote_weighted)

    await bot.slash_vote.callback(DummyInteraction(), text="hello", approve=True)

    vote_weighted.assert_awaited_once_with(voter, "hello", 1, True)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_slash_propose_law_invalid_payload_message(
    monkeypatch: pytest.MonkeyPatch, active_discord_bot: None, funded_agent: None
) -> None:
    interaction = DummyInteraction()

    await bot.slash_propose_law.callback(interaction, text="hello", weights="{not-json")

    interaction.response.send_message.assert_awaited_once_with(
        "Invalid request payload. Please verify command arguments and try again.",
        ephemeral=True,
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_slash_vote_network_error_message(
    monkeypatch: pytest.MonkeyPatch, active_discord_bot: None, funded_agent: None
) -> None:
    class FailingClient:
        async def __aenter__(self) -> "FailingClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> object:
            raise httpx.ConnectError("down")

    interaction = DummyInteraction()
    monkeypatch.setattr(bot.httpx, "AsyncClient", lambda *a, **k: FailingClient())
    monkeypatch.setitem(bot.DEFAULT_CONTEXT.sim_state, "simulation", None)

    await bot.slash_vote.callback(interaction, text="hello", approve=True)

    interaction.response.send_message.assert_awaited_once_with(
        "Governance service is unavailable right now. Please try again shortly.",
        ephemeral=True,
    )
