from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infra import config
from src.interfaces.discord_bot import SimulationDiscordBot


class DummyChannel:
    def __init__(self) -> None:
        self.send = AsyncMock()


class DummyDiscordClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.event = lambda fn: fn
        self.user = "dummy"

    def get_channel(self, channel_id: int) -> DummyChannel:
        return DummyChannel()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_opa_blocks_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(config._CONFIG, "OPA_URL", "http://opa")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"result": {"allow": False}}
    monkeypatch.setattr("src.utils.policy.requests.post", MagicMock(return_value=mock_resp))
    with (
        patch("src.interfaces.discord_bot.discord.Client", DummyDiscordClient),
        patch("src.interfaces.discord_bot.discord.TextChannel", DummyChannel),
        patch("src.interfaces.discord_bot.discord.Thread", DummyChannel),
        patch("src.interfaces.discord_bot.discord.DiscordException", Exception),
    ):
        bot = SimulationDiscordBot("token", 1)
        bot.is_ready = True
        channel = DummyChannel()
        bot.client.get_channel = MagicMock(return_value=channel)
        result = await bot.send_simulation_update(content="hi")
        assert result is False
        channel.send.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_opa_modifies_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(config._CONFIG, "OPA_URL", "http://opa")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"result": {"allow": True, "content": "bar"}}
    monkeypatch.setattr("src.utils.policy.requests.post", MagicMock(return_value=mock_resp))
    with (
        patch("src.interfaces.discord_bot.discord.Client", DummyDiscordClient),
        patch("src.interfaces.discord_bot.discord.TextChannel", DummyChannel),
        patch("src.interfaces.discord_bot.discord.Thread", DummyChannel),
        patch("src.interfaces.discord_bot.discord.DiscordException", Exception),
    ):
        bot = SimulationDiscordBot("token", 1)
        bot.is_ready = True
        channel = DummyChannel()
        bot.client.get_channel = MagicMock(return_value=channel)
        result = await bot.send_simulation_update(content="foo")
        assert result is True
        channel.send.assert_awaited_once_with("bar")
