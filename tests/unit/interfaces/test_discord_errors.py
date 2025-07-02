import asyncio
from unittest.mock import patch

import pytest

from src.interfaces.discord_bot import SimulationDiscordBot


class DummyChannel:
    async def send(self, *args: object, **kwargs: object) -> None:
        raise DummyException("fail")


class DummyClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.event = lambda fn: fn
        self.user = "dummy"

    def get_channel(self, channel_id: int) -> DummyChannel:
        return DummyChannel()


class DummyException(Exception):
    pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_send_simulation_update_logs_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with patch("src.interfaces.discord_bot.discord.Client", DummyClient), patch(
        "src.interfaces.discord_bot.discord.DiscordException",
        DummyException,
    ):
        bot = SimulationDiscordBot("token", 999)
        bot.is_ready = True
        error_called = asyncio.Event()

        def fake_error(msg: str, exc_info: bool = False) -> None:
            error_called.set()

        monkeypatch.setattr("src.interfaces.discord_bot.logger.error", fake_error)
        result = await bot.send_simulation_update(content="hi")
        assert result is False
        assert error_called.is_set()

