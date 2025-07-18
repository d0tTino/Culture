import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.infra import config
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
    with (
        patch("src.interfaces.discord_bot.discord.Client", DummyClient),
        patch(
            "src.interfaces.discord_bot.discord.DiscordException",
            DummyException,
        ),
    ):
        bot = SimulationDiscordBot("token", 999)
        bot.is_ready = True
        error_called = asyncio.Event()

        def fake_error(msg: str, exc_info: bool = False) -> None:
            error_called.set()

        monkeypatch.setattr("src.interfaces.discord_bot.logger.error", fake_error)
        monkeypatch.setitem(config._CONFIG, "OPA_URL", "")
        monkeypatch.setattr(
            "src.interfaces.discord_bot.evaluate_with_opa",
            AsyncMock(return_value=(True, "hi")),
        )
        result = await bot.send_simulation_update(content="hi")
        assert result is False
        assert error_called.is_set()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_bot_retries_on_start_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: int = 0

    class FailingClient(DummyClient):
        async def start(self, token: str) -> None:  # type: ignore[override]
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise DummyException("fail")

    sleep_mock = AsyncMock()

    with (
        patch("src.interfaces.discord_bot.discord.Client", FailingClient),
        patch("src.interfaces.discord_bot.discord.DiscordException", DummyException),
        patch("asyncio.sleep", sleep_mock),
    ):
        bot = SimulationDiscordBot("token", 123)
        tasks = bot.run_bot()
        await asyncio.gather(*tasks[:-1])
        await bot.stop_bot()

    assert attempts == 2
    sleep_mock.assert_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_bot_cancels_pending_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    start_event = asyncio.Event()

    class HangingClient(DummyClient):
        async def start(self, token: str) -> None:  # type: ignore[override]
            await start_event.wait()

        def get_channel(self, channel_id: int) -> DummyChannel:
            class DummyNoFailChannel:
                async def send(self, *args: object, **kwargs: object) -> None:
                    pass

            return DummyNoFailChannel()

        async def close(self) -> None:
            pass

    with (
        patch("src.interfaces.discord_bot.discord.Client", HangingClient),
        patch("src.interfaces.discord_bot.discord.DiscordException", DummyException),
    ):
        bot = SimulationDiscordBot("token", 123)
        bot.message_queue = asyncio.Queue()
        tasks = bot.run_bot()
        await asyncio.sleep(0)
        await bot.stop_bot()
        assert all(t.done() for t in bot._client_tasks)
        assert bot._forward_task is None or bot._forward_task.done()
