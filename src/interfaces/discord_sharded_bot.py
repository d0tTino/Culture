"""Sharded Discord bot interface using discord.py."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.interfaces import metrics
from src.utils.policy import allow_message, evaluate_with_opa

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import discord
    from discord.ext import commands
else:  # pragma: no cover - runtime import with fallback
    try:
        import discord  # type: ignore
        from discord.ext import commands  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        from unittest.mock import MagicMock

        discord = MagicMock()
        commands = MagicMock()

from typing_extensions import Self

logger = logging.getLogger(__name__)


class SimulationShardedDiscordBot:
    """Discord bot that uses the AutoShardedClient."""

    def __init__(self: Self, bot_token: str, channel_id: int) -> None:
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.is_ready = False

        intents = discord.Intents.default()
        intents.message_content = True
        self.client: Any = discord.AutoShardedClient(intents=intents)

        @self.client.event
        async def on_ready() -> None:  # pragma: no cover - minimal
            self.is_ready = True
            logger.info(f"Discord bot {self.client.user} connected and ready!")
            channel = self.client.get_channel(self.channel_id)
            if channel and hasattr(channel, "send"):
                embed = discord.Embed(
                    title="🤖 Culture Simulation Bot Online",
                    description="Connected and ready to provide simulation updates!",
                    color=discord.Color.blue(),
                )
                embed.set_footer(text=f"Channel ID: {self.channel_id}")
                await channel.send(embed=embed)

    async def send_simulation_update(
        self: Self,
        content: str | None = None,
        embed: Any | None = None,
    ) -> bool | None:
        if not self.is_ready:
            logger.warning("Discord bot not ready yet, message not sent")
            return False
        if not allow_message(content):
            logger.debug("Message blocked by policy")
            return False
        if content is not None:
            allowed, content = await evaluate_with_opa(content)
            if not allowed:
                logger.debug("Message blocked by OPA policy")
                return False
        try:
            channel = self.client.get_channel(self.channel_id)
            if not channel:
                logger.warning(f"Could not find Discord channel with ID: {self.channel_id}")
                return False
            if embed and hasattr(channel, "send"):
                await channel.send(embed=embed)
                return True
            if content and hasattr(channel, "send"):
                if len(content) > 1990:
                    content = content[:1990] + "..."
                await channel.send(content)
                return True
            logger.warning("send_simulation_update called with no content or embed")
            return False
        except (discord.DiscordException, OSError) as e:  # pragma: no cover - minimal
            logger.error(f"Discord API/network error sending message: {e}")
            return False

    async def run_bot(self: Self) -> None:
        try:
            await self.client.start(self.bot_token)
            # Mark ready after connecting when used in tests without the on_ready event
            self.is_ready = True
        except (discord.DiscordException, OSError) as e:  # pragma: no cover - minimal
            logger.error(f"Error starting Discord bot: {e}")

    async def stop_bot(self: Self) -> None:
        try:
            if self.is_ready:
                channel = self.client.get_channel(self.channel_id)
                if channel and hasattr(channel, "send"):
                    embed = discord.Embed(
                        title="🛑 Simulation Complete",
                        description="The Culture simulation has ended. Bot going offline.",
                        color=discord.Color.red(),
                    )
                    await channel.send(embed=embed)
            await self.client.close()
            self.is_ready = False
        except (discord.DiscordException, OSError) as e:  # pragma: no cover - minimal
            logger.error(f"Error stopping Discord bot: {e}")


intents = discord.Intents.default()
intents.message_content = True
bot = commands.AutoShardedBot(command_prefix="!", intents=intents)


@bot.command(name="say")
async def say(ctx: Any, *, message: str) -> None:  # pragma: no cover - simple
    await ctx.send(f"Simulated message received: {message}")


@bot.command(name="stats")
async def stats(ctx: Any) -> None:  # pragma: no cover - simple
    stats_text = f"LLM latency: {metrics.get_llm_latency()} ms; KB size: {metrics.get_kb_size()}"
    await ctx.send(stats_text)
