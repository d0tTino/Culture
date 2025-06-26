"""
Discord bot interface for the Culture simulation.
Provides real-time updates about the simulation to a Discord channel.
"""

# ruff: noqa: ANN401

import asyncio
import logging
import typing
from typing import TYPE_CHECKING, Any, Optional

from src.infra import config
from src.interfaces import metrics
from src.utils.policy import allow_message, evaluate_with_opa

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import discord
    from discord.ext import commands
else:  # pragma: no cover - runtime import with fallback
    try:
        import discord  # type: ignore
        from discord.ext import commands  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        from unittest.mock import MagicMock

        discord = MagicMock()
        commands = MagicMock()
from typing_extensions import Self

logger = logging.getLogger(__name__)


class SimulationDiscordBot:
    """
    A Discord bot that provides real-time updates about the Culture simulation.

    This bot connects to a specified Discord channel and sends read-only updates
    about simulation events, including Knowledge Board updates, agent messages,
    role changes, and other significant state changes.
    """

    def __init__(
        self: Self,
        bot_token: str | list[str] | None,
        channel_id: int,
        token_lookup: (
            Optional[typing.Callable[[str], typing.Awaitable[str | None] | str]] | None
        ) = None,
    ) -> None:
        """
        Initialize the Discord bot with token and target channel.

        Args:
            bot_token (str | list[str] | None): Discord bot token(s) or ``None`` to
                load from the database.
            channel_id (int): The ID of the Discord channel to send updates to
        """
        tokens: list[str] = []
        if bot_token:
            tokens = [bot_token] if isinstance(bot_token, str) else list(bot_token)
        else:
            db_url = str(config.get_config("DISCORD_TOKENS_DB_URL") or "")
            if db_url:
                try:
                    from .token_store import list_tokens
                except ImportError:
                    logger.exception("Failed to import token store for loading tokens")
                else:

                    async def _load() -> list[str]:
                        return await list_tokens()

                    try:
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            tokens = asyncio.run(_load())
                        else:
                            new_loop = asyncio.new_event_loop()
                            try:
                                tokens = new_loop.run_until_complete(_load())
                            finally:
                                new_loop.close()
                    except Exception:
                        logger.exception("Failed to load tokens from store")
        if not tokens:
            raise RuntimeError("No Discord bot tokens provided")
        self.bot_tokens = tokens
        self.channel_id = channel_id
        self.is_ready = False
        if token_lookup is None:
            db_url = str(config.get_config("DISCORD_TOKENS_DB_URL") or "")
            if db_url:
                try:
                    from .token_store import lookup_token as db_lookup
                except ImportError:
                    logger.exception("Failed to import token store")
                else:
                    token_lookup = db_lookup
        self.token_lookup = token_lookup

        # Set up intents (permissions)
        intents = discord.Intents.default()
        intents.message_content = True  # Enable if you plan to add commands later

        # Create Discord clients (one per token)
        self.clients: dict[str, Any] = {
            token: discord.Client(intents=intents) for token in self.bot_tokens
        }
        self.client = self.clients[self.bot_tokens[0]]

    async def _select_client(self: Self, agent_id: Optional[str]) -> Any:
        """Return the Discord client for the given agent."""
        if agent_id and self.token_lookup:
            token = self.token_lookup(agent_id)
            if asyncio.iscoroutine(token):
                token = await token
            if isinstance(token, str):
                return self.clients.get(token, self.client)
        return self.client

        # Set up event handlers for the first client only
        @self.client.event
        async def on_ready() -> None:
            """Event handler that fires when the bot connects to Discord."""
            self.is_ready = True
            logger.info(f"Discord bot {self.client.user} connected and ready!")

            # Get the target channel and send a startup message
            channel = self.client.get_channel(self.channel_id)
            if channel:
                embed = discord.Embed(
                    title="🤖 Culture Simulation Bot Online",
                    description="Connected and ready to provide simulation updates!",
                    color=discord.Color.blue(),
                )
                embed.set_footer(text=f"Channel ID: {self.channel_id}")
                if hasattr(channel, "send"):
                    await channel.send(embed=embed)
                else:
                    if channel is not None:
                        chan_id = getattr(channel, "id", "unknown")
                        logger.warning(
                            f"Attempted to send message to channel {chan_id} "
                            f"of type {type(channel).__name__}, which does not support .send()"
                        )
                    else:
                        logger.warning("Attempted to send message to a None channel.")
            else:
                logger.warning(f"Could not find Discord channel with ID: {self.channel_id}")

    async def send_simulation_update(
        self: Self,
        content: Optional[str] = None,
        embed: Optional[Any] = None,
        agent_id: Optional[str] = None,
    ) -> Optional[bool]:
        """
        Send a simulation update message to the configured Discord channel.

        Args:
            content (Optional[str]): The text message content to send
            embed (Optional[Any]): The embed object to send

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
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
            client = await self._select_client(agent_id)
            channel = client.get_channel(self.channel_id)
            if not channel:
                logger.warning(f"Could not find Discord channel with ID: {self.channel_id}")
                return False
            if embed:
                if hasattr(channel, "send"):
                    await channel.send(embed=embed)
                    logger.debug("Sent Discord embed update")
                    return True
                else:
                    if channel is not None:
                        chan_id = getattr(channel, "id", "unknown")
                        logger.warning(
                            f"Attempted to send embed to channel {chan_id} "
                            f"of type {type(channel).__name__}, which does not support .send()"
                        )
                    else:
                        logger.warning("Attempted to send embed to a None channel.")
                    return False
            elif content:
                if len(content) > 1990:
                    content = content[:1990] + "..."
                if hasattr(channel, "send"):
                    await channel.send(content)
                    logger.debug(f"Sent Discord text update: {content[:50]}...")
                    return True
                else:
                    if channel is not None:
                        chan_id = getattr(channel, "id", "unknown")
                        logger.warning(
                            f"Attempted to send text to channel {chan_id} "
                            f"of type {type(channel).__name__}, which does not support .send()"
                        )
                    else:
                        logger.warning("Attempted to send text to a None channel.")
                    return False
            else:
                logger.warning("send_simulation_update called with no content or embed")
                return False
        except (discord.DiscordException, OSError) as e:
            logger.error(f"Discord API/network error sending message: {e}", exc_info=True)
            return False
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Unexpected error sending Discord message: {e}", exc_info=True)
            return False

    def create_step_start_embed(self: Self, step: int) -> Any:
        """Creates an embed for simulation step start"""
        embed = discord.Embed(
            title=f"📊 Simulation Step {step} Started", color=discord.Color.blue()
        )
        return embed

    def create_step_end_embed(self: Self, step: int) -> Any:
        """Creates an embed for simulation step end"""
        embed = discord.Embed(
            title=f"✅ Simulation Step {step} Completed", color=discord.Color.green()
        )
        return embed

    def create_knowledge_board_embed(self: Self, agent_id: str, content: str, step: int) -> Any:
        """Creates an embed for Knowledge Board posts"""
        embed = discord.Embed(
            title=f"📝 New Knowledge Board Entry (Step {step})",
            description=f"```{content}```",
            color=discord.Color.gold(),
        )
        embed.set_author(name=f"Posted by Agent {agent_id[:8]}")
        return embed

    def create_role_change_embed(
        self: Self, agent_id: str, old_role: str, new_role: str, step: int
    ) -> Any:
        """Creates an embed for agent role changes"""
        embed = discord.Embed(
            title=f"🔄 Agent Role Change (Step {step})",
            description=f"Agent {agent_id[:8]} changed from **{old_role}** to **{new_role}**",
            color=discord.Color.purple(),
        )
        return embed

    def create_project_embed(
        self: Self, action: str, project_name: str, project_id: str, agent_id: str, step: int
    ) -> Any:
        """Creates an embed for project creation/joining/leaving"""
        if action == "create":
            title = f"🏗️ New Project Created (Step {step})"
            description = (
                f"Agent {agent_id[:8]} created project **{project_name}** (ID: {project_id})"
            )
            color = discord.Color.teal()
        elif action == "join":
            title = f"+ Agent Joined Project (Step {step})"
            description = (
                f"Agent {agent_id[:8]} joined project **{project_name}** (ID: {project_id})"
            )
            color = discord.Color.dark_green()
        elif action == "leave":
            title = f"- Agent Left Project (Step {step})"
            description = (
                f"Agent {agent_id[:8]} left project **{project_name}** (ID: {project_id})"
            )
            color = discord.Color.dark_orange()
        else:
            title = f"🏢 Project Update (Step {step})"
            description = f"Project **{project_name}** (ID: {project_id}) was updated"
            color = discord.Color.light_grey()

        embed = discord.Embed(title=title, description=description, color=color)
        return embed

    def create_agent_message_embed(
        self: Self,
        agent_id: str,
        message_content: str,
        recipient_id: Optional[str] = None,
        action_intent: str = "continue_collaboration",
        agent_role: str = "Unknown",
        mood: str = "neutral",
        step: int = 0,
    ) -> Any:
        """Creates an embed for agent messages (broadcast or targeted)"""
        target_info = f"to Agent {recipient_id[:8]}" if recipient_id else "to All (Broadcast)"

        # Determine color based on action intent
        color = discord.Color.blue()  # Default color
        if action_intent == "propose_idea":
            color = discord.Color.gold()
        elif action_intent == "ask_clarification":
            color = discord.Color.purple()
        elif action_intent == "perform_deep_analysis":
            color = discord.Color.dark_teal()
        elif action_intent == "create_project":
            color = discord.Color.teal()
        elif action_intent == "join_project":
            color = discord.Color.dark_green()
        elif action_intent == "leave_project":
            color = discord.Color.dark_orange()

        embed = discord.Embed(
            title=f"💬 Agent Message (Step {step})",
            description=f"```{message_content}```",
            color=color,
        )
        embed.set_author(name=f"From Agent {agent_id[:8]} {target_info}")
        embed.add_field(name="Role", value=agent_role, inline=True)
        embed.add_field(name="Mood", value=mood, inline=True)
        embed.add_field(name="Intent", value=action_intent, inline=True)
        return embed

    def create_ip_change_embed(
        self: Self, agent_id: str, old_ip: int, new_ip: int, reason: str, step: int
    ) -> Any:
        """Creates an embed for influence point changes"""
        change = new_ip - old_ip
        change_text = f"+{change}" if change > 0 else f"{change}"
        color = discord.Color.green() if change > 0 else discord.Color.red()

        embed = discord.Embed(
            title=f"💰 Influence Points Change (Step {step})",
            description=f"Agent {agent_id[:8]} IP: {old_ip} → {new_ip} ({change_text})",
            color=color,
        )
        embed.add_field(name="Reason", value=reason, inline=False)
        return embed

    def create_du_change_embed(
        self: Self, agent_id: str, old_du: float, new_du: float, reason: str, step: int
    ) -> Any:
        """Creates an embed for decision unit changes"""
        change = new_du - old_du
        change_text = f"+{change:.2f}" if change > 0 else f"{change:.2f}"
        color = discord.Color.green() if change > 0 else discord.Color.red()

        embed = discord.Embed(
            title=f"💾 Decision Units Change (Step {step})",
            description=f"Agent {agent_id[:8]} DU: {old_du:.2f} → {new_du:.2f} ({change_text})",
            color=color,
        )
        embed.add_field(name="Reason", value=reason, inline=False)
        return embed

    def create_agent_action_embed(
        self: Self,
        agent_id: str,
        action_intent: str,
        agent_role: str = "Unknown",
        mood: str = "neutral",
        step: int = 0,
    ) -> Any:
        """Creates an embed for agent actions that don't involve messages"""

        # Determine action description and color based on intent
        if action_intent == "idle":
            action_desc = "is observing"
            color = discord.Color.light_grey()
        elif action_intent == "perform_deep_analysis":
            action_desc = "is performing deep analysis"
            color = discord.Color.dark_teal()
        elif action_intent == "create_project":
            action_desc = "is creating a new project"
            color = discord.Color.teal()
        elif action_intent == "join_project":
            action_desc = "is joining a project"
            color = discord.Color.dark_green()
        elif action_intent == "leave_project":
            action_desc = "is leaving a project"
            color = discord.Color.dark_orange()
        else:
            action_desc = f"is performing action '{action_intent}'"
            color = discord.Color.blue()

        embed = discord.Embed(
            title=f"🔄 Agent Action (Step {step})",
            description=f"Agent {agent_id[:8]} {action_desc}",
            color=color,
        )
        embed.add_field(name="Role", value=agent_role, inline=True)
        embed.add_field(name="Mood", value=mood, inline=True)
        embed.add_field(name="Intent", value=action_intent, inline=True)
        return embed

    def create_map_action_embed(
        self: Self,
        agent_id: str,
        action: str,
        details: dict[str, Any],
        step: int,
    ) -> Any:
        """Creates an embed describing a world map action."""

        color = discord.Color.blue()
        if action == "move":
            pos = details.get("position")
            desc = f"Agent {agent_id[:8]} moved to {pos}"
        elif action == "gather":
            resource = details.get("resource")
            success = details.get("success")
            desc = (
                f"Agent {agent_id[:8]} gathered {resource}"
                if success
                else f"Agent {agent_id[:8]} failed to gather {resource}"
            )
            color = discord.Color.green() if success else discord.Color.red()
        elif action == "build":
            structure = details.get("structure")
            success = details.get("success")
            desc = (
                f"Agent {agent_id[:8]} built {structure}"
                if success
                else f"Agent {agent_id[:8]} failed to build {structure}"
            )
            color = discord.Color.dark_orange() if success else discord.Color.red()
        else:
            desc = f"Agent {agent_id[:8]} performed {action}"

        embed = discord.Embed(
            title=f"🗺️ Map Action (Step {step})",
            description=desc,
            color=color,
        )
        return embed

    async def run_bot(self: Self) -> None:
        """
        Start the Discord bot and connect to Discord.
        This is a blocking call that should be run in an asyncio task.
        """
        try:
            logger.info(f"Starting Discord bot(s), connecting to channel ID: {self.channel_id}")
            tasks = []
            for token, client in self.clients.items():
                setattr(client, "token", token)
                tasks.append(client.start(token))
            await asyncio.gather(*tasks)
            # Mark bot as ready when all clients have started
            self.is_ready = True
        except (discord.DiscordException, OSError) as e:
            logger.error(f"Error starting Discord bot: {e}")

    async def stop_bot(self: Self) -> None:
        """Stop the Discord bot and close the connection."""
        try:
            logger.info("Stopping Discord bot...")
            if self.is_ready:
                channel = self.client.get_channel(self.channel_id)
                if channel and len(self.bot_tokens) == 1:
                    embed = discord.Embed(
                        title="🛑 Simulation Complete",
                        description="The Culture simulation has ended. Bot going offline.",
                        color=discord.Color.red(),
                    )
                    if hasattr(channel, "send"):
                        await channel.send(embed=embed)
                    else:
                        if channel is not None:
                            chan_id = getattr(channel, "id", "unknown")
                            logger.warning(
                                f"Attempted to send message to channel {chan_id} "
                                f"of type {type(channel).__name__}, which does not support .send()"
                            )
                        else:
                            logger.warning("Attempted to send message to a None channel.")
            for client in self.clients.values():
                await client.close()
            self.is_ready = False
            logger.info("Discord bot stopped")
        except (discord.DiscordException, OSError) as e:
            logger.error(f"Error stopping Discord bot: {e}")


# --- Minimal command interface for manual testing ---

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


def get_llm_latency() -> float:
    return metrics.get_llm_latency()


def get_kb_size() -> int:
    return metrics.get_kb_size()


@typing.no_type_check
@bot.command(name="say")
async def say(ctx: Any, *, message: str) -> None:
    """Echo a user-provided message for smoke testing."""
    await ctx.send(f"Simulated message received: {message}")


@typing.no_type_check
@bot.command(name="stats")
async def stats(ctx: Any) -> None:
    """Return basic runtime statistics."""
    stats_text = f"LLM latency: {get_llm_latency()} ms; KB size: {get_kb_size()}"
    await ctx.send(stats_text)
