"""
Discord bot interface for the Culture simulation.
Provides real-time updates about the simulation to a Discord channel.
"""

# ruff: noqa: ANN401

import logging
import typing
from typing import TYPE_CHECKING, Any, Optional

from src.interfaces import metrics

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


class SimulationDiscordBot:
    """
    A Discord bot that provides real-time updates about the Culture simulation.

    This bot connects to a specified Discord channel and sends read-only updates
    about simulation events, including Knowledge Board updates, agent messages,
    role changes, and other significant state changes.
    """

    def __init__(self: Self, bot_token: str, channel_id: int) -> None:
        """
        Initialize the Discord bot with token and target channel.

        Args:
            bot_token (str): The Discord bot token for authentication
            channel_id (int): The ID of the Discord channel to send updates to
        """
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.is_ready = False

        # Set up intents (permissions)
        intents = discord.Intents.default()
        intents.message_content = True  # Enable if you plan to add commands later

        # Create Discord client
        self.client: Any = discord.Client(intents=intents)

        # Set up event handlers
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
                if isinstance(channel, (discord.TextChannel, discord.Thread)):
                    await channel.send(embed=embed)
                else:
                    if channel is not None:
                        logger.warning(
                            f"Attempted to send message to channel {channel.id} "
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
        try:
            channel = self.client.get_channel(self.channel_id)
            if not channel:
                logger.warning(f"Could not find Discord channel with ID: {self.channel_id}")
                return False
            if embed:
                if isinstance(channel, (discord.TextChannel, discord.Thread)):
                    await channel.send(embed=embed)
                    logger.debug("Sent Discord embed update")
                    return True
                else:
                    if channel is not None:
                        logger.warning(
                            f"Attempted to send embed to channel {channel.id} "
                            f"of type {type(channel).__name__}, which does not support .send()"
                        )
                    else:
                        logger.warning("Attempted to send embed to a None channel.")
                    return False
            elif content:
                if len(content) > 1990:
                    content = content[:1990] + "..."
                if isinstance(channel, (discord.TextChannel, discord.Thread)):
                    await channel.send(content)
                    logger.debug(f"Sent Discord text update: {content[:50]}...")
                    return True
                else:
                    if channel is not None:
                        logger.warning(
                            f"Attempted to send text to channel {channel.id} "
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

    async def run_bot(self: Self) -> None:
        """
        Start the Discord bot and connect to Discord.
        This is a blocking call that should be run in an asyncio task.
        """
        try:
            logger.info(f"Starting Discord bot, connecting to channel ID: {self.channel_id}")
            await self.client.start(self.bot_token)
        except (discord.DiscordException, OSError) as e:
            logger.error(f"Error starting Discord bot: {e}")

    async def stop_bot(self: Self) -> None:
        """Stop the Discord bot and close the connection."""
        try:
            logger.info("Stopping Discord bot...")
            if self.is_ready:
                channel = self.client.get_channel(self.channel_id)
                if channel:
                    embed = discord.Embed(
                        title="🛑 Simulation Complete",
                        description="The Culture simulation has ended. Bot going offline.",
                        color=discord.Color.red(),
                    )
                    if isinstance(channel, (discord.TextChannel, discord.Thread)):
                        await channel.send(embed=embed)
                    else:
                        if channel is not None:
                            logger.warning(
                                f"Attempted to send message to channel {channel.id} "
                                f"of type {type(channel).__name__}, which does not support .send()"
                            )
                        else:
                            logger.warning("Attempted to send message to a None channel.")
            await self.client.close()
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
