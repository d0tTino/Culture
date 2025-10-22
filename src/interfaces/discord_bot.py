"""Discord bot interface for the Culture simulation.

Provides real-time updates about the simulation to a Discord channel and
forwards user messages to :meth:`Simulation._handle_human_command`. Messages
prefixed with ``/broadcast`` will be delivered to all agents, incurring a single
IP/DU cost for the currently active agent.
"""

import asyncio
import json
import logging
import time
import typing
from collections import deque
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import httpx
from opentelemetry import trace
from typing_extensions import Self

from src.app import spawn_agent_command, start_simulation, stop_simulation
from src.infra import config, event_log
from src.infra.ledger import ledger
from src.interfaces import dashboard_backend as db
from src.interfaces import metrics
from src.sim.context import SimulationContext
from src.utils.policy import allow_message, evaluate_with_opa

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import discord
    from discord import app_commands
    from discord.ext import commands
else:  # pragma: no cover - runtime import with fallback
    try:
        import discord
        from discord import app_commands
        from discord.ext import commands
    except ImportError:  # pragma: no cover - optional dependency
        from unittest.mock import MagicMock

        discord = MagicMock()
        commands = MagicMock()
        app_commands = MagicMock()

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

_fallback_context = SimulationContext()
DEFAULT_CONTEXT = cast(SimulationContext, getattr(db, "DEFAULT_CONTEXT", _fallback_context))

_default_snapshot_dir = Path(__file__).resolve().parents[2] / "snapshots"
SNAPSHOT_DIR = cast(Path, getattr(db, "SNAPSHOT_DIR", _default_snapshot_dir))

AgentMessage = getattr(db, "AgentMessage", SimpleNamespace)
SimulationEvent = getattr(db, "SimulationEvent", SimpleNamespace)

dashboard_message_queue = getattr(db, "message_sse_queue", None)
if dashboard_message_queue is None or not hasattr(dashboard_message_queue, "put_nowait"):
    dashboard_message_queue = SimpleNamespace(put_nowait=lambda *a, **k: None)

message_sse_queue = dashboard_message_queue


@contextmanager
def command_span(name: str, interaction: Any, *, agent_id: str | None = None) -> Iterator[Any]:
    start = time.perf_counter()
    with tracer.start_as_current_span("discord.command") as span:
        span.set_attribute("discord.command.name", name)
        channel = getattr(interaction, "channel", None)
        span.set_attribute("discord.channel.id", getattr(channel, "id", None))
        user = getattr(interaction, "user", None)
        span.set_attribute("discord.user.id", getattr(user, "id", None))
        if agent_id is not None:
            span.set_attribute("discord.agent.id", agent_id)
        try:
            yield span
        finally:
            span.set_attribute("discord.latency_ms", (time.perf_counter() - start) * 1000)


@contextmanager
def message_span(message: Any) -> Iterator[Any]:
    start = time.perf_counter()
    with tracer.start_as_current_span("discord.message") as span:
        channel = getattr(message, "channel", None)
        span.set_attribute("discord.channel.id", getattr(channel, "id", None))
        user = getattr(message, "author", None)
        span.set_attribute("discord.user.id", getattr(user, "id", None))
        try:
            yield span
        finally:
            span.set_attribute("discord.latency_ms", (time.perf_counter() - start) * 1000)


async def send_interaction_response(interaction: Any, content: str, **kwargs: Any) -> None:
    """Send a Discord interaction response with tracing."""
    with tracer.start_as_current_span("discord.send_message") as span:
        span.set_attribute("discord.message.length", len(content) if content else 0)
        await interaction.response.send_message(content, **kwargs)


async def send_channel_message(
    channel: Any, *, content: str | None = None, embed: Any | None = None
) -> None:
    """Send a Discord channel message with tracing."""
    with tracer.start_as_current_span("discord.send_message") as span:
        if content is not None:
            span.set_attribute("discord.message.length", len(content))
        span.set_attribute("discord.embed", embed is not None)
        if hasattr(channel, "send"):
            if embed is not None:
                await channel.send(content, embed=embed)
            else:
                await channel.send(content)
        else:  # pragma: no cover - defensive
            chan_id = getattr(channel, "id", "unknown")
            logger.warning(
                f"Attempted to send message to channel {chan_id} of type {type(channel).__name__}, which does not support .send()"
            )


def embed_from_payload(payload: dict[str, Any]) -> Any:
    """Create a ``discord.Embed`` from a payload dictionary."""
    embed = discord.Embed(
        title=payload.get("title"),
        description=payload.get("description"),
        color=payload.get("color"),
    )
    author = payload.get("author")
    if author:
        try:
            embed.set_author(**author)
        except Exception:  # pragma: no cover - best effort
            pass
    for field in payload.get("fields", []):
        try:
            embed.add_field(
                name=field.get("name"),
                value=field.get("value"),
                inline=field.get("inline", True),
            )
        except Exception:  # pragma: no cover - best effort
            pass
    return embed


MAX_EMBED_DESCRIPTION_LENGTH = 4096


def _truncate_for_code_block(content: str, max_length: int = MAX_EMBED_DESCRIPTION_LENGTH) -> str:
    """Wrap ``content`` in a code block, truncating safely to ``max_length`` characters."""

    prefix = "```"
    suffix = "```"
    # Reserve space for the code fence markers so that the block always closes.
    available = max_length - len(prefix) - len(suffix)
    if available <= 0:
        # Fallback: return the maximum slice of repeated backticks to avoid empty output.
        return (prefix + suffix)[:max_length]

    if len(content) > available:
        ellipsis = "…"
        if available >= len(ellipsis):
            truncated = content[: available - len(ellipsis)] + ellipsis
        else:  # pragma: no cover - defensive for pathological limits
            truncated = content[:available]
    else:
        truncated = content

    return f"{prefix}{truncated}{suffix}"


def board_payload_to_embed(payload: dict[str, Any]) -> dict[str, Any]:
    """Map a knowledge board payload to Discord embed fields."""
    agent_id = str(payload.get("agent_id", ""))
    content = str(payload.get("content", ""))
    step = int(payload.get("step", 0))
    return {
        "title": f"📝 New Knowledge Board Entry (Step {step})",
        "description": _truncate_for_code_block(content),
        "color": 0xFFD700,
        "author": {"name": f"Posted by Agent {agent_id[:8]}"},
    }


def simulation_event_to_embed(event: SimulationEvent) -> dict[str, Any] | None:
    """Convert a :class:`SimulationEvent` into an embed payload."""
    data = event.data or {}
    step = int(data.get("step", 0))
    if event.type == "step_start":
        return {
            "title": f"📊 Simulation Step {step} Started",
            "color": 0x0000FF,
        }
    if event.type == "step_end":
        return {
            "title": f"✅ Simulation Step {step} Completed",
            "color": 0x00FF00,
        }
    if event.type == "knowledge_board":
        return board_payload_to_embed(data)
    return None


def _usage_values(agent_id: str | None) -> tuple[float, float, float]:
    """Return IP, DU, and p95 latency metrics for an agent."""
    ip = du = 0.0
    if agent_id:
        try:
            ip, du = ledger.get_balance(agent_id)
        except Exception:  # pragma: no cover - best effort
            ip = du = 0.0
    latency = metrics.get_llm_latency_p95()
    return ip, du, latency


def _usage_fields_from_values(ip: float, du: float, latency: float) -> list[dict[str, Any]]:
    """Create embed fields from usage metrics."""
    return [
        {"name": "IP", "value": f"{ip:.2f}", "inline": True},
        {"name": "DU", "value": f"{du:.2f}", "inline": True},
        {"name": "p95 latency (ms)", "value": f"{latency:.2f}", "inline": True},
    ]


def _usage_fields(agent_id: str | None) -> list[dict[str, Any]]:
    """Convenience wrapper returning usage fields for an agent."""
    ip, du, latency = _usage_values(agent_id)
    return _usage_fields_from_values(ip, du, latency)


def notify_budget_exceeded(agent_id: str, required: float, remaining: float) -> None:
    """Notify via Discord when an agent exceeds its DU budget."""
    msg = (
        f"Agent {agent_id} exceeded DU budget: required {required:.2f}, remaining {remaining:.2f}"
    )
    ip, du, latency = _usage_values(agent_id)
    embed = {
        "title": "❌ DU Budget Exceeded",
        "description": msg,
        "color": 0xFF0000,
        "fields": _usage_fields_from_values(ip, du, latency),
    }
    try:
        message_sse_queue.put_nowait(
            AgentMessage(
                agent_id=agent_id,
                content=msg,
                step=0,
                extra={
                    "ip": ip,
                    "du": du,
                    "p95_latency_ms": latency,
                    "required": required,
                    "remaining": remaining,
                },
            )
        )
    except Exception:  # pragma: no cover - best effort
        logger.exception("Failed to enqueue budget exceeded notification")
    bot_instance = get_active_bot()
    if bot_instance is not None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(bot_instance.send_simulation_update(embed=embed))
            except Exception:  # pragma: no cover - best effort
                logger.exception("Failed to send budget exceeded embed")
        else:
            try:
                loop.create_task(bot_instance.send_simulation_update(embed=embed))  # noqa: RUF006
            except Exception:  # pragma: no cover - best effort
                logger.exception("Failed to send budget exceeded embed")


class SimulationDiscordBot:
    """
    A Discord bot that provides real-time updates about the Culture simulation.

    This bot connects to a specified Discord channel and sends updates about
    simulation events, including Knowledge Board updates, agent messages, role
    changes, and other significant state changes. Incoming Discord messages are
    placed on the shared event queue and ultimately handled by
    ``Simulation._handle_human_command``.
    """

    @classmethod
    async def create(
        cls: type[Self],
        bot_token: str | list[str] | None,
        channel_id: int,
        token_lookup: (
            Optional[typing.Callable[[str], typing.Awaitable[str | None] | str]] | None
        ) = None,
        *,
        channel_map: dict[str, int] | None = None,
        context: SimulationContext | None = None,
    ) -> Self:
        """Asynchronously construct a ``SimulationDiscordBot`` instance."""
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
                    try:
                        tokens = await list_tokens()
                    except Exception:
                        logger.exception("Failed to load tokens from store")

        if not tokens:
            raise RuntimeError("No Discord bot tokens provided")

        ctx = context or DEFAULT_CONTEXT
        return cls(
            tokens,
            channel_id,
            token_lookup=token_lookup,
            channel_map=channel_map,
            context=ctx,
        )

    def __init__(
        self: Self,
        bot_token: str | list[str],
        channel_id: int,
        token_lookup: (
            Optional[typing.Callable[[str], typing.Awaitable[str | None] | str]] | None
        ) = None,
        *,
        channel_map: dict[str, int] | None = None,
        context: SimulationContext = DEFAULT_CONTEXT,
    ) -> None:
        """
        Initialize the Discord bot with token and target channel.

        Args:
            bot_token (str | list[str]): Discord bot token(s).
            channel_id (int): The ID of the Discord channel to send updates to
        """
        tokens = [bot_token] if isinstance(bot_token, str) else list(bot_token)
        if not tokens:
            raise RuntimeError("No Discord bot tokens provided")
        self.bot_tokens = tokens
        self.channel_id = channel_id
        self.channel_map: dict[str, int] = channel_map or {}
        self.channel_to_agent: dict[int, str] = {v: k for k, v in self.channel_map.items()}
        self.user_channels: dict[str, int] = {}
        self.user_agents: dict[str, str] = {}
        self.last_agent_id: str | None = None
        self.last_channel_id: int | None = None
        self.last_user_id: str | None = None
        self.is_ready = False
        self.context = context
        self.context.sim_state["discord_bot"] = self
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
        queue = context._event_queue or db.get_event_queue()
        self.context._event_queue = queue
        if self.context._event_queue_loop is None:
            try:
                self.context._event_queue_loop = asyncio.get_event_loop()
            except RuntimeError:  # pragma: no cover - no running loop
                self.context._event_queue_loop = None
        self.event_queue = queue
        if message_sse_queue is not dashboard_message_queue:
            self.message_queue = message_sse_queue
            self.context.message_queue = self.message_queue
        else:
            self.message_queue = context.message_queue
        self._forward_task: asyncio.Task[Any] | None = None
        self._client_tasks: list[asyncio.Task[Any]] = []
        self.command_trees: dict[str, app_commands.CommandTree] = {}

        # Set up event handlers and slash commands for all clients
        for _token, client in self.clients.items():
            tree: app_commands.CommandTree | None = None
            try:
                if hasattr(client, "http"):
                    tree = app_commands.CommandTree(client)
            except Exception:
                tree = None
            if tree is not None:
                self.command_trees[_token] = tree

                register_moderation_commands(
                    tree,
                    resolve_context=lambda self=self: (self.context, self.event_queue),
                )

                @tree.command(name="start")
                @moderation_rate_limit("start")
                async def _tree_start(interaction: "discord.Interaction") -> None:
                    with command_span("start", interaction) as span:
                        chan = getattr(interaction, "channel", None)
                        chan_id = getattr(chan, "id", None)
                        agent_id = self.channel_to_agent.get(chan_id)
                        if not await _has_control_command_permission(
                            getattr(interaction, "user", None),
                            "start",
                            agent_id=agent_id,
                        ):
                            await interaction.response.send_message(
                                "unauthorized", ephemeral=True
                            )
                            return
                        try:
                            await start_simulation(self.context)
                        except Exception as exc:
                            embed = self.create_start_embed(False, str(exc), agent_id)
                            await self.send_simulation_update(embed=embed)
                            await send_interaction_response(
                                interaction,
                                "",
                                embed=embed_from_payload(embed),
                                ephemeral=True,
                            )
                        else:
                            embed = self.create_start_embed(True, agent_id=agent_id)
                            await self.send_simulation_update(embed=embed)
                            await send_interaction_response(
                                interaction,
                                "",
                                embed=embed_from_payload(embed),
                                ephemeral=True,
                            )

                @tree.command(name="stop")
                @moderation_rate_limit("stop")
                async def _tree_stop(interaction: "discord.Interaction") -> None:
                    with command_span("stop", interaction) as span:
                        chan = getattr(interaction, "channel", None)
                        chan_id = getattr(chan, "id", None)
                        agent_id = self.channel_to_agent.get(chan_id)
                        if not await _has_control_command_permission(
                            getattr(interaction, "user", None),
                            "stop",
                            agent_id=agent_id,
                        ):
                            await interaction.response.send_message(
                                "unauthorized", ephemeral=True
                            )
                            return
                        try:
                            await stop_simulation(self.context)
                        except Exception as exc:
                            embed = self.create_stop_embed(False, str(exc), agent_id)
                            await self.send_simulation_update(embed=embed)
                            await send_interaction_response(
                                interaction,
                                "",
                                embed=embed_from_payload(embed),
                                ephemeral=True,
                            )
                        else:
                            embed = self.create_stop_embed(True, agent_id=agent_id)
                            await self.send_simulation_update(embed=embed)
                            await send_interaction_response(
                                interaction,
                                "",
                                embed=embed_from_payload(embed),
                                ephemeral=True,
                            )

                @tree.command(name="spawn")
                @app_commands.describe(agent_id="ID of the agent to spawn")
                @moderation_rate_limit("spawn")
                async def _tree_spawn(interaction: "discord.Interaction", agent_id: str) -> None:
                    with command_span("spawn", interaction, agent_id=agent_id) as span:
                        if not await _has_control_command_permission(
                            getattr(interaction, "user", None),
                            "spawn",
                            agent_id=agent_id,
                        ):
                            await interaction.response.send_message(
                                "unauthorized", ephemeral=True
                            )
                            return
                        try:
                            await spawn_agent_command(agent_id, self.context)
                        except Exception as exc:
                            embed = self.create_spawn_embed(agent_id, False, str(exc))
                            await self.send_simulation_update(embed=embed)
                            await send_interaction_response(
                                interaction,
                                "",
                                embed=embed_from_payload(embed),
                                ephemeral=True,
                            )
                        else:
                            embed = self.create_spawn_embed(agent_id, True)
                            await self.send_simulation_update(embed=embed)
                            await send_interaction_response(
                                interaction,
                                "",
                                embed=embed_from_payload(embed),
                                ephemeral=True,
                            )

                @tree.command(name="pause")
                async def _tree_pause(interaction: "discord.Interaction") -> None:
                    with command_span("pause", interaction) as span:
                        await self.event_queue.put(
                            SimulationEvent(type="control", data={"command": "pause"})
                        )
                        await interaction.response.send_message("pause", ephemeral=True)

                @tree.command(name="resume")
                async def _tree_resume(interaction: "discord.Interaction") -> None:
                    with command_span("resume", interaction) as span:
                        await self.event_queue.put(
                            SimulationEvent(type="control", data={"command": "resume"})
                        )
                        await interaction.response.send_message("resume", ephemeral=True)

                @tree.command(name="pause_all")
                async def _tree_pause_all(interaction: "discord.Interaction") -> None:
                    with command_span("pause_all", interaction) as span:
                        if not has_admin_permission(getattr(interaction, "user", None)):
                            await interaction.response.send_message("unauthorized", ephemeral=True)
                            return
                        await self.event_queue.put(
                            SimulationEvent(type="control", data={"command": "pause_all"})
                        )
                        await interaction.response.send_message("pause all", ephemeral=True)

                @tree.command(name="kill_agent")
                @app_commands.describe(agent_id="ID of the agent to kill")
                async def _tree_kill_agent(
                    interaction: "discord.Interaction", agent_id: str
                ) -> None:
                    with command_span("kill_agent", interaction, agent_id=agent_id) as span:
                        if not has_admin_permission(getattr(interaction, "user", None)):
                            await interaction.response.send_message("unauthorized", ephemeral=True)
                            return
                        await self.event_queue.put(
                            SimulationEvent(
                                type="control",
                                data={
                                    "command": "kill_agent",
                                    "agent_id": agent_id,
                                },
                            )
                        )
                        await interaction.response.send_message("killed", ephemeral=True)

                @tree.command(name="nudge")
                @app_commands.describe(prompt="Prompt to nudge the simulation")
                @moderation_rate_limit("nudge")
                async def _tree_nudge(interaction: "discord.Interaction", prompt: str) -> None:
                    with command_span("nudge", interaction) as span:
                        span.set_attribute("discord.message.length", len(prompt))
                        await self.event_queue.put(
                            SimulationEvent(type="nudge", data={"prompt": prompt})
                        )
                        await interaction.response.send_message("nudge sent", ephemeral=True)

            @client.event
            async def on_ready(
                client: Any = client, tree: app_commands.CommandTree | None = tree
            ) -> None:
                """Event handler that fires when the bot connects to Discord."""
                self.is_ready = True
                logger.info(f"Discord bot {client.user} connected and ready!")

                # Get the target channel and send a startup message
                channel = client.get_channel(self.channel_id)
                if channel:
                    embed = discord.Embed(
                        title="🤖 Culture Simulation Bot Online",
                        description="Connected and ready to provide simulation updates!",
                        color=discord.Color.blue(),
                    )
                    embed.set_footer(text=f"Channel ID: {self.channel_id}")
                    await send_channel_message(channel, embed=embed)
                else:
                    logger.warning(f"Could not find Discord channel with ID: {self.channel_id}")

                if tree is not None:
                    try:
                        await tree.sync()
                    except Exception:  # pragma: no cover - sync may fail in tests
                        logger.exception("Failed to sync command tree")

            @client.event
            async def on_message(message: Any, client: Any = client) -> None:
                with message_span(message) as span:
                    if getattr(message, "author", None) == client.user:
                        return
                    content = getattr(message, "content", "")
                    span.set_attribute("discord.message.length", len(content))
                    channel = getattr(message, "channel", None)
                    user = getattr(message, "author", None)
                    if not allow_message(content):
                        logger.debug("Message blocked by policy")
                        return
                    allowed, content = await evaluate_with_opa(content)
                    if not allowed:
                        logger.debug("Message blocked by OPA policy")
                        return
                    metrics.HUMAN_MESSAGES_TOTAL.inc()
                    channel_id = getattr(channel, "id", None)
                    user_id = getattr(user, "id", None)
                    if user_id and channel_id:
                        self.user_channels[str(user_id)] = channel_id
                        self.last_user_id = str(user_id)
                    recipient = self.channel_to_agent.get(channel_id)
                    agent_id = None
                    if user_id:
                        agent_id = self.user_agents.get(str(user_id))
                        if agent_id is None and recipient:
                            agent_id = recipient
                            self.user_agents[str(user_id)] = agent_id
                    span.set_attribute("discord.agent.id", agent_id or "")
                    if not agent_id:
                        await send_channel_message(channel, content="Unknown agent mapping")
                        return
                    ip_cost = float(
                        config.get_config("IP_COST_BROADCAST_MESSAGE")
                        or config.get_config("IP_COST_SEND_DIRECT_MESSAGE")
                        or 0.0
                    )
                    du_cost = float(
                        config.get_config("DU_COST_BROADCAST_ACTION")
                        or config.get_config("DU_COST_PER_ACTION")
                        or 0.0
                    )
                    ip_bal, du_bal = await ledger.get_balance_async(agent_id)
                    if ip_bal < ip_cost or du_bal < du_cost:
                        await send_channel_message(channel, content="Insufficient IP/DU")
                        return
                    self.last_agent_id = agent_id
                    self.last_channel_id = channel_id
                    evt_type = "broadcast"
                    data = {"author": agent_id, "content": content}
                    if recipient:
                        data["recipient_id"] = recipient
                    await self.event_queue.put(SimulationEvent(type=evt_type, data=data))

    async def _select_client(self: Self, agent_id: Optional[str]) -> Any:
        """Return the Discord client for the given agent."""
        if agent_id and self.token_lookup:
            token = self.token_lookup(agent_id)
            if asyncio.iscoroutine(token):
                token = await token
            if isinstance(token, str):
                return self.clients.get(token, self.client)
        return self.client

    async def send_simulation_update(
        self: Self,
        content: Optional[str] = None,
        embed: Optional[Any] = None,
        agent_id: Optional[str] = None,
        *,
        target_channel_id: Optional[int] = None,
        recipient: Optional[str] = None,
    ) -> Optional[bool]:
        """
        Send a simulation update message to Discord.

        Args:
            content (Optional[str]): The text message content to send
            embed (Optional[Any]): The embed object to send
            target_channel_id (Optional[int]): Explicit channel to send to
            recipient (Optional[str]): User ID to reply to

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        with tracer.start_as_current_span("discord.send_simulation_update") as span:
            span.set_attribute("discord.agent_id", agent_id or "")
            span.set_attribute("discord.content_length", len(content) if content else 0)
            if target_channel_id is not None:
                span.set_attribute("discord.target_channel_id", target_channel_id)
            if recipient is not None:
                span.set_attribute("discord.recipient_id", recipient)
            if not self.is_ready:
                logger.warning("Discord bot not ready yet, message not sent")
                return False
            if not allow_message(content):
                logger.debug("Message blocked by policy")
                return False
            if content is not None and agent_id is None:
                allowed, content = await evaluate_with_opa(content)
                if not allowed:
                    logger.debug("Message blocked by OPA policy")
                    return False
            try:
                client = await self._select_client(agent_id)
                chan_id = target_channel_id
                if chan_id is None and recipient:
                    chan_id = self.user_channels.get(recipient)
                if chan_id is None:
                    chan_id = self.channel_map.get(agent_id, self.channel_id)
                span.set_attribute("discord.channel_id", chan_id)
                channel = client.get_channel(chan_id)
                if not channel:
                    logger.warning(f"Could not find Discord channel with ID: {chan_id}")
                    return False
                embed_obj = embed
                if isinstance(embed_obj, dict):
                    embed_obj = embed_from_payload(embed_obj)
                if embed_obj:
                    await send_channel_message(channel, embed=embed_obj)
                    logger.debug("Sent Discord embed update")
                    return True
                elif content:
                    if len(content) > 1990:
                        content = content[:1990] + "..."
                    await send_channel_message(channel, content=content)
                    logger.debug(f"Sent Discord text update: {content[:50]}...")
                    return True
                else:
                    logger.warning("send_simulation_update called with no content or embed")
                    return False
            except (discord.DiscordException, OSError) as e:
                logger.error(f"Discord API/network error sending message: {e}", exc_info=True)
                return False
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error(f"Unexpected error sending Discord message: {e}", exc_info=True)
                return False

    def create_start_embed(
        self: Self, success: bool, reason: str | None = None, agent_id: str | None = None
    ) -> dict[str, Any]:
        """Create an embed payload indicating simulation start success or failure."""
        return {
            "title": "✅ Simulation Started" if success else "❌ Simulation Start Failed",
            "description": None if success else reason,
            "color": 0x00FF00 if success else 0xFF0000,
            "fields": _usage_fields(agent_id),
        }

    def create_stop_embed(
        self: Self, success: bool, reason: str | None = None, agent_id: str | None = None
    ) -> dict[str, Any]:
        """Create an embed payload indicating simulation stop success or failure."""
        return {
            "title": "🛑 Simulation Stopped" if success else "❌ Simulation Stop Failed",
            "description": None if success else reason,
            "color": 0x00FF00 if success else 0xFF0000,
            "fields": _usage_fields(agent_id),
        }

    def create_spawn_embed(
        self: Self,
        agent_id: str,
        success: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Create an embed payload indicating agent spawn success or failure."""
        title = (
            f"🚀 Agent {agent_id[:8]} Spawned"
            if success
            else f"❌ Failed to Spawn Agent {agent_id[:8]}"
        )
        return {
            "title": title,
            "description": None if success else reason,
            "color": 0x00FF00 if success else 0xFF0000,
            "fields": _usage_fields(agent_id),
        }

    def create_step_start_embed(self: Self, step: int) -> dict[str, Any]:
        """Create an embed payload for simulation step start."""
        return {
            "title": f"📊 Simulation Step {step} Started",
            "color": 0x0000FF,
        }

    def create_step_end_embed(self: Self, step: int) -> dict[str, Any]:
        """Create an embed payload for simulation step end."""
        return {
            "title": f"✅ Simulation Step {step} Completed",
            "color": 0x00FF00,
        }

    def create_knowledge_board_embed(
        self: Self, agent_id: str, content: str, step: int
    ) -> dict[str, Any]:
        """Create an embed payload for Knowledge Board posts."""
        payload = {
            "agent_id": agent_id,
            "content": content,
            "step": step,
        }
        return board_payload_to_embed(payload)

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

    async def _forward_agent_messages(self: Self) -> None:
        """Forward AgentMessage objects from the queue to Discord."""
        try:
            while True:
                msg: AgentMessage = await self.message_queue.get()
                recipient = msg.recipient_id
                if recipient and recipient not in self.user_channels:
                    for uid, aid in self.user_agents.items():
                        if aid == recipient:
                            recipient = uid
                            break
                embed_payload = msg.extra.get("embed") if msg.extra else None
                await self.send_simulation_update(
                    content=None if embed_payload else msg.content,
                    embed=embed_payload,
                    agent_id=msg.agent_id,
                    recipient=recipient,
                )
        except (
            asyncio.CancelledError,
            RuntimeError,
        ):  # pragma: no cover - task cancelled or loop closed
            pass

    async def _start_client_with_backoff(
        self: Self,
        client: Any,
        token: str,
        max_retries: int = 3,
        base_delay: int = 1,
    ) -> None:
        """Start a Discord client with retries and exponential backoff."""
        for attempt in range(max_retries):
            try:
                await client.start(token)
                return
            except (discord.DiscordException, OSError) as e:  # pragma: no cover - minimal
                logger.error(
                    f"Discord client start failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                await asyncio.sleep(base_delay * (2**attempt))
        logger.error("Max Discord client start attempts exceeded")

    def run_bot(self: Self) -> list[asyncio.Task[Any]]:
        """Start the Discord bot and return running tasks."""
        logger.info(f"Starting Discord bot(s), connecting to channel ID: {self.channel_id}")
        tasks: list[asyncio.Task[Any]] = []
        for token, client in self.clients.items():
            setattr(client, "token", token)
            tasks.append(asyncio.create_task(self._start_client_with_backoff(client, token)))
        self._client_tasks = tasks
        self._forward_task = asyncio.create_task(self._forward_agent_messages())
        self.is_ready = True
        return [*tasks, self._forward_task]

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
                    await send_channel_message(channel, embed=embed)
            for client in self.clients.values():
                await client.close()
            for t in self._client_tasks:
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:  # pragma: no cover - expected
                        pass
            self._client_tasks.clear()
            if self._forward_task:
                self._forward_task.cancel()
                try:
                    await self._forward_task
                except asyncio.CancelledError:  # pragma: no cover - expected
                    pass
                self._forward_task = None
            self.is_ready = False
            self.context.sim_state["discord_bot"] = None
            logger.info("Discord bot stopped")
        except (discord.DiscordException, OSError) as e:
            logger.error(f"Error stopping Discord bot: {e}")


# --- Minimal command interface for manual testing ---

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
if not hasattr(bot, "tree"):
    # Provide a minimal slash command interface when the underlying Bot
    # implementation lacks the ``tree`` attribute (e.g. in unit tests).
    from types import SimpleNamespace

    bot.tree = SimpleNamespace(
        command=lambda *args, **kwargs: (lambda fn: fn),
        add_check=lambda *a, **k: None,
    )  # type: ignore[assignment]


def get_llm_latency() -> float:
    return metrics.get_llm_latency()


def get_kb_size() -> int:
    return metrics.get_kb_size()


def get_active_bot(ctx: SimulationContext = DEFAULT_CONTEXT) -> "SimulationDiscordBot | None":
    """Return the active bot stored in the given context."""
    return cast("SimulationDiscordBot | None", ctx.sim_state.get("discord_bot"))


# --- Command rate limiting -------------------------------------------------

_COMMAND_HISTORY: dict[str, deque[float]] = {}
_COMMAND_LOCKS: dict[str, asyncio.Lock] = {}
_MAX_RATE: int = 5
_DEFAULT_RATE_LIMIT_WINDOW_SECONDS: float = 60.0


def has_admin_permission(user: Any) -> bool:
    """Return True if the Discord user has administrator permissions."""
    perms = getattr(getattr(user, "guild_permissions", None), "administrator", False)
    return bool(perms)


_TRUE_BOOL_VALUES = {"1", "true", "yes", "on"}
_FALSE_BOOL_VALUES = {"0", "false", "no", "off"}


def _coerce_to_bool(value: object) -> bool:
    """Normalize truthy and falsy values retrieved from configuration."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_BOOL_VALUES:
            return True
        if lowered in _FALSE_BOOL_VALUES:
            return False
    return bool(value)


def _allow_control_via_opa() -> bool:
    """Return True when control commands may defer authorization to OPA."""
    overrides = getattr(config, "CONFIG_OVERRIDES", {})
    value = overrides.get("DISCORD_ALLOW_OPA_CONTROL_COMMANDS")
    if value is None:
        value = config.get_config("DISCORD_ALLOW_OPA_CONTROL_COMMANDS")
    return _coerce_to_bool(value)


def _get_command_rate_limit_window() -> float:
    """Return the configured rate limit window in seconds."""
    overrides = getattr(config, "CONFIG_OVERRIDES", {})
    value = overrides.get("DISCORD_COMMAND_RATE_LIMIT_SECONDS")
    if value is None:
        value = config.get_config("DISCORD_COMMAND_RATE_LIMIT_SECONDS")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return _DEFAULT_RATE_LIMIT_WINDOW_SECONDS


async def _has_control_command_permission(
    user: Any, command: str, *, agent_id: str | None = None
) -> bool:
    """Return True when the user may execute privileged control commands."""

    if has_admin_permission(user):
        return True
    if not _allow_control_via_opa():
        return False
    payload = {
        "command": command,
        "user_id": str(getattr(user, "id", "")),
    }
    if agent_id is not None:
        payload["agent_id"] = agent_id
    allowed, _ = await evaluate_with_opa(json.dumps(payload))
    return bool(allowed)


async def check_command_rate_limit(user: Any) -> bool:
    """Increment and check the rate limit for the given user."""
    user_id = str(getattr(user, "id", ""))
    if not user_id:
        return True
    lock = _COMMAND_LOCKS.setdefault(user_id, asyncio.Lock())
    async with lock:
        history = _COMMAND_HISTORY.setdefault(user_id, deque())
        now = time.monotonic()
        window_seconds = _get_command_rate_limit_window()
        if window_seconds <= 0:
            history.clear()
        else:
            cutoff = now - window_seconds
            while history and history[0] <= cutoff:
                history.popleft()
        if len(history) >= _MAX_RATE:
            logger.warning("Rate limit exceeded for user %s", user_id)
            return False
        history.append(now)
    return True


def reset_command_counts(user_id: str | None = None) -> None:
    """Reset stored command counts for a user or all users."""
    if user_id is not None:
        history = _COMMAND_HISTORY.pop(user_id, None)
        if history is not None:
            history.clear()
    else:
        _COMMAND_HISTORY.clear()


def set_max_rate(value: int) -> None:
    """Set the maximum allowed commands per user."""
    global _MAX_RATE
    _MAX_RATE = max(1, int(value))


async def _rate_limit_check(interaction: Any) -> bool:
    """Global slash-command check enforcing per-user rate limits."""
    if await check_command_rate_limit(getattr(interaction, "user", None)):
        return True
    try:
        await send_interaction_response(interaction, "rate limit exceeded", ephemeral=True)
    except Exception:  # pragma: no cover - best effort
        pass
    return False


if hasattr(bot.tree, "add_check"):
    bot.tree.add_check(_rate_limit_check)


async def record_misbehavior(interaction: Any, agent_id: str, reason: str) -> None:
    """Record a misbehavior event and capture a replay slice."""
    bot_instance = get_active_bot()
    ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
    sim = ctx.sim_state.get("simulation")
    step = int(getattr(sim, "current_step", 0))
    path: str | None = None
    try:
        slice_path = event_log.store_replay_slice(step, step, directory=SNAPSHOT_DIR)
    except Exception:  # pragma: no cover - best effort
        slice_path = None
    if slice_path is not None:
        path = str(slice_path)
    event: dict[str, Any] = {"step": step, "agent_id": agent_id, "reason": reason}
    if path is not None:
        event["replay_path"] = path
    try:  # pragma: no cover - best effort
        event_log.log_misbehavior(event)
    except Exception:
        pass
    await ctx.get_event_queue().put(SimulationEvent(type="misbehavior", data=event))
    await send_interaction_response(interaction, "misbehavior recorded", ephemeral=True)


from src.interfaces.discord_moderation import (  # noqa: E402
    moderation_rate_limit,
    register_moderation_commands,
)


@bot.command(name="say")
async def say(ctx: Any, *, message: str) -> None:
    """Echo a user-provided message for smoke testing."""
    with command_span("say", ctx) as span:
        await send_channel_message(ctx, content=f"Simulated message received: {message}")


@bot.command(name="stats")
async def stats(ctx: Any) -> None:
    """Return basic runtime statistics."""
    stats_text = f"LLM latency: {get_llm_latency()} ms; KB size: {get_kb_size()}"
    with command_span("stats", ctx) as span:
        span.set_attribute("discord.message.length", len(stats_text))
        await send_channel_message(ctx, content=stats_text)


@bot.tree.command(name="status")
async def slash_status(interaction: Any) -> None:
    """Return IP/DU balance for the mapped agent."""
    with command_span("status", interaction) as span:
        agent_id = None
        bot_instance = get_active_bot()
        if bot_instance is not None:
            channel = getattr(interaction, "channel", None)
            chan_id = getattr(channel, "id", None)
            agent_id = bot_instance.channel_to_agent.get(chan_id)
        if agent_id:
            span.set_attribute("discord.agent.id", agent_id)
            ip, du = await ledger.get_balance_async(agent_id)
            if ip <= 0 or du <= 0:
                await send_interaction_response(interaction, "Insufficient IP/DU", ephemeral=True)
                return
            await send_interaction_response(
                interaction, f"IP: {ip:.1f}; DU: {du:.1f}", ephemeral=True
            )
        else:
            await send_interaction_response(interaction, "Unknown channel", ephemeral=True)


@bot.tree.command(name="stats")
async def slash_stats(interaction: Any) -> None:
    """Return runtime metrics if the agent has resources."""
    with command_span("stats", interaction) as span:
        agent_id = None
        bot_instance = get_active_bot()
        if bot_instance is not None:
            channel = getattr(interaction, "channel", None)
            chan_id = getattr(channel, "id", None)
            agent_id = bot_instance.channel_to_agent.get(chan_id)
        if agent_id:
            span.set_attribute("discord.agent.id", agent_id)
            ip, du = await ledger.get_balance_async(agent_id)
            if ip <= 0 or du <= 0:
                await send_interaction_response(interaction, "Insufficient IP/DU", ephemeral=True)
                return
        stats_text = f"LLM latency: {get_llm_latency()} ms; KB size: {get_kb_size()}"
        await send_interaction_response(interaction, stats_text, ephemeral=True)


@bot.tree.command(name="pause")
async def slash_pause(interaction: Any) -> None:
    """Pause the simulation via a control command."""
    with command_span("pause", interaction) as span:
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        await ctx.get_event_queue().put(SimulationEvent(type="control", data={"command": "pause"}))
        await send_interaction_response(interaction, "pause", ephemeral=True)


@bot.tree.command(name="resume")
async def slash_resume(interaction: Any) -> None:
    """Resume the simulation via a control command."""
    with command_span("resume", interaction) as span:
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        await ctx.get_event_queue().put(
            SimulationEvent(type="control", data={"command": "resume"})
        )
        await send_interaction_response(interaction, "resume", ephemeral=True)


@bot.tree.command(name="pause_all")
async def slash_pause_all(interaction: Any) -> None:
    """Pause all activity in the simulation. Administrator only."""
    with command_span("pause_all", interaction) as span:
        if not has_admin_permission(getattr(interaction, "user", None)):
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        await ctx.get_event_queue().put(
            SimulationEvent(type="control", data={"command": "pause_all"})
        )
        await send_interaction_response(interaction, "pause all", ephemeral=True)


@bot.tree.command(name="kill_agent")
@app_commands.describe(agent_id="ID of the agent to kill")
async def slash_kill_agent(interaction: Any, agent_id: str) -> None:
    """Remove an agent from the simulation. Administrator only."""
    with command_span("kill_agent", interaction, agent_id=agent_id) as span:
        if not has_admin_permission(getattr(interaction, "user", None)):
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        await ctx.get_event_queue().put(
            SimulationEvent(type="control", data={"command": "kill_agent", "agent_id": agent_id})
        )
        await send_interaction_response(interaction, "killed", ephemeral=True)


@bot.tree.command(name="nudge")
@app_commands.describe(prompt="Prompt to nudge the simulation")
async def slash_nudge(interaction: Any, prompt: str) -> None:
    """Send a custom prompt to the simulation."""
    with command_span("nudge", interaction) as span:
        span.set_attribute("discord.message.length", len(prompt))
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        await ctx.get_event_queue().put(SimulationEvent(type="nudge", data={"prompt": prompt}))
        await send_interaction_response(interaction, "nudge sent", ephemeral=True)


@bot.tree.command(name="start")
@moderation_rate_limit("start")
async def slash_start(interaction: Any) -> None:
    """Start the simulation via a control command."""
    with command_span("start", interaction) as span:
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        channel = getattr(interaction, "channel", None)
        chan_id = getattr(channel, "id", None)
        agent_id = None
        if bot_instance is not None:
            agent_id = bot_instance.channel_to_agent.get(chan_id)
        if not await _has_control_command_permission(
            getattr(interaction, "user", None),
            "start",
            agent_id=agent_id,
        ):
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        try:
            await start_simulation(ctx)
        except Exception as exc:
            if bot_instance is not None:
                embed = (
                    bot_instance.create_start_embed(False, str(exc))
                    if agent_id is None
                    else bot_instance.create_start_embed(False, str(exc), agent_id)
                )
                await bot_instance.send_simulation_update(embed=embed)
                await send_interaction_response(
                    interaction, "", embed=embed_from_payload(embed), ephemeral=True
                )
            else:
                await send_interaction_response(interaction, "start failed", ephemeral=True)
        else:
            if bot_instance is not None:
                embed = (
                    bot_instance.create_start_embed(True)
                    if agent_id is None
                    else bot_instance.create_start_embed(True, agent_id=agent_id)
                )
                await bot_instance.send_simulation_update(embed=embed)
                await send_interaction_response(
                    interaction, "", embed=embed_from_payload(embed), ephemeral=True
                )
            else:
                await send_interaction_response(interaction, "start", ephemeral=True)


@bot.tree.command(name="stop")
@moderation_rate_limit("stop")
async def slash_stop(interaction: Any) -> None:
    """Stop the simulation via a control command."""
    with command_span("stop", interaction) as span:
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        channel = getattr(interaction, "channel", None)
        chan_id = getattr(channel, "id", None)
        agent_id = None
        if bot_instance is not None:
            agent_id = bot_instance.channel_to_agent.get(chan_id)
        if not await _has_control_command_permission(
            getattr(interaction, "user", None),
            "stop",
            agent_id=agent_id,
        ):
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        try:
            await stop_simulation(ctx)
        except Exception as exc:
            if bot_instance is not None:
                embed = (
                    bot_instance.create_stop_embed(False, str(exc))
                    if agent_id is None
                    else bot_instance.create_stop_embed(False, str(exc), agent_id)
                )
                await bot_instance.send_simulation_update(embed=embed)
                await send_interaction_response(
                    interaction, "", embed=embed_from_payload(embed), ephemeral=True
                )
            else:
                await send_interaction_response(interaction, "stop failed", ephemeral=True)
        else:
            if bot_instance is not None:
                embed = (
                    bot_instance.create_stop_embed(True)
                    if agent_id is None
                    else bot_instance.create_stop_embed(True, agent_id=agent_id)
                )
                await bot_instance.send_simulation_update(embed=embed)
                await send_interaction_response(
                    interaction, "", embed=embed_from_payload(embed), ephemeral=True
                )
            else:
                await send_interaction_response(interaction, "stop", ephemeral=True)


@bot.tree.command(name="spawn")
@moderation_rate_limit("spawn")
async def slash_spawn(interaction: Any, agent_id: str) -> None:
    """Spawn a new agent in the simulation."""
    with command_span("spawn", interaction, agent_id=agent_id) as span:
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        if not await _has_control_command_permission(
            getattr(interaction, "user", None),
            "spawn",
            agent_id=agent_id,
        ):
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        try:
            await spawn_agent_command(agent_id, ctx)
        except Exception as exc:
            if bot_instance is not None:
                embed = bot_instance.create_spawn_embed(agent_id, False, str(exc))
                await bot_instance.send_simulation_update(embed=embed)
                await send_interaction_response(
                    interaction, "", embed=embed_from_payload(embed), ephemeral=True
                )
            else:
                await send_interaction_response(
                    interaction, f"spawn {agent_id} failed", ephemeral=True
                )
        else:
            if bot_instance is not None:
                embed = bot_instance.create_spawn_embed(agent_id, True)
                await bot_instance.send_simulation_update(embed=embed)
                await send_interaction_response(
                    interaction, "", embed=embed_from_payload(embed), ephemeral=True
                )
            else:
                await send_interaction_response(interaction, f"spawn {agent_id}", ephemeral=True)


@bot.tree.command(name="kill")
async def slash_kill(interaction: Any) -> None:
    """Shutdown the bot. Administrator only."""
    with command_span("kill", interaction) as span:
        if not has_admin_permission(getattr(interaction, "user", None)):
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        await send_interaction_response(interaction, "shutting down", ephemeral=True)
        await bot.close()


@bot.tree.command(name="set_max_rate")
async def slash_set_max_rate(interaction: Any, value: int) -> None:
    """Adjust the per-user command rate limit."""
    with command_span("set_max_rate", interaction) as span:
        if not has_admin_permission(getattr(interaction, "user", None)):
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        set_max_rate(value)
        await send_interaction_response(interaction, f"max rate set to {value}", ephemeral=True)


@bot.tree.command(name="set_speed")
async def slash_set_speed(interaction: Any, value: float) -> None:
    """Adjust the simulation speed via a control command."""
    with command_span("set_speed", interaction) as span:
        span.set_attribute("discord.speed", value)
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        await ctx.get_event_queue().put(
            SimulationEvent(type="control", data={"command": "set_speed", "value": value})
        )
        await send_interaction_response(interaction, f"speed {value}", ephemeral=True)


@bot.tree.command(name="speed")
async def slash_speed(interaction: Any, value: float) -> None:
    """Alias for ``set_speed``."""
    with command_span("speed", interaction) as span:
        callback = cast(Callable[[Any, float], Awaitable[None]], slash_set_speed.callback)
        await callback(interaction, value)


@bot.tree.command(name="kb")
async def slash_kb(interaction: Any, text: str) -> None:
    """Post an entry to the Knowledge Board."""
    with command_span("kb", interaction) as span:
        span.set_attribute("discord.message.length", len(text))
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        await ctx.get_event_queue().put(
            SimulationEvent(
                type="control",
                data={
                    "command": "post_kb",
                    "text": text,
                    "author": str(getattr(interaction, "user", "human")),
                },
            )
        )
        await send_interaction_response(interaction, "KB entry created", ephemeral=True)


@bot.tree.command(name="propose")
async def slash_propose(interaction: Any, text: str) -> None:
    """Propose a law via the dashboard API."""
    with command_span("propose", interaction) as span:
        span.set_attribute("discord.message.length", len(text))
        agent_id = None
        bot_instance = get_active_bot()
        if bot_instance is not None:
            channel = getattr(interaction, "channel", None)
            chan_id = getattr(channel, "id", None)
            agent_id = bot_instance.channel_to_agent.get(chan_id)
        if not agent_id:
            await send_interaction_response(interaction, "Unknown channel", ephemeral=True)
            return
        span.set_attribute("discord.agent.id", agent_id)
        ip, du = await ledger.get_balance_async(agent_id)
        if ip <= 0 or du <= 0:
            await send_interaction_response(interaction, "Insufficient IP/DU", ephemeral=True)
            return
        payload: dict[str, object] = {"proposer_id": agent_id, "text": text}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "http://localhost:8000/api/governance/propose", json=payload
                )
                data = json.loads(resp.text)
                approved = data.get("approved", False)
        except Exception:
            approved = False
        await send_interaction_response(
            interaction, "Approved" if approved else "Rejected", ephemeral=True
        )


@bot.tree.command(name="propose_law")
async def slash_propose_law(interaction: Any, text: str, weights: str | None = None) -> None:
    """Propose a law via the dashboard API."""
    with command_span("propose_law", interaction) as span:
        span.set_attribute("discord.message.length", len(text))
        agent_id = None
        bot_instance = get_active_bot()
        if bot_instance is not None:
            channel = getattr(interaction, "channel", None)
            chan_id = getattr(channel, "id", None)
            agent_id = bot_instance.channel_to_agent.get(chan_id)
        if not agent_id:
            await send_interaction_response(interaction, "Unknown channel", ephemeral=True)
            return
        span.set_attribute("discord.agent.id", agent_id)
        ip, du = await ledger.get_balance_async(agent_id)
        if ip <= 0 or du <= 0:
            await send_interaction_response(interaction, "Insufficient IP/DU", ephemeral=True)
            return
        payload: dict[str, object] = {"proposer_id": agent_id, "text": text}
        if weights:
            try:
                payload["vote_weights"] = json.loads(weights)
            except Exception:
                payload["vote_weights"] = None
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post("http://localhost:8000/api/propose_law", json=payload)
                data = json.loads(resp.text)
                approved = data.get("approved", False)
        except Exception:
            approved = False
        await send_interaction_response(
            interaction, "Approved" if approved else "Rejected", ephemeral=True
        )


@bot.tree.command(name="vote")
async def slash_vote(interaction: Any, text: str, approve: bool = True) -> None:
    """Cast a manual vote on a proposal via the governance service."""
    with command_span("vote", interaction) as span:
        span.set_attribute("discord.message.length", len(text))
        agent_id = None
        bot_instance = get_active_bot()
        if bot_instance is not None:
            channel = getattr(interaction, "channel", None)
            chan_id = getattr(channel, "id", None)
            agent_id = bot_instance.channel_to_agent.get(chan_id)
        if not agent_id:
            await send_interaction_response(interaction, "Unknown channel", ephemeral=True)
            return
        span.set_attribute("discord.agent.id", agent_id)
        ip, du = await ledger.get_balance_async(agent_id)
        if ip <= 0 or du <= 0:
            await send_interaction_response(interaction, "Insufficient IP/DU", ephemeral=True)
            return
        payload = {"agent_id": agent_id, "text": text, "approve": approve}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post("http://localhost:8000/api/vote", json=payload)
                data = json.loads(resp.text)
                cast = data.get("vote", False)
        except Exception:
            cast = False
        await send_interaction_response(
            interaction, "Vote cast" if cast else "Vote rejected", ephemeral=True
        )


@bot.tree.command(name="misbehavior_log")
async def slash_misbehavior_log(interaction: Any, limit: int = 20) -> None:
    """Return last ``limit`` misbehavior events."""
    events = await asyncio.to_thread(event_log.fetch_events, event_type="misbehavior")
    events = events[-limit:]
    lines = []
    for evt in events:
        step = evt.get("step")
        agent = evt.get("agent_id")
        reason = evt.get("reason")
        path = evt.get("replay_path")
        lines.append(f"{step}: {agent} - {reason} ({path})")
    msg = "\n".join(lines) if lines else "no misbehavior"
    await send_interaction_response(interaction, msg, ephemeral=True)
