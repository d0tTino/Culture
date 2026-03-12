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
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from opentelemetry import trace
from typing_extensions import Self

import src.interfaces.interaction_policy.permissions as interaction_permissions
from src.infra import config, event_log
from src.infra.ledger import ledger
from src.interfaces import dashboard_backend as db
from src.interfaces import metrics
from src.interfaces.interaction_policy import (
    check_command_rate_limit as policy_check_command_rate_limit,
)
from src.interfaces.interaction_policy import (
    discord_identity,
    discord_interaction_context,
    discord_message_to_intent_payload,
    has_control_command_permission,
    set_max_rate,
)
from src.interfaces.interaction_schema import (
    ControlEnvelope,
    InjectEventEnvelope,
    KnowledgeBoardEnvelope,
    ModerationEnvelope,
    SpawnEnvelope,
)
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


_MAX_RATE = 5
has_admin_permission = interaction_permissions.has_admin_permission
reset_command_counts = interaction_permissions.reset_command_counts


async def check_command_rate_limit(user: Any) -> bool:
    """Backward-compatible wrapper delegated to interaction policy."""
    set_max_rate(_MAX_RATE)
    interaction_permissions.time = time
    return await policy_check_command_rate_limit(user)


def _governance_message_from_outcome(
    *, approved: bool | None = None, vote_cast: bool | None = None, error_kind: str | None = None
) -> str:
    if error_kind == "invalid_payload":
        return "Invalid request payload. Please verify command arguments and try again."
    if error_kind == "network_unavailable":
        return "Governance service is unavailable right now. Please try again shortly."
    if vote_cast is not None:
        return "Vote cast" if vote_cast else "Vote rejected by policy or vote."
    if approved:
        return "Approved"
    return "Rejected by vote"


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


def build_help_text() -> str:
    """Render stable help text for Discord slash commands."""
    return "\n".join(
        [
            "## 👋 Culture Bot Help",
            "### Public commands",
            "- `/dm <agent_id> <message>` — send a direct message to one agent.",
            "- `/broadcast <message>` — send a message to all agents.",
            "- `/kb <text>` — add a note to the Knowledge Board.",
            "- `/status`, `/stats` — view current state/metrics.",
            "- `/start_here` — show onboarding, modes, and scenario cards.",
            "- `/propose`, `/propose_law`, `/vote` — governance interactions.",
            "",
            "### Moderator/Admin commands",
            "- `/nudge <prompt>` — steer agent behavior *(moderator/admin)*.",
            "- `/event <text>` — inject world events *(moderator/admin)*.",
            "- `/start`, `/stop`, `/pause`, `/resume` — sim lifecycle *(moderator/admin)*.",
            "- `/spawn`, `/kill_agent`, `/pause_all`, `/kill` — high-impact controls *(admin required for kill/pause_all/kill_agent)*.",
            "- `/set_speed`, `/speed`, `/set_max_rate` — tuning controls *(admin required for set_max_rate)*.",
            "",
            "### Quick examples",
            "- `/dm agent-2 What's your latest plan?`",
            "- `/broadcast Team sync in 2 minutes.`",
            "- `/kb Rule: cite data source before proposing policy.`",
            "- `/nudge Consider long-term coalition outcomes.`",
            "",
            "### Permission + rate-limit notes",
            "- Public commands are usable by all channel users unless noted.",
            "- Admin-only commands require Discord administrator privileges.",
            "- Slash commands are globally rate-limited per user (default: 5 commands / 60s).",
            "- Some moderation actions also have cooldowns to reduce spam.",
            "",
            "### User modes",
            "- `observer` — read-only guidance and context.",
            "- `participant` — regular conversation with agents.",
            "- `world-shaper` — propose world-level interventions.",
            "- `moderator` — policy and safety operations.",
        ]
    )


def create_onboarding_embed(channel_id: int) -> Any:
    """Create a lightweight onboarding embed for startup."""
    embed = discord.Embed(
        title="🧭 How to interact",
        description="Use slash commands to talk to agents and moderate the simulation.",
        color=discord.Color.blue(),
    )
    embed.add_field(
        name="Start with these",
        value="`/start_here`, `/help`, `/dm`, `/broadcast`, `/kb`, `/status`",
        inline=False,
    )
    embed.add_field(
        name="Modes",
        value="observer · participant · world-shaper · moderator",
        inline=False,
    )
    embed.add_field(
        name="Permissions",
        value="Public commands are open. Moderation/admin commands are labeled in `/help`.",
        inline=False,
    )
    embed.add_field(
        name="Rate limits",
        value="Global command limit applies per user (default: 5 commands per 60s).",
        inline=False,
    )
    embed.set_footer(text=f"Channel ID: {channel_id}")
    return embed


def scenario_intro_cards() -> list[dict[str, str]]:
    """Starter scenarios shown in onboarding surfaces."""
    return [
        {
            "title": "Coalition Tension",
            "prompt": "Ask two agents to align on a scarce resource policy.",
            "recommended_mode": "participant",
        },
        {
            "title": "Crisis Injection",
            "prompt": "Inject a disruption event and observe adaptation.",
            "recommended_mode": "world-shaper",
        },
        {
            "title": "Safety Review",
            "prompt": "Evaluate and enforce moderation boundaries for escalating speech.",
            "recommended_mode": "moderator",
        },
    ]


def format_explainable_acknowledgement(result: Any) -> str:
    """Return user-visible "what happened and why" summary."""
    data = getattr(result, "data", None) or {}
    provenance = getattr(result, "decision_provenance", None)
    policy_id = getattr(provenance, "policy_id", "") if provenance is not None else ""
    rule_id = getattr(provenance, "rule_id", "") if provenance is not None else ""
    action = data.get("action") or data.get("intent") or "request"
    why = "policy checks passed" if getattr(result, "status", "") == "ok" else "policy blocked"
    if policy_id or rule_id:
        why = f"{why} ({policy_id}:{rule_id})"
    return f"Action: {action}. Outcome: {getattr(result, 'user_message', '')} Why: {why}."


MAX_EMBED_DESCRIPTION_LENGTH = 4096


def _parse_json_object_argument(raw: str | None, field_name: str) -> dict[str, Any] | None:
    """Parse a JSON object argument from a slash-command string option."""
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {field_name} JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return cast(dict[str, Any], parsed)


def _spawn_kwargs_from_inputs(
    *,
    role: str | None = None,
    role_json: str | None = None,
    persona: str | None = None,
    backstory: str | None = None,
    traits_json: str | None = None,
    openness: float | None = None,
    analytical_focus: float | None = None,
    empathy: float | None = None,
    assertiveness: float | None = None,
    emotional_sensitivity: float | None = None,
    resilience: float | None = None,
    trust_baseline: float | None = None,
    adaptability: float | None = None,
) -> dict[str, Any]:
    """Build spawn payload kwargs from slash-command optional inputs."""
    role_payload = (
        _parse_json_object_argument(role_json, "role") if role_json is not None else None
    )
    if role_payload is None and role is not None and role.strip():
        role_payload = role.strip()

    traits_payload = (
        _parse_json_object_argument(traits_json, "traits") if traits_json is not None else None
    )
    explicit_traits = {
        "openness": openness,
        "analytical_focus": analytical_focus,
        "empathy": empathy,
        "assertiveness": assertiveness,
        "emotional_sensitivity": emotional_sensitivity,
        "resilience": resilience,
        "trust_baseline": trust_baseline,
        "adaptability": adaptability,
    }
    explicit_clean = {k: float(v) for k, v in explicit_traits.items() if v is not None}
    if explicit_clean:
        merged = dict(traits_payload or {})
        merged.update(explicit_clean)
        traits_payload = merged

    kwargs: dict[str, Any] = {}
    if role_payload is not None:
        kwargs["role"] = role_payload
    if persona is not None and persona.strip():
        kwargs["persona"] = persona.strip()
    if backstory is not None and backstory.strip():
        kwargs["backstory"] = backstory.strip()
    if traits_payload is not None:
        kwargs["traits"] = traits_payload
    return kwargs


def _default_agent_for_channel(self: "SimulationDiscordBot", channel_id: int | None) -> str | None:
    """Resolve a deterministic fallback agent for an unmapped incoming message."""
    if channel_id is not None:
        mapped = self.channel_to_agent.get(channel_id)
        if mapped:
            return mapped

    if self.last_agent_id:
        return self.last_agent_id

    if self.channel_map:
        return sorted(self.channel_map)[0]

    return None


def _default_human_message_broadcast() -> bool:
    """Return True when plain human messages should fan out to all agents."""
    overrides = getattr(config, "CONFIG_OVERRIDES", {})
    value = overrides.get("DISCORD_DEFAULT_BROADCAST")
    if value is None:
        value = config.get_config("DISCORD_DEFAULT_BROADCAST")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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
        token_lookup: (Callable[[str], Awaitable[str | None] | str] | None) = None,
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
        token_lookup: (Callable[[str], Awaitable[str | None] | str] | None) = None,
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
        queue = context.get_event_queue()
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
                register_slash_commands(tree)

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
                    await send_channel_message(
                        channel, embed=create_onboarding_embed(self.channel_id)
                    )
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
                    sender = self.user_agents.get(str(user_id)) if user_id else None
                    fallback_agent = _default_agent_for_channel(self, channel_id)
                    payload, validation_error = discord_message_to_intent_payload(
                        content=content,
                        sender_agent_id=sender,
                        fallback_agent_id=fallback_agent,
                        raw_metadata={"channel_id": channel_id, "user_id": user_id},
                        mode="participant",
                    )
                    if validation_error is not None:
                        await send_channel_message(channel, content=validation_error)
                        return
                    if payload is None:
                        return
                    routing = (
                        payload.get("routing") if isinstance(payload.get("routing"), dict) else {}
                    )
                    target_agent_id = routing.get("target_agent_id")
                    span.set_attribute("discord.agent.id", target_agent_id or "")
                    if user_id and sender is None and target_agent_id is not None:
                        self.user_agents[str(user_id)] = str(target_agent_id)
                    self.last_agent_id = target_agent_id
                    self.last_channel_id = channel_id
                    bus = get_command_bus(self.context)
                    if bus is None:
                        return
                    result = await bus.dispatch_payload(
                        payload,
                        context=discord_interaction_context(
                            user=user,
                            channel=channel,
                        ),
                    )
                    acknowledgement = format_explainable_acknowledgement(result)
                    await send_channel_message(channel, content=acknowledgement)

    async def _select_client(self: Self, agent_id: str | None) -> Any:
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
        content: str | None = None,
        embed: Any | None = None,
        agent_id: str | None = None,
        *,
        target_channel_id: int | None = None,
        recipient: str | None = None,
    ) -> bool | None:
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
            if content:
                if not allow_message(content):
                    span.set_attribute("discord.message.blocked", True)
                    span.set_attribute("discord.message.block_reason", "allow_message")
                    if agent_id is not None:
                        metrics.DISCORD_AGENT_OUTPUTS_BLOCKED_TOTAL.inc()
                    logger.debug("Message blocked by policy")
                    return False
                allowed, content = await evaluate_with_opa(content)
                if not allowed:
                    span.set_attribute("discord.message.blocked", True)
                    span.set_attribute("discord.message.block_reason", "opa")
                    if agent_id is not None:
                        metrics.DISCORD_AGENT_OUTPUTS_BLOCKED_TOTAL.inc()
                    logger.debug("Message blocked by OPA policy")
                    return False
            # NOTE: embed-only updates preserve current behavior (no embed text moderation yet).
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
        recipient_id: str | None = None,
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


def get_command_bus(ctx: SimulationContext = DEFAULT_CONTEXT) -> Any | None:
    simulation = ctx.sim_state.get("simulation")
    return getattr(simulation, "command_bus", None)


def _global_context_resolver() -> tuple[Any, Any]:
    """Resolve context and event queue for slash command registration."""
    bot_instance = get_active_bot()
    if bot_instance is not None:
        return bot_instance.context, bot_instance.event_queue
    ctx = DEFAULT_CONTEXT
    return ctx, ctx.get_event_queue()


# --- Command rate limiting -------------------------------------------------


async def _has_control_command_permission(
    user: Any,
    channel: Any,
    command: str,
    *,
    agent_id: str | None = None,
) -> bool:
    """Return True when the user may execute privileged control commands."""

    identity = discord_identity(user=user, channel=channel)
    return await has_control_command_permission(identity, command, agent_id=agent_id)


async def _rate_limit_check(interaction: Any) -> bool:
    """Global slash-command check enforcing per-user rate limits."""
    identity = discord_identity(
        user=getattr(interaction, "user", None), channel=getattr(interaction, "channel", None)
    )
    if await check_command_rate_limit(identity):
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


async def slash_pause(interaction: Any) -> None:
    """Pause the simulation via a control command."""
    with command_span("pause", interaction) as span:
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        bus = get_command_bus(ctx)
        if bus is not None:
            await bus.dispatch(
                ControlEnvelope(
                    action="pause",
                    routing={
                        "sender_id": str(getattr(interaction, "user", "discord")),
                        "source": "discord",
                    },
                    auth={"permissions": {"admin", "moderator"}},
                )
            )
        await send_interaction_response(interaction, "pause", ephemeral=True)


async def slash_resume(interaction: Any) -> None:
    """Resume the simulation via a control command."""
    with command_span("resume", interaction) as span:
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        bus = get_command_bus(ctx)
        if bus is not None:
            await bus.dispatch(
                ControlEnvelope(
                    action="resume",
                    routing={
                        "sender_id": str(getattr(interaction, "user", "discord")),
                        "source": "discord",
                    },
                    auth={"permissions": {"admin", "moderator"}},
                )
            )
        await send_interaction_response(interaction, "resume", ephemeral=True)


async def slash_pause_all(interaction: Any) -> None:
    """Pause all activity in the simulation. Administrator only."""
    with command_span("pause_all", interaction) as span:
        if not discord_identity(
            user=getattr(interaction, "user", None), channel=getattr(interaction, "channel", None)
        ).is_admin:
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        bus = get_command_bus(ctx)
        if bus is not None:
            await bus.dispatch(
                ControlEnvelope(
                    action="pause_all",
                    routing={
                        "sender_id": str(getattr(interaction, "user", "discord")),
                        "source": "discord",
                    },
                    auth={"permissions": {"admin", "moderator"}},
                )
            )
        await send_interaction_response(interaction, "pause all", ephemeral=True)


async def slash_kill_agent(interaction: Any, agent_id: str) -> None:
    """Remove an agent from the simulation. Administrator only."""
    with command_span("kill_agent", interaction, agent_id=agent_id) as span:
        if not discord_identity(
            user=getattr(interaction, "user", None), channel=getattr(interaction, "channel", None)
        ).is_admin:
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        bus = get_command_bus(ctx)
        if bus is not None:
            await bus.dispatch(
                ControlEnvelope(
                    action="kill_agent",
                    agent_id=agent_id,
                    routing={
                        "sender_id": str(getattr(interaction, "user", "discord")),
                        "source": "discord",
                    },
                    auth={"permissions": {"admin", "moderator"}},
                )
            )
        await send_interaction_response(interaction, "killed", ephemeral=True)


async def slash_nudge(interaction: Any, prompt: str) -> None:
    """Send a custom prompt through the command bus."""
    with command_span("nudge", interaction) as span:
        span.set_attribute("discord.message.length", len(prompt))
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        bus = get_command_bus(ctx)
        if bus is None:
            await send_interaction_response(interaction, "command bus unavailable", ephemeral=True)
            return
        result = await bus.dispatch(
            ModerationEnvelope(
                action="nudge",
                agent_id=str(getattr(interaction, "user", "human")),
                routing={
                    "sender_id": str(getattr(interaction, "user", "human")),
                    "source": "discord",
                },
                auth={"permissions": {"admin", "moderator"}},
                metadata={
                    "prompt": prompt,
                    "correlation_id": str(getattr(interaction, "id", "")) or None,
                },
                correlation_id=str(getattr(interaction, "id", "")) or None,
            )
        )
        await send_interaction_response(interaction, result.user_message, ephemeral=True)


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
            getattr(interaction, "channel", None),
            "start",
            agent_id=agent_id,
        ):
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        try:
            bus = get_command_bus(ctx)
            if bus is None:
                raise RuntimeError("command bus unavailable")
            await bus.dispatch(
                ControlEnvelope(
                    action="start",
                    routing={
                        "sender_id": str(getattr(interaction, "user", "discord")),
                        "source": "discord",
                    },
                    auth={"permissions": {"admin", "moderator"}},
                )
            )
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
            getattr(interaction, "channel", None),
            "stop",
            agent_id=agent_id,
        ):
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        try:
            bus = get_command_bus(ctx)
            if bus is None:
                raise RuntimeError("command bus unavailable")
            await bus.dispatch(
                ControlEnvelope(
                    action="stop",
                    routing={
                        "sender_id": str(getattr(interaction, "user", "discord")),
                        "source": "discord",
                    },
                    auth={"permissions": {"admin", "moderator"}},
                )
            )
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


async def slash_spawn(
    interaction: Any,
    agent_id: str,
    role: str | None = None,
    role_json: str | None = None,
    persona: str | None = None,
    backstory: str | None = None,
    traits_json: str | None = None,
    openness: float | None = None,
    analytical_focus: float | None = None,
    empathy: float | None = None,
    assertiveness: float | None = None,
    emotional_sensitivity: float | None = None,
    resilience: float | None = None,
    trust_baseline: float | None = None,
    adaptability: float | None = None,
) -> None:
    """Spawn a new agent in the simulation."""
    with command_span("spawn", interaction, agent_id=agent_id) as span:
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        if not await _has_control_command_permission(
            getattr(interaction, "user", None),
            getattr(interaction, "channel", None),
            "spawn",
            agent_id=agent_id,
        ):
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        try:
            spawn_kwargs = _spawn_kwargs_from_inputs(
                role=role,
                role_json=role_json,
                persona=persona,
                backstory=backstory,
                traits_json=traits_json,
                openness=openness,
                analytical_focus=analytical_focus,
                empathy=empathy,
                assertiveness=assertiveness,
                emotional_sensitivity=emotional_sensitivity,
                resilience=resilience,
                trust_baseline=trust_baseline,
                adaptability=adaptability,
            )
            bus = get_command_bus(ctx)
            if bus is None:
                raise RuntimeError("command bus unavailable")
            await bus.dispatch(
                SpawnEnvelope(
                    agent_id=agent_id,
                    role=spawn_kwargs.get("role"),
                    persona=spawn_kwargs.get("persona"),
                    backstory=spawn_kwargs.get("backstory"),
                    traits=spawn_kwargs.get("traits"),
                    routing={
                        "sender_id": str(getattr(interaction, "user", "discord")),
                        "source": "discord",
                    },
                    auth={"permissions": {"admin", "moderator"}},
                )
            )
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


async def slash_kill(interaction: Any) -> None:
    """Shutdown the bot. Administrator only."""
    with command_span("kill", interaction) as span:
        if not discord_identity(
            user=getattr(interaction, "user", None), channel=getattr(interaction, "channel", None)
        ).is_admin:
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        await send_interaction_response(interaction, "shutting down", ephemeral=True)
        await bot.close()


async def slash_set_max_rate(interaction: Any, value: int) -> None:
    """Adjust the per-user command rate limit."""
    with command_span("set_max_rate", interaction) as span:
        if not discord_identity(
            user=getattr(interaction, "user", None), channel=getattr(interaction, "channel", None)
        ).is_admin:
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        set_max_rate(value)
        await send_interaction_response(interaction, f"max rate set to {value}", ephemeral=True)


async def slash_set_speed(interaction: Any, value: float) -> None:
    """Adjust the simulation speed via a control command."""
    with command_span("set_speed", interaction) as span:
        span.set_attribute("discord.speed", value)
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        bus = get_command_bus(ctx)
        if bus is not None:
            await bus.dispatch(
                ControlEnvelope(
                    action="set_speed",
                    value=value,
                    routing={
                        "sender_id": str(getattr(interaction, "user", "discord")),
                        "source": "discord",
                    },
                    auth={"permissions": {"admin", "moderator"}},
                )
            )
        await send_interaction_response(interaction, f"speed {value}", ephemeral=True)


async def slash_speed(interaction: Any, value: float) -> None:
    """Alias for ``set_speed``."""
    with command_span("speed", interaction) as span:
        await slash_set_speed(interaction, value)


async def slash_help(interaction: Any) -> None:
    """Show command catalog, permission hints, and examples."""
    with command_span("help", interaction) as span:
        help_text = build_help_text()
        span.set_attribute("discord.message.length", len(help_text))
        await send_interaction_response(interaction, help_text, ephemeral=True)




async def slash_start_here(interaction: Any) -> None:
    """Show onboarding flow with modes and scenario cards."""
    with command_span("start_here", interaction):
        cards = scenario_intro_cards()
        lines = ["## 🚀 Start Here", "Modes: observer · participant · world-shaper · moderator", "", "Scenario cards:"]
        for card in cards:
            lines.append(
                f"- **{card['title']}** ({card['recommended_mode']}): {card['prompt']}"
            )
        await send_interaction_response(interaction, "\n".join(lines), ephemeral=True)

async def slash_kb(interaction: Any, text: str) -> None:
    """Post an entry to the Knowledge Board."""
    with command_span("kb", interaction) as span:
        span.set_attribute("discord.message.length", len(text))
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        bus = get_command_bus(ctx)
        if bus is not None:
            await bus.dispatch(
                KnowledgeBoardEnvelope(
                    text=text,
                    routing={
                        "sender_id": str(getattr(interaction, "user", "human")),
                        "source": "discord",
                    },
                )
            )
        await send_interaction_response(interaction, "KB entry created", ephemeral=True)


async def slash_event(interaction: Any, text: str) -> None:
    """Inject a world event that all agents will perceive next cycle."""
    with command_span("event", interaction) as span:
        span.set_attribute("discord.message.length", len(text))
        if not await _has_control_command_permission(
            getattr(interaction, "user", None),
            getattr(interaction, "channel", None),
            "inject_event",
        ):
            await send_interaction_response(interaction, "unauthorized", ephemeral=True)
            return
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        bus = get_command_bus(ctx)
        if bus is not None:
            await bus.dispatch(
                InjectEventEnvelope(
                    text=text,
                    scope="global",
                    agent_id=str(getattr(interaction, "user", "human")),
                    routing={
                        "sender_id": str(getattr(interaction, "user", "human")),
                        "source": "discord",
                    },
                    auth={"permissions": {"admin", "moderator"}},
                )
            )
        await send_interaction_response(interaction, "event injected", ephemeral=True)


async def slash_propose(interaction: Any, text: str) -> None:
    """Propose a law through the command bus."""
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
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        bus = get_command_bus(ctx)
        if bus is None:
            await send_interaction_response(interaction, "command bus unavailable", ephemeral=True)
            return
        result = await bus.dispatch(
            ModerationEnvelope(
                action="propose",
                agent_id=agent_id,
                routing={"sender_id": agent_id, "source": "discord", "target_agent_id": agent_id},
                metadata={"text": text},
                correlation_id=str(getattr(interaction, "id", "")) or None,
            )
        )
        approved = bool((result.data or {}).get("approved", False))
        await send_interaction_response(
            interaction,
            _governance_message_from_outcome(approved=approved),
            ephemeral=True,
        )


async def slash_propose_law(interaction: Any, text: str, weights: str | None = None) -> None:
    """Propose a law through the command bus."""
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
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        bus = get_command_bus(ctx)
        if bus is None:
            await send_interaction_response(interaction, "command bus unavailable", ephemeral=True)
            return
        result = await bus.dispatch(
            ModerationEnvelope(
                action="propose_law",
                agent_id=agent_id,
                routing={"sender_id": agent_id, "source": "discord", "target_agent_id": agent_id},
                metadata={"text": text, "vote_weights": weights},
                correlation_id=str(getattr(interaction, "id", "")) or None,
            )
        )
        approved = bool((result.data or {}).get("approved", False))
        await send_interaction_response(
            interaction,
            _governance_message_from_outcome(approved=approved),
            ephemeral=True,
        )


async def slash_vote(interaction: Any, text: str, approve: bool = True) -> None:
    """Cast a governance vote through the command bus."""
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
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        bus = get_command_bus(ctx)
        if bus is None:
            await send_interaction_response(interaction, "command bus unavailable", ephemeral=True)
            return
        result = await bus.dispatch(
            ModerationEnvelope(
                action="vote",
                agent_id=agent_id,
                routing={"sender_id": agent_id, "source": "discord", "target_agent_id": agent_id},
                metadata={"text": text, "approve": approve},
                correlation_id=str(getattr(interaction, "id", "")) or None,
            )
        )
        vote_cast = bool((result.data or {}).get("vote", False))
        await send_interaction_response(
            interaction,
            _governance_message_from_outcome(vote_cast=vote_cast),
            ephemeral=True,
        )


async def slash_gov(interaction: Any) -> None:
    """Show active governance rules and enforcement stats."""
    with command_span("gov", interaction):
        bot_instance = get_active_bot()
        ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
        bus = get_command_bus(ctx)
        if bus is None:
            await send_interaction_response(
                interaction, "No active governance rules.", ephemeral=True
            )
            return
        result = await bus.dispatch(
            ModerationEnvelope(
                action="gov",
                routing={
                    "sender_id": str(getattr(interaction, "user", "human")),
                    "source": "discord",
                },
                correlation_id=str(getattr(interaction, "id", "")) or None,
            )
        )
        rules = cast(list[dict[str, Any]], (result.data or {}).get("rules", []))
        if not rules:
            await send_interaction_response(
                interaction, "No active governance rules.", ephemeral=True
            )
            return
        lines = []
        for rule in rules[:8]:
            stats = cast(dict[str, Any], rule.get("enforcement_stats", {}))
            lines.append(
                f"{rule.get('rule_id')}: {rule.get('action_intent')} ({rule.get('decision_mode')}) "
                f"effective={rule.get('effective_date')} rejected={stats.get('rejected', 0)} "
                f"overridden={stats.get('overridden', 0)}"
            )
        await send_interaction_response(interaction, "\n".join(lines), ephemeral=True)


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


def register_slash_commands(tree: Any) -> dict[str, Callable[..., Any]]:
    """Register Discord slash commands on the given command tree."""
    commands: dict[str, Callable[..., Any]] = {}

    if tree is not bot.tree:
        register_moderation_commands(tree, resolve_context=_global_context_resolver)

    def _register(
        name: str,
        callback: Callable[..., Awaitable[None]],
        *,
        descriptions: dict[str, str] | None = None,
        moderation_action: str | None = None,
    ) -> Callable[..., Awaitable[None]]:
        handler = callback
        if moderation_action is not None:
            handler = cast(
                Callable[..., Awaitable[None]],
                moderation_rate_limit(moderation_action)(handler),
            )
        if descriptions:
            handler = cast(
                Callable[..., Awaitable[None]], app_commands.describe(**descriptions)(handler)
            )
        if not hasattr(callback, "callback"):
            setattr(callback, "callback", callback)
        registered = tree.command(name=name)(handler)
        commands[name] = registered
        return registered

    _register("status", slash_status)
    _register("stats", slash_stats)
    _register("pause", slash_pause)
    _register("resume", slash_resume)
    _register("pause_all", slash_pause_all)
    _register("kill_agent", slash_kill_agent, descriptions={"agent_id": "ID of the agent to kill"})
    _register(
        "nudge",
        slash_nudge,
        descriptions={"prompt": "Prompt to nudge the simulation"},
        moderation_action="nudge",
    )
    _register("start", slash_start, moderation_action="start")
    _register("stop", slash_stop, moderation_action="stop")
    _register(
        "spawn",
        slash_spawn,
        descriptions={
            "agent_id": "ID of the agent to spawn",
            "role": "Role name",
            "role_json": "Role profile JSON object",
            "persona": "Persona text",
            "backstory": "Backstory text",
            "traits_json": "Trait overrides JSON object",
            "openness": "Trait override",
            "analytical_focus": "Trait override",
            "empathy": "Trait override",
            "assertiveness": "Trait override",
            "emotional_sensitivity": "Trait override",
            "resilience": "Trait override",
            "trust_baseline": "Trait override",
            "adaptability": "Trait override",
        },
        moderation_action="spawn",
    )
    _register("kill", slash_kill)
    _register("set_max_rate", slash_set_max_rate)
    _register("set_speed", slash_set_speed)
    _register("speed", slash_speed)
    _register("help", slash_help)
    _register("start_here", slash_start_here)
    _register("kb", slash_kb)
    _register("event", slash_event)
    _register("propose", slash_propose)
    _register("propose_law", slash_propose_law)
    _register("vote", slash_vote)
    _register("gov", slash_gov)
    _register("misbehavior_log", slash_misbehavior_log)

    return commands


register_slash_commands(bot.tree)
