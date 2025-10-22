"""Discord moderation commands with OPA-based per-user rate limiting."""

import time
from functools import wraps
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable

from src.infra.ledger import log_penalty
from src.interfaces import discord_bot
from src.interfaces.dashboard_backend import DEFAULT_CONTEXT, SimulationEvent
from src.utils.policy import evaluate_with_opa

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from discord.ext.commands import Bot as DiscordBot  # noqa: F401

bot = getattr(discord_bot, "bot")
get_active_bot = discord_bot.get_active_bot
has_admin_permission = discord_bot.has_admin_permission

_APP_COMMANDS = getattr(discord_bot, "app_commands", SimpleNamespace(describe=lambda *a, **k: (lambda f: f)))

_ACTION_COUNTS: dict[str, int] = {}
_COOLDOWNS: dict[str, float] = {}
_COOLDOWN_SECONDS = 1.0
_IP_PENALTY = 1.0
_DU_PENALTY = 1.0


async def _rate_limit(user: Any, action: str, agent_id: str | None = None) -> bool:
    """Use OPA to determine if the action is allowed for the user and agent."""
    user_id = str(getattr(user, "id", ""))
    opa_key = f"{user_id}:{action}"
    if agent_id is not None:
        opa_key += f":{agent_id}"

    count_key = f"{user_id}:{action}"
    _ACTION_COUNTS[count_key] = _ACTION_COUNTS.get(count_key, 0) + 1

    now = time.monotonic()
    if _COOLDOWNS.get(count_key, 0.0) > now:
        return False

    allow, _ = await evaluate_with_opa(opa_key)
    if allow:
        _COOLDOWNS[count_key] = now + _COOLDOWN_SECONDS
    return allow


def moderation_rate_limit(action: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator applying per-user rate limiting for moderation commands."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(interaction: Any, *args: Any, **kwargs: Any) -> Any:
            agent_id_value: str | None = None
            if args:
                agent_id_value = str(args[0])
            elif "agent_id" in kwargs and kwargs.get("agent_id") is not None:
                agent_id_value = str(kwargs.get("agent_id"))

            with discord_bot.command_span(action, interaction, agent_id=agent_id_value):
                bot_instance = get_active_bot()
                ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
                if not await _rate_limit(
                    getattr(interaction, "user", None), action, agent_id_value
                ):
                    await ctx.get_event_queue().put(
                        SimulationEvent(
                            type="moderation",
                            data={
                                "command": action,
                                "agent_id": agent_id_value,
                                "violation": True,
                            },
                        )
                    )
                    await ctx.get_event_queue().put(
                        SimulationEvent(
                            type="moderation",
                            data={
                                "command": "penalty",
                                "agent_id": agent_id_value,
                                "ip": _IP_PENALTY,
                                "du": _DU_PENALTY,
                            },
                        )
                    )
                    try:  # pragma: no cover - best effort
                        log_penalty(
                            agent_id_value, _IP_PENALTY, _DU_PENALTY, "rate_limit_violation"
                        )
                    except Exception:
                        pass
                    await interaction.response.send_message("rate limited", ephemeral=True)
                    return None
                return await func(interaction, *args, **kwargs)

        return wrapper

    return decorator


def _context_description(**kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return an ``app_commands.describe`` decorator when available."""

    describe = getattr(_APP_COMMANDS, "describe", None)
    if describe is None:
        def _noop(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return _noop

    return describe(**kwargs)


def register_moderation_commands(
    tree: Any,
    *,
    resolve_context: Callable[[], tuple[Any, Any]],
) -> dict[str, Callable[..., Any]]:
    """Register moderation commands on the provided command tree."""

    commands: dict[str, Callable[..., Any]] = {}

    @tree.command(name="reset_memory")
    @_context_description(agent_id="ID of the agent to reset")
    @moderation_rate_limit("reset_memory")
    async def slash_reset_memory(interaction: Any, agent_id: str) -> None:
        ctx, event_queue = resolve_context()
        if not has_admin_permission(getattr(interaction, "user", None)):
            await interaction.response.send_message("unauthorized", ephemeral=True)
            return
        await event_queue.put(
            SimulationEvent(type="moderation", data={"command": "reset_memory", "agent_id": agent_id})
        )
        await discord_bot.send_interaction_response(interaction, "memory reset", ephemeral=True)

    commands["reset_memory"] = slash_reset_memory

    @tree.command(name="penalty")
    @_context_description(
        agent_id="ID of the agent to penalize",
        ip="Influence points to deduct",
        du="Decision units to deduct",
    )
    @moderation_rate_limit("penalty")
    async def slash_penalty(
        interaction: Any,
        agent_id: str,
        ip: float = 0.0,
        du: float = 0.0,
    ) -> None:
        ctx, event_queue = resolve_context()
        if not has_admin_permission(getattr(interaction, "user", None)):
            await interaction.response.send_message("unauthorized", ephemeral=True)
            return
        await event_queue.put(
            SimulationEvent(
                type="moderation",
                data={"command": "penalty", "agent_id": agent_id, "ip": ip, "du": du},
            )
        )
        try:  # pragma: no cover - best effort
            log_penalty(agent_id, abs(ip), abs(du), "moderation_penalty")
        except Exception:
            pass
        await discord_bot.send_interaction_response(interaction, "penalty applied", ephemeral=True)

    commands["penalty"] = slash_penalty

    @tree.command(name="mute")
    @_context_description(agent_id="ID of the agent to mute")
    @moderation_rate_limit("mute")
    async def slash_mute(interaction: Any, agent_id: str) -> None:
        ctx, event_queue = resolve_context()
        await event_queue.put(
            SimulationEvent(type="moderation", data={"command": "mute", "agent_id": agent_id})
        )
        await discord_bot.send_interaction_response(interaction, "muted", ephemeral=True)

    commands["mute"] = slash_mute

    @tree.command(name="unmute")
    @_context_description(agent_id="ID of the agent to unmute")
    @moderation_rate_limit("unmute")
    async def slash_unmute(interaction: Any, agent_id: str) -> None:
        ctx, event_queue = resolve_context()
        await event_queue.put(
            SimulationEvent(type="moderation", data={"command": "unmute", "agent_id": agent_id})
        )
        await discord_bot.send_interaction_response(interaction, "unmuted", ephemeral=True)

    commands["unmute"] = slash_unmute

    return commands


def _global_context_resolver() -> tuple[Any, Any]:
    bot_instance = get_active_bot()
    if bot_instance is not None:
        return bot_instance.context, bot_instance.event_queue
    ctx = DEFAULT_CONTEXT
    return ctx, ctx.get_event_queue()


_REGISTERED_COMMANDS = register_moderation_commands(
    getattr(bot, "tree"), resolve_context=_global_context_resolver
)

slash_reset_memory = _REGISTERED_COMMANDS["reset_memory"]
slash_penalty = _REGISTERED_COMMANDS["penalty"]
slash_mute = _REGISTERED_COMMANDS["mute"]
slash_unmute = _REGISTERED_COMMANDS["unmute"]


__all__ = [
    "moderation_rate_limit",
    "register_moderation_commands",
    "slash_mute",
    "slash_penalty",
    "slash_reset_memory",
    "slash_unmute",
]
