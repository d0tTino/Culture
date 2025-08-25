"""Discord moderation commands with OPA-based per-user rate limiting."""

import time
from functools import wraps
from typing import Any, Callable

from src.infra.ledger import log_penalty
from src.interfaces.dashboard_backend import DEFAULT_CONTEXT, SimulationEvent
from src.interfaces.discord_bot import bot, get_active_bot, has_admin_permission
from src.utils.policy import evaluate_with_opa

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
            agent_id: str | None = None
            if args:
                agent_id = args[0]
            elif "agent_id" in kwargs:
                agent_id = str(kwargs.get("agent_id"))
            bot_instance = get_active_bot()
            ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
            if not await _rate_limit(getattr(interaction, "user", None), action, agent_id):
                await ctx.get_event_queue().put(
                    SimulationEvent(
                        type="moderation",
                        data={"command": action, "agent_id": agent_id, "violation": True},
                    )
                )
                await ctx.get_event_queue().put(
                    SimulationEvent(
                        type="moderation",
                        data={
                            "command": "penalty",
                            "agent_id": agent_id,
                            "ip": _IP_PENALTY,
                            "du": _DU_PENALTY,
                        },
                    )
                )
                try:  # pragma: no cover - best effort
                    log_penalty(agent_id, _IP_PENALTY, _DU_PENALTY, "rate_limit_violation")
                except Exception:
                    pass
                await interaction.response.send_message("rate limited", ephemeral=True)
                return None
            return await func(interaction, *args, **kwargs)

        return wrapper

    return decorator


@bot.tree.command(name="mute")
@moderation_rate_limit("mute")
async def slash_mute(interaction: Any, agent_id: str) -> None:
    """Mute an agent in the simulation."""
    bot_instance = get_active_bot()
    ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
    await ctx.get_event_queue().put(
        SimulationEvent(type="moderation", data={"command": "mute", "agent_id": agent_id})
    )
    await interaction.response.send_message("muted", ephemeral=True)


@bot.tree.command(name="reset_memory")
@moderation_rate_limit("reset_memory")
async def slash_reset_memory(interaction: Any, agent_id: str) -> None:
    """Reset the memory of an agent."""
    bot_instance = get_active_bot()
    ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
    if not has_admin_permission(getattr(interaction, "user", None)):
        await interaction.response.send_message("unauthorized", ephemeral=True)
        return
    await ctx.get_event_queue().put(
        SimulationEvent(type="moderation", data={"command": "reset_memory", "agent_id": agent_id})
    )
    await interaction.response.send_message("memory reset", ephemeral=True)


@bot.tree.command(name="penalty")
@moderation_rate_limit("penalty")
async def slash_penalty(interaction: Any, agent_id: str, ip: float = 0.0, du: float = 0.0) -> None:
    """Apply an IP/DU penalty to an agent."""
    bot_instance = get_active_bot()
    ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
    await ctx.get_event_queue().put(
        SimulationEvent(
            type="moderation",
            data={"command": "penalty", "agent_id": agent_id, "ip": ip, "du": du},
        )
    )
    try:  # pragma: no cover - best effort
        log_penalty(agent_id, abs(ip), abs(du), "moderation_penalty")
    except Exception:
        pass
    await interaction.response.send_message("penalty applied", ephemeral=True)


@bot.tree.command(name="unmute")
@moderation_rate_limit("unmute")
async def slash_unmute(interaction: Any, agent_id: str) -> None:
    """Unmute an agent in the simulation."""
    bot_instance = get_active_bot()
    ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
    await ctx.get_event_queue().put(
        SimulationEvent(type="moderation", data={"command": "unmute", "agent_id": agent_id})
    )
    await interaction.response.send_message("unmuted", ephemeral=True)


__all__ = ["slash_mute", "slash_penalty", "slash_reset_memory", "slash_unmute"]
