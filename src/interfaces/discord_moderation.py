"""Discord moderation commands with OPA-based rate limiting."""

import time
from typing import Any

from src.interfaces.dashboard_backend import (
    DEFAULT_CONTEXT,
    SimulationEvent,
)
from src.interfaces.discord_bot import bot, get_active_bot
from src.utils.policy import evaluate_with_opa

_ACTION_COUNTS: dict[str, int] = {}
_COOLDOWNS: dict[str, float] = {}
_COOLDOWN_SECONDS = 1.0


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


@bot.tree.command(name="mute")
async def slash_mute(interaction: Any, agent_id: str) -> None:
    """Mute an agent in the simulation."""
    bot_instance = get_active_bot()
    ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
    if not await _rate_limit(getattr(interaction, "user", None), "mute", agent_id):
        await ctx.get_event_queue().put(
            SimulationEvent(
                type="moderation",
                data={"command": "mute", "agent_id": agent_id, "violation": True},
            )
        )
        await interaction.response.send_message("rate limited", ephemeral=True)
        return
    await ctx.get_event_queue().put(
        SimulationEvent(type="moderation", data={"command": "mute", "agent_id": agent_id})
    )
    await interaction.response.send_message("muted", ephemeral=True)


@bot.tree.command(name="reset_memory")
async def slash_reset_memory(interaction: Any, agent_id: str) -> None:
    """Reset the memory of an agent."""
    bot_instance = get_active_bot()
    ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
    if not await _rate_limit(getattr(interaction, "user", None), "reset_memory", agent_id):
        await ctx.get_event_queue().put(
            SimulationEvent(
                type="moderation",
                data={"command": "reset_memory", "agent_id": agent_id, "violation": True},
            )
        )
        await interaction.response.send_message("rate limited", ephemeral=True)
        return
    await ctx.get_event_queue().put(
        SimulationEvent(type="moderation", data={"command": "reset_memory", "agent_id": agent_id})
    )
    await interaction.response.send_message("memory reset", ephemeral=True)


@bot.tree.command(name="penalty")
async def slash_penalty(interaction: Any, agent_id: str, ip: float = 0.0, du: float = 0.0) -> None:
    """Apply an IP/DU penalty to an agent."""
    bot_instance = get_active_bot()
    ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
    if not await _rate_limit(getattr(interaction, "user", None), "penalty", agent_id):
        await ctx.get_event_queue().put(
            SimulationEvent(
                type="moderation",
                data={"command": "penalty", "agent_id": agent_id, "violation": True},
            )
        )
        await interaction.response.send_message("rate limited", ephemeral=True)
        return
    await ctx.get_event_queue().put(
        SimulationEvent(
            type="moderation",
            data={"command": "penalty", "agent_id": agent_id, "ip": ip, "du": du},
        )
    )
    await interaction.response.send_message("penalty applied", ephemeral=True)


@bot.tree.command(name="unmute")
async def slash_unmute(interaction: Any, agent_id: str) -> None:
    """Unmute an agent in the simulation."""
    bot_instance = get_active_bot()
    ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
    if not await _rate_limit(getattr(interaction, "user", None), "unmute", agent_id):
        await ctx.get_event_queue().put(
            SimulationEvent(
                type="moderation",
                data={"command": "unmute", "agent_id": agent_id, "violation": True},
            )
        )
        await interaction.response.send_message("rate limited", ephemeral=True)
        return
    await ctx.get_event_queue().put(
        SimulationEvent(type="moderation", data={"command": "unmute", "agent_id": agent_id})
    )
    await interaction.response.send_message("unmuted", ephemeral=True)


__all__ = ["slash_mute", "slash_penalty", "slash_reset_memory", "slash_unmute"]
