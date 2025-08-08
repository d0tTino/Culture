"""Discord moderation commands with OPA-based rate limiting."""

from typing import Any

from src.interfaces.dashboard_backend import (
    DEFAULT_CONTEXT,
    SimulationEvent,
)
from src.interfaces.discord_bot import bot, get_active_bot
from src.utils.policy import evaluate_with_opa


async def _rate_limit(user: Any, action: str) -> bool:
    """Use OPA to determine if the action is allowed for the user."""
    user_id = str(getattr(user, "id", ""))
    allow, _ = await evaluate_with_opa(f"{user_id}:{action}")
    return allow


@bot.tree.command(name="mute")
async def slash_mute(interaction: Any, agent_id: str) -> None:
    """Mute an agent in the simulation."""
    if not await _rate_limit(getattr(interaction, "user", None), "mute"):
        await interaction.response.send_message("rate limited", ephemeral=True)
        return
    bot_instance = get_active_bot()
    ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
    await ctx.get_event_queue().put(
        SimulationEvent(
            type="moderation", data={"command": "mute", "agent_id": agent_id}
        )
    )
    await interaction.response.send_message("muted", ephemeral=True)


@bot.tree.command(name="reset_memory")
async def slash_reset_memory(interaction: Any, agent_id: str) -> None:
    """Reset the memory of an agent."""
    if not await _rate_limit(getattr(interaction, "user", None), "reset_memory"):
        await interaction.response.send_message("rate limited", ephemeral=True)
        return
    bot_instance = get_active_bot()
    ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
    await ctx.get_event_queue().put(
        SimulationEvent(
            type="moderation", data={"command": "reset_memory", "agent_id": agent_id}
        )
    )
    await interaction.response.send_message("memory reset", ephemeral=True)


@bot.tree.command(name="penalty")
async def slash_penalty(
    interaction: Any, agent_id: str, ip: float = 0.0, du: float = 0.0
) -> None:
    """Apply an IP/DU penalty to an agent."""
    if not await _rate_limit(getattr(interaction, "user", None), "penalty"):
        await interaction.response.send_message("rate limited", ephemeral=True)
        return
    bot_instance = get_active_bot()
    ctx = bot_instance.context if bot_instance is not None else DEFAULT_CONTEXT
    await ctx.get_event_queue().put(
        SimulationEvent(
            type="moderation",
            data={"command": "penalty", "agent_id": agent_id, "ip": ip, "du": du},
        )
    )
    await interaction.response.send_message("penalty applied", ephemeral=True)


__all__ = ["slash_mute", "slash_penalty", "slash_reset_memory"]

