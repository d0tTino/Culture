from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CommandPolicyGateway:
    """Centralized policy gateway for all slash command checks."""

    async def enforce(self, interaction: Any, command_name: str, *, admin_only: bool = False) -> bool:
        from src.interfaces import discord_bot

        identity = discord_bot.discord_identity(
            user=getattr(interaction, "user", None),
            channel=getattr(interaction, "channel", None),
        )
        if not await discord_bot.check_command_rate_limit(identity):
            await discord_bot.send_interaction_response(
                interaction,
                "rate limit exceeded",
                ephemeral=True,
            )
            return False

        if admin_only:
            allowed = await discord_bot._has_control_command_permission(
                getattr(interaction, "user", None),
                getattr(interaction, "channel", None),
                command_name,
            )
            if not allowed:
                await discord_bot.send_interaction_response(interaction, "unauthorized", ephemeral=True)
                return False
        return True


POLICY_GATEWAY = CommandPolicyGateway()
