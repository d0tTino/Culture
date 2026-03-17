from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

CommandHandler = Callable[..., Awaitable[None]]


def public_command_registry() -> list[dict[str, Any]]:
    from src.interfaces import discord_bot

    return [
        {"name": "status", "handler": discord_bot.slash_status},
        {"name": "stats", "handler": discord_bot.slash_stats},
        {"name": "help", "handler": discord_bot.slash_help},
        {"name": "start_here", "handler": discord_bot.slash_start_here},
        {"name": "kb", "handler": discord_bot.slash_kb},
        {"name": "kb_timeline", "handler": discord_bot.slash_kb_timeline},
        {"name": "kb_thread", "handler": discord_bot.slash_kb_thread},
        {"name": "kb_proposal_status", "handler": discord_bot.slash_kb_proposal_status},
        {"name": "kb_agent", "handler": discord_bot.slash_kb_agent},
        {"name": "kb_causal_chain", "handler": discord_bot.slash_kb_causal_chain},
        {"name": "kb_digest", "handler": discord_bot.slash_kb_digest},
        {"name": "propose", "handler": discord_bot.slash_propose},
        {"name": "propose_law", "handler": discord_bot.slash_propose_law},
        {"name": "vote", "handler": discord_bot.slash_vote},
        {"name": "speed", "handler": discord_bot.slash_speed},
        {"name": "gov", "handler": discord_bot.slash_gov},
    ]
