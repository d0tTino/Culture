from __future__ import annotations

from typing import Any


def admin_command_registry() -> list[dict[str, Any]]:
    from src.interfaces import discord_bot

    return [
        {"name": "pause", "handler": discord_bot.slash_pause, "admin_only": True},
        {"name": "resume", "handler": discord_bot.slash_resume, "admin_only": True},
        {"name": "pause_all", "handler": discord_bot.slash_pause_all, "admin_only": True},
        {
            "name": "kill_agent",
            "handler": discord_bot.slash_kill_agent,
            "admin_only": True,
            "descriptions": {"agent_id": "ID of the agent to kill"},
        },
        {
            "name": "nudge",
            "handler": discord_bot.slash_nudge,
            "admin_only": True,
            "descriptions": {"prompt": "Prompt to nudge the simulation"},
            "moderation_action": "nudge",
        },
        {"name": "start", "handler": discord_bot.slash_start, "admin_only": True, "moderation_action": "start"},
        {"name": "stop", "handler": discord_bot.slash_stop, "admin_only": True, "moderation_action": "stop"},
        {
            "name": "spawn",
            "handler": discord_bot.slash_spawn,
            "admin_only": True,
            "moderation_action": "spawn",
            "descriptions": {
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
        },
        {"name": "kill", "handler": discord_bot.slash_kill, "admin_only": True},
        {"name": "set_max_rate", "handler": discord_bot.slash_set_max_rate, "admin_only": True},
        {"name": "set_speed", "handler": discord_bot.slash_set_speed, "admin_only": True},
        {"name": "event", "handler": discord_bot.slash_event, "admin_only": True},
        {"name": "misbehavior_log", "handler": discord_bot.slash_misbehavior_log, "admin_only": True},
    ]
