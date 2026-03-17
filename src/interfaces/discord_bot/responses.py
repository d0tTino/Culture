from __future__ import annotations

from typing import Any


def build_help_text() -> str:
    from src.interfaces.discord_bot import build_help_text as legacy

    return legacy()


def create_onboarding_embed(channel_id: int) -> Any:
    from src.interfaces.discord_bot import create_onboarding_embed as legacy

    return legacy(channel_id)


def scenario_intro_cards() -> list[dict[str, str]]:
    from src.interfaces.discord_bot import scenario_intro_cards as legacy

    return legacy()


def format_explainable_acknowledgement(result: Any) -> str:
    from src.interfaces.discord_bot import format_explainable_acknowledgement as legacy

    return legacy(result)
