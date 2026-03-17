from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def command_span(name: str, interaction: Any, *, agent_id: str | None = None) -> Iterator[Any]:
    from src.interfaces.discord_bot import command_span as legacy

    return legacy(name, interaction, agent_id=agent_id)


def message_span(message: Any) -> Iterator[Any]:
    from src.interfaces.discord_bot import message_span as legacy

    return legacy(message)


async def send_interaction_response(interaction: Any, content: str, **kwargs: Any) -> None:
    from src.interfaces.discord_bot import send_interaction_response as legacy

    await legacy(interaction, content, **kwargs)
