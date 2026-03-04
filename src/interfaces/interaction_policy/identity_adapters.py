from __future__ import annotations

from typing import Any

from src.interfaces.interaction_schema import InteractionContext


def discord_interaction_context(*, user: Any, channel: Any) -> InteractionContext:
    user_id = getattr(user, "id", None)
    channel_id = getattr(channel, "id", None)
    return InteractionContext(
        sender_id=str(user_id) if user_id is not None else "human",
        channel_id=str(channel_id) if channel_id is not None else None,
        source="discord",
    )
