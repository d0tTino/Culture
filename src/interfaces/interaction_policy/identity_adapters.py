from __future__ import annotations

from typing import Any

from src.interfaces.interaction_schema import InteractionContext, InteractionIdentity


def discord_identity(*, user: Any, channel: Any) -> InteractionIdentity:
    user_id = getattr(user, "id", None)
    channel_id = getattr(channel, "id", None)
    is_admin = bool(getattr(getattr(user, "guild_permissions", None), "administrator", False))
    return InteractionIdentity(
        principal_id=str(user_id) if user_id is not None else "",
        source="discord",
        channel_id=str(channel_id) if channel_id is not None else None,
        is_admin=is_admin,
        attributes={"discord_user_id": str(user_id) if user_id is not None else ""},
    )


def discord_interaction_context(*, user: Any, channel: Any) -> InteractionContext:
    identity = discord_identity(user=user, channel=channel)
    return InteractionContext(
        sender_id=identity.principal_id or "human",
        channel_id=identity.channel_id,
        source=identity.source,
        metadata={"identity": identity.model_dump()},
    )
