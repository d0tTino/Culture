from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, cast

from src.interfaces.discord_bot.commands.admin import admin_command_registry
from src.interfaces.discord_bot.commands.public import public_command_registry
from src.interfaces.discord_bot.permissions import POLICY_GATEWAY


async def _policy_wrapped(handler: Callable[..., Awaitable[None]], interaction: Any, *, command_name: str, admin_only: bool, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    if not await POLICY_GATEWAY.enforce(interaction, command_name, admin_only=admin_only):
        return
    await handler(interaction, *args, **kwargs)


def get_registry_table() -> list[dict[str, Any]]:
    return [*public_command_registry(), *admin_command_registry()]


def register_slash_commands(tree: Any) -> dict[str, Callable[..., Any]]:
    from src.interfaces import discord_bot

    commands: dict[str, Callable[..., Any]] = {}

    if tree is not discord_bot.bot.tree:
        from src.interfaces.discord_moderation import register_moderation_commands

        register_moderation_commands(tree, resolve_context=discord_bot._global_context_resolver)

    def _register(defn: dict[str, Any]) -> None:
        name = cast(str, defn["name"])
        callback = cast(Callable[..., Awaitable[None]], defn["handler"])
        admin_only = bool(defn.get("admin_only", False))

        @wraps(callback)
        async def guarded(*args: Any, **kwargs: Any) -> None:
            interaction = args[0] if args else kwargs.get("interaction")
            await _policy_wrapped(
                callback,
                interaction,
                command_name=name,
                admin_only=admin_only,
                args=args[1:] if args else tuple(),
                kwargs=kwargs,
            )

        try:
            guarded.__signature__ = inspect.signature(callback)
        except (TypeError, ValueError):
            pass
        guarded.__annotations__ = dict(getattr(callback, "__annotations__", {}) or {})
        handler = guarded
        descriptions = cast(dict[str, str] | None, defn.get("descriptions"))
        moderation_action = cast(str | None, defn.get("moderation_action"))
        if moderation_action is not None:
            from src.interfaces.discord_moderation import moderation_rate_limit

            handler = cast(Callable[..., Awaitable[None]], moderation_rate_limit(moderation_action)(handler))
        if descriptions:
            handler = cast(Callable[..., Awaitable[None]], discord_bot.app_commands.describe(**descriptions)(handler))
        setattr(callback, "callback", callback)
        commands[name] = tree.command(name=name)(handler)

    for definition in get_registry_table():
        _register(definition)

    return commands
