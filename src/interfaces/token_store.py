import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from src.infra import config

if TYPE_CHECKING:  # pragma: no cover - typing only
    import asyncpg
else:  # pragma: no cover - optional dependency
    asyncpg = None

logger = logging.getLogger(__name__)

_pool: Any | None = None


async def _get_pool() -> Any:
    global asyncpg
    if asyncpg is None:
        try:
            import asyncpg as _asyncpg
        except Exception as exc:
            raise RuntimeError("asyncpg is required for Discord token store") from exc
        asyncpg = _asyncpg

    global _pool
    if _pool is None:
        db_url = os.environ.get("DISCORD_TOKENS_DB_URL") or str(
            config.get_config("DISCORD_TOKENS_DB_URL") or ""
        )
        if not db_url:
            raise RuntimeError("DISCORD_TOKENS_DB_URL is not set")
        _pool = await asyncpg.create_pool(db_url)
    return _pool


async def get_token(agent_id: str) -> Optional[str]:
    """Return the Discord token for the given agent ID if present."""
    pool = await _get_pool()
    row = await pool.fetchrow(
        "SELECT token FROM discord_tokens WHERE agent_id=$1",
        agent_id,
    )
    return row["token"] if row else None


async def lookup_token(agent_id: str) -> Optional[str]:
    """Return or automatically assign the Discord token for ``agent_id``."""
    token = await get_token(agent_id)
    if token:
        return token

    env_tokens = os.environ.get("DISCORD_BOT_TOKEN") or str(
        config.get_config("DISCORD_BOT_TOKEN") or ""
    )
    tokens = [t.strip() for t in env_tokens.split(",") if t.strip()]
    if not tokens:
        return None

    token = tokens[hash(agent_id) % len(tokens)]
    try:
        await save_token(agent_id, token)
    except Exception:
        logger.exception("Failed to persist token mapping")
    return token


async def save_token(agent_id: str, token: str) -> None:
    """Insert or update a Discord token for the given agent ID."""
    pool = await _get_pool()
    await pool.execute(
        "INSERT INTO discord_tokens(agent_id, token) VALUES($1, $2)\n"
        "ON CONFLICT(agent_id) DO UPDATE SET token = EXCLUDED.token",
        agent_id,
        token,
    )


async def list_tokens() -> list[str]:
    """Return all stored Discord tokens."""
    pool = await _get_pool()
    rows = await pool.fetch("SELECT token FROM discord_tokens")
    return [row["token"] for row in rows]
