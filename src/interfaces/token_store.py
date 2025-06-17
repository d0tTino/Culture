import logging
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
        db_url = str(config.get_config("DISCORD_TOKENS_DB_URL") or "")
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
