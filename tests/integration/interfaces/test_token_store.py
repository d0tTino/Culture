from pathlib import Path

import pytest

pytest.importorskip("asyncpg")
import shutil

import asyncpg
import testing.postgresql

from src.infra import config
from src.interfaces import token_store


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_token(monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("initdb") is None:
        pytest.skip("PostgreSQL binaries not available")
    sql_path = Path("scripts/init_discord_tokens.sql")
    sql = sql_path.read_text()

    with testing.postgresql.Postgresql() as pg:
        conn = await asyncpg.connect(pg.url())
        await conn.execute(sql)
        await conn.execute(
            "INSERT INTO discord_tokens(agent_id, token) VALUES($1, $2)",
            "agent_a",
            "tok_a",
        )
        await conn.close()

        monkeypatch.setitem(config.CONFIG_OVERRIDES, "DISCORD_TOKENS_DB_URL", pg.url())
        # Reset pool in case previous tests populated it
        token_store._pool = None  # type: ignore[attr-defined]

        assert await token_store.get_token("agent_a") == "tok_a"
        assert await token_store.get_token("missing") is None
