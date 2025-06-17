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
async def test_save_token(monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("initdb") is None:
        pytest.skip("PostgreSQL binaries not available")
    sql_path = Path("scripts/init_discord_tokens.sql")
    sql = sql_path.read_text()

    with testing.postgresql.Postgresql() as pg:
        conn = await asyncpg.connect(pg.url())
        await conn.execute(sql)
        await conn.close()

        monkeypatch.setitem(config.CONFIG_OVERRIDES, "DISCORD_TOKENS_DB_URL", pg.url())
        token_store._pool = None  # type: ignore[attr-defined]

        await token_store.save_token("agent_z", "tok_z")
        assert await token_store.get_token("agent_z") == "tok_z"
