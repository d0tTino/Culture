import asyncio
import shutil
from pathlib import Path

import pytest

pytest.importorskip("asyncpg")
import asyncpg
import testing.postgresql

from src.interfaces import token_sql


@pytest.mark.integration
def test_token_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test save_token, get_token, and list_tokens using in-memory SQLite."""
    db_url = "sqlite+aiosqlite:///:memory:"
    monkeypatch.setenv("DISCORD_TOKENS_DB_URL", db_url)

    # reset globals in case other tests have used them
    token_sql._engine = None  # type: ignore[attr-defined]
    token_sql._sessionmaker = None  # type: ignore[attr-defined]

    async def run_tests() -> None:
        # Saving and retrieving tokens
        await token_sql.save_token("agent_a", "tok_a")
        assert await token_sql.get_token("agent_a") == "tok_a"
        assert await token_sql.get_token("missing") is None

        await token_sql.save_token("agent_b", "tok_b")
        await token_sql.save_token("agent_a", "tok_c")  # update existing

        tokens = await token_sql.list_tokens()
        assert set(tokens) == {"tok_b", "tok_c"}

    asyncio.run(run_tests())

    # cleanup engine
    if token_sql._engine is not None:  # type: ignore[attr-defined]
        asyncio.run(token_sql._engine.dispose())
    token_sql._engine = None  # type: ignore[attr-defined]
    token_sql._sessionmaker = None  # type: ignore[attr-defined]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_token_sql_postgres_concurrent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate parallel inserts/lookups against a Postgres backend."""
    if shutil.which("initdb") is None:
        pytest.skip("PostgreSQL binaries not available")
    sql_path = Path("scripts/init_discord_tokens.sql")
    sql = sql_path.read_text()

    with testing.postgresql.Postgresql() as pg:
        conn = await asyncpg.connect(pg.url())
        await conn.execute(sql)
        await conn.close()

        monkeypatch.setenv("DISCORD_TOKENS_DB_URL", pg.url())
        token_sql._engine = None  # type: ignore[attr-defined]
        token_sql._sessionmaker = None  # type: ignore[attr-defined]

        async def worker(i: int) -> str | None:
            agent = f"agent_{i}"
            await token_sql.save_token(agent, f"tok_{i}")
            return await token_sql.get_token(agent)

        results = await asyncio.gather(*(worker(i) for i in range(10)))

        assert results == [f"tok_{i}" for i in range(10)]
        tokens = await token_sql.list_tokens()
        assert set(tokens) == {f"tok_{i}" for i in range(10)}

        if token_sql._engine is not None:  # type: ignore[attr-defined]
            await token_sql._engine.dispose()
        token_sql._engine = None  # type: ignore[attr-defined]
        token_sql._sessionmaker = None  # type: ignore[attr-defined]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_token_sql_concurrent_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure concurrent updates on the same agent_id do not deadlock."""
    if shutil.which("initdb") is None:
        pytest.skip("PostgreSQL binaries not available")
    sql_path = Path("scripts/init_discord_tokens.sql")
    sql = sql_path.read_text()

    with testing.postgresql.Postgresql() as pg:
        conn = await asyncpg.connect(pg.url())
        await conn.execute(sql)
        await conn.close()

        monkeypatch.setenv("DISCORD_TOKENS_DB_URL", pg.url())
        token_sql._engine = None  # type: ignore[attr-defined]
        token_sql._sessionmaker = None  # type: ignore[attr-defined]

        async def updater(token: str) -> None:
            await token_sql.save_token("agent_shared", token)

        await asyncio.gather(*(updater(f"tok_{i}") for i in range(5)))

        stored = await token_sql.get_token("agent_shared")
        assert stored in {f"tok_{i}" for i in range(5)}
        tokens = await token_sql.list_tokens()
        assert tokens == [stored]

        if token_sql._engine is not None:  # type: ignore[attr-defined]
            await token_sql._engine.dispose()
        token_sql._engine = None  # type: ignore[attr-defined]
        token_sql._sessionmaker = None  # type: ignore[attr-defined]
