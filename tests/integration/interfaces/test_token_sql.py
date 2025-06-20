import asyncio

import pytest

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
