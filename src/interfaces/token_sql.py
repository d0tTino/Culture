"""SQLAlchemy-based Discord token store."""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import Column, MetaData, String, Table, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.infra import config

metadata = MetaData()

discord_tokens = Table(
    "discord_tokens",
    metadata,
    Column("agent_id", String, primary_key=True),
    Column("token", String, nullable=False),
)

_engine: Any | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


async def _get_session() -> AsyncSession:
    global _engine, _sessionmaker
    if _sessionmaker is None:
        db_url = os.environ.get("DISCORD_TOKENS_DB_URL") or str(
            config.get_config("DISCORD_TOKENS_DB_URL") or ""
        )
        if not db_url:
            raise RuntimeError("DISCORD_TOKENS_DB_URL is not set")
        _engine = create_async_engine(db_url)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
        async with _engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
    return _sessionmaker()


async def get_token(agent_id: str) -> str | None:
    async with await _get_session() as session:
        result = await session.execute(
            select(discord_tokens.c.token).where(discord_tokens.c.agent_id == agent_id)
        )
        row = result.first()
        return row[0] if row else None


async def save_token(agent_id: str, token: str) -> None:
    async with await _get_session() as session:
        stmt = text(
            "INSERT INTO discord_tokens(agent_id, token) VALUES (:agent_id, :token) "
            "ON CONFLICT(agent_id) DO UPDATE SET token = excluded.token"
        )
        await session.execute(stmt, {"agent_id": agent_id, "token": token})
        await session.commit()


async def list_tokens() -> list[str]:
    async with await _get_session() as session:
        result = await session.execute(select(discord_tokens.c.token))
        return [row[0] for row in result.fetchall()]
