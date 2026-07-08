import hashlib

import pytest

from src.interfaces import token_store


class DummyPool:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def fetchrow(self, query: str, agent_id: str) -> dict[str, str] | None:
        token = self.data.get(agent_id)
        return {"token": token} if token else None

    async def execute(self, query: str, *args: object) -> None:
        agent_id, token = args[-2], args[-1]
        self.data[str(agent_id)] = str(token)

    async def fetch(self, query: str) -> list[dict[str, str]]:
        return [{"token": t} for t in self.data.values()]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_and_get_token(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = DummyPool()

    async def dummy_get_pool() -> DummyPool:
        return pool

    monkeypatch.setattr(token_store, "_get_pool", dummy_get_pool)
    monkeypatch.setattr(token_store, "_pool", pool)

    await token_store.save_token("agent_x", "tok_x")
    assert await token_store.get_token("agent_x") == "tok_x"
    assert await token_store.get_token("missing") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lookup_token_uses_deterministic_agent_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def missing_get_token(agent_id: str) -> None:
        return None

    saved_tokens: list[tuple[str, str]] = []

    async def dummy_save_token(agent_id: str, token: str) -> None:
        saved_tokens.append((agent_id, token))

    monkeypatch.setattr(token_store, "get_token", missing_get_token)
    monkeypatch.setattr(token_store, "save_token", dummy_save_token)
    tokens = ["tok_a", "tok_b", "tok_c", "tok_d"]
    monkeypatch.setenv("DISCORD_BOT_TOKEN", ",".join(tokens))
    stable_hash = int.from_bytes(hashlib.sha256(b"agent_x").digest()[:8], "big")
    expected_token = tokens[stable_hash % len(tokens)]

    first_token = await token_store.lookup_token("agent_x")
    second_token = await token_store.lookup_token("agent_x")

    assert first_token == expected_token
    assert second_token == expected_token
    assert saved_tokens == [("agent_x", expected_token), ("agent_x", expected_token)]
