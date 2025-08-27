from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from typing import cast

from pydantic import BaseModel

from src.infra import llm_client
from src.infra.ledger import ledger


class Quest(BaseModel):
    """Simple quest data structure."""

    id: int
    title: str
    description: str
    progress: int = 0
    status: str = "pending"


QUESTS: list[Quest] = []

_quest_task: asyncio.Task[None] | None = None


async def _quest_loop(interval: float) -> None:
    while True:
        try:
            await generate_quest("Create a new quest for the agents")
        except Exception:  # pragma: no cover - defensive
            pass
        await asyncio.sleep(interval)


def start_quest_generation(interval: float = 60.0) -> None:
    """Start periodic quest generation in the background.

    Parameters
    ----------
    interval:
        Number of seconds between quest generations. If ``interval`` is
        less than or equal to zero the generator is not started.
    """
    global _quest_task
    if interval <= 0:
        return
    if _quest_task is None or _quest_task.done():
        _quest_task = asyncio.create_task(_quest_loop(interval))


async def stop_quest_generation() -> None:
    """Stop the running background quest generator task."""
    global _quest_task
    if _quest_task:
        _quest_task.cancel()
        try:
            await _quest_task
        except asyncio.CancelledError:  # pragma: no cover - expected
            pass
        _quest_task = None


async def generate_quest(
    prompt: str,
    *,
    model: str = "mistral:latest",
    temperature: float = 0.2,
) -> Quest | None:
    """Generate a quest using the LLM client and persist it.

    Parameters
    ----------
    prompt:
        Natural language instruction describing the quest to create.
    model:
        Name of the model used to generate the quest.
    temperature:
        Sampling temperature for the model.

    Returns
    -------
    Quest | None
        The generated :class:`Quest` instance or ``None`` if generation
        fails.
    """

    result = await llm_client.async_generate_structured_output(
        prompt,
        Quest,
        model=model,
        temperature=temperature,
    )
    if result is None:
        return None
    if isinstance(result, Quest):
        quest = result
    elif isinstance(result, dict):
        quest = Quest(**result)
    else:
        # Defensive: unexpected return type
        try:
            quest = Quest.model_validate(result)
        except Exception:
            return None
    QUESTS.append(quest)
    try:
        ledger.record_quest(
            quest.id,
            quest.title,
            quest.description,
            quest.progress,
            quest.status,
        )
    except Exception:
        pass
    return quest


def get_quests() -> list[Quest]:
    """Return the list of generated quests.

    Quests are loaded from the ledger if possible. If ledger access fails,
    the in-memory list of quests is returned instead.
    """

    try:
        rows = cast(Iterable[Mapping[str, object]], ledger.get_quests())
        return [Quest(**r) for r in rows]
    except Exception:  # pragma: no cover - defensive
        return list(QUESTS)
