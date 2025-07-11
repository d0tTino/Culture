from __future__ import annotations

from pydantic import BaseModel

from src.infra import llm_client


class Quest(BaseModel):
    """Simple quest data structure."""

    id: int
    title: str
    description: str
    progress: int = 0
    status: str = "pending"


QUESTS: list[Quest] = []


def generate_quest(
    prompt: str,
    *,
    model: str = "mistral:latest",
    temperature: float = 0.2,
) -> Quest | None:
    """Generate a quest using the LLM client and store it."""

    result = llm_client.generate_structured_output(
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
            quest = Quest.model_validate(result)  # type: ignore[arg-type]
        except Exception:
            return None
    QUESTS.append(quest)
    return quest


def get_quests() -> list[Quest]:
    """Return the list of generated quests."""

    return list(QUESTS)
