from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class RankingWeights:
    recency: float = 0.45
    endorsement: float = 0.35
    proximity: float = 0.20


@dataclass(frozen=True)
class QueryPagination:
    page: int = 1
    page_size: int = 20

    @property
    def offset(self) -> int:
        return max(0, (self.page - 1) * self.page_size)


@dataclass(frozen=True)
class QueryFilters:
    agent_id: str | None = None
    entry_types: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    search: str | None = None
    start_step: int | None = None
    end_step: int | None = None


@dataclass(frozen=True)
class TimelineQueryDTO:
    filters: QueryFilters = field(default_factory=QueryFilters)
    pagination: QueryPagination = field(default_factory=QueryPagination)
    anchor_entry_id: str | None = None
    ranking: RankingWeights = field(default_factory=RankingWeights)


@dataclass(frozen=True)
class ThreadQueryDTO:
    root_entry_id: str
    pagination: QueryPagination = field(default_factory=QueryPagination)
    ranking: RankingWeights = field(default_factory=RankingWeights)


@dataclass(frozen=True)
class ProposalStatusQueryDTO:
    proposal_id: str
    pagination: QueryPagination = field(default_factory=QueryPagination)


@dataclass(frozen=True)
class AgentContributionQueryDTO:
    agent_id: str
    filters: QueryFilters = field(default_factory=QueryFilters)
    pagination: QueryPagination = field(default_factory=QueryPagination)
    ranking: RankingWeights = field(default_factory=RankingWeights)


@dataclass(frozen=True)
class CausalChainQueryDTO:
    entry_id: str
    depth: int = 3


DigestPeriod = Literal["daily", "weekly"]


@dataclass(frozen=True)
class StoryDigestDTO:
    period: DigestPeriod
    start_step: int
    end_step: int
    generated_at_step: int
    highlights: tuple[str, ...]
    source_entry_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankedEntryDTO:
    entry_id: str
    step: int
    agent_id: str
    entry_type: str
    content_summary: str
    parent_entry_id: str | None
    tags: tuple[str, ...]
    score: float
    signals: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PagedQueryResultDTO:
    total: int
    page: int
    page_size: int
    items: tuple[RankedEntryDTO, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "page": self.page,
            "page_size": self.page_size,
            "items": [item.to_dict() for item in self.items],
        }
