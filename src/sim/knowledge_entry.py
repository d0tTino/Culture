"""Typed ontology for knowledge board entries and relationships."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class _StrEnum(str, Enum):
    pass


class KnowledgeEntryType(_StrEnum):
    """Canonical entry types shared by all knowledge board backends."""

    IDEA = "idea"
    PROPOSAL = "proposal"
    VOTE = "vote"
    LAW = "law"
    EVENT = "event"
    RETIREMENT = "retirement"
    GOVERNANCE_DECISION = "governance_decision"
    ENDORSEMENT = "endorsement"
    NOTE = "note"
    PROPOSAL_RESULT = "proposal_result"
    CHECKPOINT = "checkpoint"
    PROJECT_UPDATE = "project_update"
    SPAWN_EVENT = "spawn_event"
    WORLD_EVENT = "world_event"
    HUMAN_MESSAGE = "human_message"


class KnowledgeRelationshipType(_StrEnum):
    """Typed edge labels used by graph-backed knowledge boards."""

    AUTHORED = "AUTHORED"
    ENDORSED = "ENDORSED"
    VOTED = "VOTED"
    AMENDS = "AMENDS"
    SUPERCEDES = "SUPERCEDES"


@dataclass(slots=True)
class KnowledgeEntry:
    """Typed payload describing a single board entry and canonical references."""

    content_full: str
    entry_type: KnowledgeEntryType | str
    content_display: str | None = None
    content_summary: str | None = None
    tags: list[str] | None = None
    parent_entry_id: str | None = None
    target_agent_id: str | None = None
    project_id: str | None = None
    governance_rule_id: str | None = None
    reference_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content_full, str):
            raise TypeError("content_full must be a string")
        self.entry_type = parse_entry_type(self.entry_type)

    def normalized_tags(self) -> list[str]:
        if not self.tags:
            return []
        seen: set[str] = set()
        result: list[str] = []
        for tag in self.tags:
            if isinstance(tag, str) and tag and tag not in seen:
                seen.add(tag)
                result.append(tag)
        return result


def parse_entry_type(raw_type: KnowledgeEntryType | str | None) -> KnowledgeEntryType:
    """Parse an entry type and enforce ontology strictness."""

    if isinstance(raw_type, KnowledgeEntryType):
        return raw_type
    if isinstance(raw_type, str):
        normalized = raw_type.strip().lower()
        try:
            return KnowledgeEntryType(normalized)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported knowledge entry_type '{raw_type}'") from exc
    return KnowledgeEntryType.NOTE


def migrate_legacy_entry_dict(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy entry dictionaries into canonical typed entry fields."""

    migrated = dict(entry)
    metadata = migrated.get("reference_metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    try:
        migrated["entry_type"] = parse_entry_type(migrated.get("entry_type")).value
    except ValueError:
        migrated["entry_type"] = KnowledgeEntryType.NOTE.value

    parent_entry_id = (
        migrated.get("parent_entry_id")
        or metadata.get("parent_entry_id")
        or metadata.get("parent")
    )
    target_agent_id = migrated.get("target_agent_id") or metadata.get("target_agent_id")
    project_id = migrated.get("project_id") or metadata.get("project_id")
    governance_rule_id = migrated.get("governance_rule_id") or metadata.get("governance_rule_id")

    migrated["parent_entry_id"] = str(parent_entry_id) if parent_entry_id else None
    migrated["target_agent_id"] = str(target_agent_id) if target_agent_id else None
    migrated["project_id"] = str(project_id) if project_id else None
    migrated["governance_rule_id"] = str(governance_rule_id) if governance_rule_id else None

    tags = migrated.get("tags")
    if not isinstance(tags, list):
        tags = []
    migrated["tags"] = [str(t) for t in tags if isinstance(t, str) and t]
    migrated["reference_metadata"] = metadata or None
    return migrated


__all__ = [
    "KnowledgeEntry",
    "KnowledgeEntryType",
    "KnowledgeRelationshipType",
    "migrate_legacy_entry_dict",
    "parse_entry_type",
]
