"""Defines the Knowledge Board class for maintaining shared knowledge."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Generic, SupportsIndex, TypeVar

from typing_extensions import Self

from src.infra import config
from src.interfaces import metrics
from src.shared.telemetry import trace_agent_action

from .version_vector import VersionVector

# Configure logger
logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(slots=True)
class BoardEntry:
    """Typed payload describing a knowledge board entry."""

    content_full: str
    entry_type: str
    content_display: str | None = None
    content_summary: str | None = None
    tags: Iterable[str] | None = None
    reference_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content_full, str):
            raise TypeError("content_full must be a string")
        if not self.entry_type:
            raise ValueError("entry_type must be provided")

    def normalized_tags(self) -> list[str]:
        """Return tags as a list preserving order without duplicates."""

        if self.tags is None:
            return []
        seen: set[str] = set()
        result: list[str] = []
        for tag in self.tags:
            if not isinstance(tag, str):
                continue
            if tag not in seen:
                seen.add(tag)
                result.append(tag)
        return result


_DEFAULT_ENTRY_TYPE = "note"

__all__ = ["BoardEntry", "KnowledgeBoard", "prepare_entry_payload"]


def prepare_entry_payload(
    entry: str | BoardEntry, agent_id: str, step: int
) -> tuple[str, dict[str, Any]]:
    if isinstance(entry, BoardEntry):
        payload = entry
    else:
        payload = BoardEntry(entry, entry_type=_DEFAULT_ENTRY_TYPE)

    max_length = int(getattr(config, "MAX_KB_ENTRY_LENGTH", 2048))
    content_full = payload.content_full[:max_length]
    content_summary = (
        payload.content_summary[:max_length]
        if isinstance(payload.content_summary, str)
        else content_full
    )
    explicit_display = (
        payload.content_display[:max_length]
        if isinstance(payload.content_display, str)
        else None
    )
    display_content = explicit_display or (
        f"Step {step} (Agent: {agent_id}): {content_full}"
    )
    tags = payload.normalized_tags()
    reference_metadata = (
        dict(payload.reference_metadata)
        if isinstance(payload.reference_metadata, dict)
        else None
    )

    if (
        payload.entry_type == _DEFAULT_ENTRY_TYPE
        and not tags
        and reference_metadata is None
        and payload.content_summary is None
        and explicit_display is None
    ):
        seed = f"{agent_id}:{step}:{content_full}"
    else:
        seed_payload: dict[str, Any] = {
            "agent_id": agent_id,
            "step": step,
            "entry_type": payload.entry_type,
            "content_full": content_full,
            "content_summary": content_summary,
            "tags": tags,
            "reference_metadata": reference_metadata,
        }
        if explicit_display is not None:
            seed_payload["content_display"] = explicit_display
        seed = json.dumps(seed_payload, sort_keys=True, default=str)

    entry_id = str(uuid.uuid5(uuid.NAMESPACE_OID, seed))
    entry_dict = {
        "entry_id": entry_id,
        "step": step,
        "agent_id": agent_id,
        "entry_type": payload.entry_type,
        "tags": tags,
        "reference_metadata": reference_metadata,
        "content_full": content_full,
        "content_display": display_content,
        "content_summary": content_summary,
    }
    return entry_id, entry_dict


class LoggingList(list[T], Generic[T]):
    """List that emits a brief debug message when modified."""

    def _log_change(self: Self) -> None:
        logger.debug("LoggingList length now %d", len(self))

    def __init__(self: Self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        logger.debug("LoggingList initialized with %d entries", len(self))

    def clear(self: Self) -> None:
        super().clear()
        self._log_change()

    def __delitem__(self: Self, key: SupportsIndex | slice) -> None:
        super().__delitem__(key)
        self._log_change()

    # For MyPy, if we override methods that are checked for type hints:
    def append(self: Self, item: T) -> None:
        super().append(item)
        self._log_change()

    # __setitem__ is not overridden to keep superclass type checking intact


class KnowledgeBoard:
    """
    Represents a shared knowledge board that agents can read from and eventually write to.
    Maintains a list of knowledge entries as structured dictionaries.
    """

    entries: LoggingList[dict[str, Any]]

    def __init__(self: Self, entries: list[dict[str, Any]] | None = None) -> None:
        """
        Initialize an empty knowledge board.
        """
        # If initial entries are provided, use them, otherwise start with an empty LoggingList
        if entries is not None:
            self.entries = LoggingList(entries)
        else:
            self.entries = LoggingList()
        self.vector = VersionVector()
        # Lock to synchronize concurrent access to ``entries``
        self.lock = asyncio.Lock()
        logger.info(
            f"KnowledgeBoard initialized. Instance ID: {id(self)}. Entries list ID: {id(self.entries)} type: {type(self.entries)}"
        )
        metrics.KNOWLEDGE_BOARD_SIZE.set(len(self.entries))


    def get_state(self: Self, max_entries: int = 10) -> list[str]:
        """
        Returns the current state of the knowledge board, limited to the most recent entries.

        Args:
            max_entries (int): Maximum number of entries to return, starting from most recent.
                               Default is 10.

        Returns:
            list[str]: The most recent entries on the board, up to max_entries.
        """
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        with trace_agent_action("knowledge_board.get_state", max_entries=max_entries):
            # Return the display_content for the most recent entries, up to max_entries
            recent_entries = (
                [entry["content_display"] for entry in self.entries[-max_entries:]]
                if self.entries
                else []
            )
            logger.debug(
                "KnowledgeBoard: Returning "
                f"{len(recent_entries)} entries (of {len(self.entries)} total)"
            )
            return recent_entries

    def get_full_entries(self: Self) -> list[dict[str, Any]]:
        """Returns a copy of all entries on the board."""
        return list(self.entries)  # Return a copy

    def to_dict(self: Self) -> dict[str, Any]:
        """Serialize the knowledge board to a dictionary."""
        return {"entries": self.get_full_entries(), "vector": self.vector.to_dict()}

    def replace_entries(self: Self, entries: list[dict[str, Any]]) -> None:
        """Replace the board contents with precomputed entries."""
        self.entries = LoggingList(list(entries))
        metrics.KNOWLEDGE_BOARD_SIZE.set(len(self.entries))

    def get_recent_entries_for_prompt(self: Self, max_entries: int = 5) -> list[str]:
        """
        Returns a list of formatted strings for the most recent entries,
        suitable for an LLM prompt. ``content_summary`` is cast to ``str`` to
        gracefully handle ``None`` values before length checks.

        Args:
            max_entries (int): The maximum number of recent entries to return.

        Returns:
            list[str]: A list of formatted strings, e.g., "[Step X, Agent Y]: Content..."
        """
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        with trace_agent_action(
            "knowledge_board.get_recent_entries_for_prompt",
            max_entries=max_entries,
        ):
            if not self.entries:
                return ["(Knowledge Board is empty)"]

            # Get the most recent entries (up to max_entries)
            # Entries are appended, so the most recent are at the end of the list.
            recent_raw_entries = self.entries[-max_entries:]

            formatted_entries = []
            for entry in recent_raw_entries:
                step = entry.get("step", "N/A")
                agent_id = entry.get("agent_id", "Unknown Agent")
                content_summary = entry.get("content_summary") or entry.get(
                    "content_full",
                    "N/A",
                )
                # Cast to string so None or other non-string values don't raise
                # errors when we check the length for truncation
                content_summary = str(content_summary)
                # Truncate content for brevity in prompt if necessary
                max_content_len = 150  # Example max length
                if len(content_summary) > max_content_len:
                    content_summary = content_summary[:max_content_len] + "..."

                formatted_entries.append(f"[Step {step}, {agent_id}]: {content_summary}")

            return formatted_entries

    def add_entry(
        self: Self,
        entry: str | BoardEntry,
        agent_id: str,
        step: int,
        vector: dict[str, int] | None = None,
    ) -> bool:
        """
        Adds an entry to the knowledge board.

        Args:
            entry: The knowledge entry to add to the board.
            agent_id (str): ID of the agent proposing the entry.
            step (int): The simulation step when this entry was proposed.

        Returns:
            bool: True if the entry was successfully added, False otherwise.
        """
        try:
            with trace_agent_action("knowledge_board.add_entry", agent_id=agent_id, step=step):
                entry_id, new_entry_dict = prepare_entry_payload(entry, agent_id, step)
                self.entries.append(new_entry_dict)  # Append first
                if vector is not None:
                    self.vector.merge(VersionVector(vector))
                else:
                    self.vector.increment(agent_id)

                # Enforce maximum board size
                max_entries = int(config.MAX_KB_ENTRIES)
                if len(self.entries) > max_entries:
                    excess = len(self.entries) - max_entries
                    del self.entries[:excess]
                    logger.info(
                        "KnowledgeBoard: pruned %s old entries to maintain max size %s",
                        excess,
                        max_entries,
                    )

                metrics.KNOWLEDGE_BOARD_SIZE.set(len(self.entries))

                logger.info(  # Log after append
                    f"KnowledgeBoard: Added entry ID {entry_id} by {agent_id} at step {step}: '{entry}'. "
                    f"New board size: {len(self.entries)}. Instance ID: {id(self)}. Entries list ID: {id(self.entries)}"
                )
                logger.debug(
                    f"KB_ADD_ENTRY_CONTENT_DEBUG: Instance ID {id(self)}, Entries list ID {id(self.entries)}, Content via list(): {list(self.entries)}, Direct repr: {self.entries!r}"
                )

                # Optional: Limit board size if needed (e.g., keep only last 100 entries)
                # max_board_size = 100
                # if len(self.entries) > max_board_size:
                #     self.entries = self.entries[-max_board_size:]

                return True
        except Exception as e:
            logger.error(f"Failed to add entry to knowledge board: {e}")
            return False

    def add_law_proposal(
        self: Self,
        proposal: str,
        agent_id: str,
        step: int,
        vector: dict[str, int] | None = None,
    ) -> bool:
        """Record a law proposal on the board."""
        return self.add_entry(
            BoardEntry(
                content_full=f"Law proposed: {proposal}",
                entry_type="proposal",
                tags=["governance", "proposal"],
            ),
            agent_id,
            step,
            vector,
        )

    def clear_board(self: Self) -> None:
        """Clears all entries from the Knowledge Board."""
        logger.info(
            f"KNOWLEDGE_BOARD_DEBUG: Clearing board. Instance ID: {id(self)}. Old entries list ID: {id(self.entries)}"
        )
        self.entries = LoggingList()  # Assign new LoggingList
        self.vector = VersionVector()
        logger.info(
            f"KNOWLEDGE_BOARD_DEBUG: Board cleared. Instance ID: {id(self)}. New entries list ID: {id(self.entries)}"
        )
        metrics.KNOWLEDGE_BOARD_SIZE.set(len(self.entries))
