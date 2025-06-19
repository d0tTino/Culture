"""
Defines the Knowledge Board class for maintaining shared knowledge among agents.
"""

import logging
import uuid
from typing import Any, Generic, SupportsIndex, TypeVar

from typing_extensions import Self

from src.interfaces import metrics

# Configure logger
logger = logging.getLogger(__name__)

T = TypeVar("T")


class LoggingList(list[T], Generic[T]):
    def clear(self: Self) -> None:
        logger.info(f"LOGGING_LIST_DEBUG ({id(self)}): clear() called")
        super().clear()

    def __delitem__(self: Self, key: SupportsIndex | slice) -> None:
        if isinstance(key, slice) and key.start is None and key.stop is None and key.step is None:
            logger.info(f"LOGGING_LIST_DEBUG ({id(self)}): __delitem__[:] called (del self[:])")
        super().__delitem__(key)

    def __init__(self: Self, *args: Any, **kwargs: Any) -> None:
        logger.info(
            f"LOGGING_LIST_DEBUG: Initializing new LoggingList instance ({id(self)}) from args: {args}"
        )
        super().__init__(*args, **kwargs)

    # For MyPy, if we override methods that are checked for type hints:
    def append(self: Self, item: T) -> None:
        super().append(item)

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
        return {"entries": self.get_full_entries()}

    def get_recent_entries_for_prompt(self: Self, max_entries: int = 5) -> list[str]:
        """
        Returns a list of formatted strings for the most recent entries, suitable for an LLM prompt.

        Args:
            max_entries (int): The maximum number of recent entries to return.

        Returns:
            list[str]: A list of formatted strings, e.g., "[Step X, Agent Y]: Content..."
        """
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        if not self.entries:
            return ["(Knowledge Board is empty)"]

        # Get the most recent entries (up to max_entries)
        # Entries are appended, so the most recent are at the end of the list.
        recent_raw_entries = self.entries[-max_entries:]

        formatted_entries = []
        for entry in recent_raw_entries:
            step = entry.get("step", "N/A")
            agent_id = entry.get("agent_id", "Unknown Agent")
            content_summary = entry.get("content_summary", entry.get("content_full", "N/A"))
            # Truncate content for brevity in prompt if necessary
            max_content_len = 150  # Example max length
            if len(content_summary) > max_content_len:
                content_summary = content_summary[:max_content_len] + "..."

            formatted_entries.append(f"[Step {step}, {agent_id}]: {content_summary}")

        return formatted_entries

    def add_entry(self: Self, entry: str, agent_id: str, step: int) -> bool:
        """
        Adds an entry to the knowledge board.

        Args:
            entry (str): The knowledge entry to add to the board.
            agent_id (str): ID of the agent proposing the entry.
            step (int): The simulation step when this entry was proposed.

        Returns:
            bool: True if the entry was successfully added, False otherwise.
        """
        try:
            entry_id = str(uuid.uuid4())  # Unique ID for this board entry
            formatted_content = (
                f"Step {step} (Agent: {agent_id}): {entry}"  # Keep this for display
            )

            new_entry_dict = {
                "entry_id": entry_id,
                "step": step,
                "agent_id": agent_id,  # Store original proposer ID
                "content_full": entry,  # Store raw entry
                "content_display": formatted_content,  # Store formatted entry for display
            }
            self.entries.append(new_entry_dict)  # Append first
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

    def clear_board(self: Self) -> None:
        """Clears all entries from the Knowledge Board."""
        logger.info(
            f"KNOWLEDGE_BOARD_DEBUG: Clearing board. Instance ID: {id(self)}. Old entries list ID: {id(self.entries)}"
        )
        self.entries = LoggingList()  # Assign new LoggingList
        logger.info(
            f"KNOWLEDGE_BOARD_DEBUG: Board cleared. Instance ID: {id(self)}. New entries list ID: {id(self.entries)}"
        )
        metrics.KNOWLEDGE_BOARD_SIZE.set(len(self.entries))
