from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from src.sim.knowledge_board_protocol import EntryStore
from src.sim.knowledge_entry import KnowledgeEntry


@dataclass(slots=True)
class GovernanceMutation:
    """A governance-state mutation with explicit rollback token support."""

    apply: Callable[[], Any]
    rollback: Callable[[Any], None]


class KnowledgeGovernanceTransaction:
    """Atomic governance transaction for board write + state update + event emission."""

    def __init__(self, board: EntryStore, *, step_provider: Callable[[], int]) -> None:
        self._board = board
        self._step_provider = step_provider
        self._completed_idempotency_keys: set[str] = set()

    async def execute(
        self,
        *,
        actor_id: str,
        board_entry: KnowledgeEntry,
        idempotency_key: str,
        governance_mutation: GovernanceMutation,
        emit_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        event_payload: dict[str, Any] | None = None,
    ) -> bool:
        if idempotency_key in self._completed_idempotency_keys:
            return True

        lock = getattr(self._board, "lock", None)
        if lock is not None:
            async with lock:
                return await self._execute_locked(
                    actor_id=actor_id,
                    board_entry=board_entry,
                    idempotency_key=idempotency_key,
                    governance_mutation=governance_mutation,
                    emit_event=emit_event,
                    event_payload=event_payload,
                )
        return await self._execute_locked(
            actor_id=actor_id,
            board_entry=board_entry,
            idempotency_key=idempotency_key,
            governance_mutation=governance_mutation,
            emit_event=emit_event,
            event_payload=event_payload,
        )

    async def _execute_locked(
        self,
        *,
        actor_id: str,
        board_entry: KnowledgeEntry,
        idempotency_key: str,
        governance_mutation: GovernanceMutation,
        emit_event: Callable[[dict[str, Any]], Awaitable[None]] | None,
        event_payload: dict[str, Any] | None,
    ) -> bool:
        if idempotency_key in self._completed_idempotency_keys or self._entry_exists(idempotency_key):
            self._completed_idempotency_keys.add(idempotency_key)
            return True

        tx_context: object | None = None
        if hasattr(self._board, "begin_transaction"):
            tx_context = getattr(self._board, "begin_transaction")()

        rollback_token: Any = None
        try:
            if not self._board.add_entry(board_entry, actor_id, self._step_provider()):
                raise RuntimeError("knowledge_board_add_entry_failed")

            rollback_token = governance_mutation.apply()

            if emit_event is not None and event_payload is not None:
                payload = dict(event_payload)
                payload["idempotency_key"] = idempotency_key
                await emit_event(payload)

            if tx_context is not None and hasattr(self._board, "commit_transaction"):
                getattr(self._board, "commit_transaction")(tx_context)

            self._completed_idempotency_keys.add(idempotency_key)
            return True
        except Exception:
            if tx_context is not None and hasattr(self._board, "rollback_transaction"):
                getattr(self._board, "rollback_transaction")(tx_context)
            governance_mutation.rollback(rollback_token)
            return False

    def _entry_exists(self, idempotency_key: str) -> bool:
        snapshotter = getattr(self._board, "to_snapshot", None)
        if not callable(snapshotter):
            return False
        snapshot = snapshotter()
        if not isinstance(snapshot, dict):
            return False
        for entry in snapshot.get("entries", []):
            if not isinstance(entry, dict):
                continue
            metadata = entry.get("reference_metadata") or {}
            if isinstance(metadata, dict) and metadata.get("idempotency_key") == idempotency_key:
                return True
        return False


def make_proposal_idempotency_key(*, proposer_id: str, text: str, step: int) -> str:
    return f"proposal:{proposer_id}:{step}:{text.strip().lower()}"


def make_vote_idempotency_key(*, voter_id: str, proposal_entry_id: str, approve: bool, step: int) -> str:
    stance = "approve" if approve else "reject"
    return f"vote:{voter_id}:{proposal_entry_id}:{stance}:{step}"
