"""Durable, fenced pause/resume/cancel semantics for active workers.

The actual queue/lease backends already belong to RigorousRAG's distributed execution
layer.  This module supplies the missing control-plane contract: monotonic revisions,
fencing tokens, cooperative checkpoints, terminal cancellation, bounded pause polling,
and explicit safe-point semantics.  Importing it starts no worker or thread.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid")
    return result


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class WorkerControlState(str, Enum):
    RUNNING = "running"
    PAUSE_REQUESTED = "pause_requested"
    PAUSED = "paused"
    RESUME_REQUESTED = "resume_requested"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


_TERMINAL = {WorkerControlState.CANCELLED, WorkerControlState.COMPLETED, WorkerControlState.FAILED}


@dataclass(frozen=True)
class WorkerControlRecord:
    operation_id: str
    state: WorkerControlState
    revision: int
    fencing_token: int
    updated_at: datetime
    requested_by: str
    reason_code: str
    checkpoint_digest: str | None = None
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        for name in ("operation_id", "requested_by", "reason_code"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if not isinstance(self.state, WorkerControlState):
            object.__setattr__(self, "state", WorkerControlState(self.state))
        for name in ("revision", "fencing_token"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(self, "updated_at", _utc(self.updated_at, "updated_at"))
        if self.checkpoint_digest is not None:
            digest = _identifier(self.checkpoint_digest, "checkpoint_digest", 64).lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("checkpoint_digest must be SHA-256")
            object.__setattr__(self, "checkpoint_digest", digest)
        metadata = self.metadata or {}
        if len(metadata) > 1_000:
            raise ValueError("metadata is too large")
        object.__setattr__(
            self,
            "metadata",
            {
                _identifier(key, "metadata key", 300): _identifier(value, "metadata value", 10_000)
                for key, value in metadata.items()
            },
        )

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


class WorkerControlStore(Protocol):
    def read(self, operation_id: str) -> WorkerControlRecord | None: ...

    def compare_and_swap(
        self,
        operation_id: str,
        *,
        expected_revision: int,
        expected_fencing_token: int,
        replacement: WorkerControlRecord,
    ) -> bool: ...


@dataclass(frozen=True)
class WorkerSafePoint:
    checkpoint_digest: str
    resumable: bool
    side_effects_committed: bool

    def __post_init__(self) -> None:
        digest = _identifier(self.checkpoint_digest, "checkpoint_digest", 64).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("checkpoint_digest must be SHA-256")
        object.__setattr__(self, "checkpoint_digest", digest)
        if not isinstance(self.resumable, bool) or not isinstance(self.side_effects_committed, bool):
            raise ValueError("safe-point flags must be boolean")


@dataclass(frozen=True)
class WorkerDirective:
    continue_running: bool
    enter_pause: bool
    cancel: bool
    terminal: bool
    state: WorkerControlState


def directive_for(record: WorkerControlRecord | None) -> WorkerDirective:
    if record is None:
        return WorkerDirective(True, False, False, False, WorkerControlState.RUNNING)
    if record.state in _TERMINAL:
        return WorkerDirective(False, False, record.state == WorkerControlState.CANCELLED, True, record.state)
    if record.state in {WorkerControlState.PAUSE_REQUESTED, WorkerControlState.PAUSED}:
        return WorkerDirective(False, True, False, False, record.state)
    if record.state == WorkerControlState.CANCEL_REQUESTED:
        return WorkerDirective(False, False, True, False, record.state)
    return WorkerDirective(True, False, False, False, record.state)


def _allowed_transition(current: WorkerControlState, target: WorkerControlState) -> bool:
    allowed = {
        WorkerControlState.RUNNING: {
            WorkerControlState.PAUSE_REQUESTED,
            WorkerControlState.CANCEL_REQUESTED,
            WorkerControlState.COMPLETED,
            WorkerControlState.FAILED,
        },
        WorkerControlState.PAUSE_REQUESTED: {
            WorkerControlState.PAUSED,
            WorkerControlState.CANCEL_REQUESTED,
            WorkerControlState.FAILED,
        },
        WorkerControlState.PAUSED: {
            WorkerControlState.RESUME_REQUESTED,
            WorkerControlState.CANCEL_REQUESTED,
            WorkerControlState.FAILED,
        },
        WorkerControlState.RESUME_REQUESTED: {
            WorkerControlState.RUNNING,
            WorkerControlState.CANCEL_REQUESTED,
            WorkerControlState.FAILED,
        },
        WorkerControlState.CANCEL_REQUESTED: {
            WorkerControlState.CANCELLED,
            WorkerControlState.FAILED,
        },
        WorkerControlState.CANCELLED: set(),
        WorkerControlState.COMPLETED: set(),
        WorkerControlState.FAILED: set(),
    }
    return target in allowed[current]


def transition_control(
    current: WorkerControlRecord,
    target: WorkerControlState,
    *,
    now: datetime,
    requested_by: str,
    reason_code: str,
    safe_point: WorkerSafePoint | None = None,
) -> WorkerControlRecord:
    """Create the next immutable control record; persistence remains a CAS in the backend."""

    selected_target = WorkerControlState(target)
    if not _allowed_transition(current.state, selected_target):
        raise ValueError(f"invalid worker-control transition {current.state.value} -> {selected_target.value}")
    if selected_target in {WorkerControlState.PAUSED, WorkerControlState.CANCELLED}:
        if safe_point is None:
            raise ValueError("paused/cancelled transition requires an explicit safe point")
        if selected_target == WorkerControlState.PAUSED and not safe_point.resumable:
            raise ValueError("paused worker must have a resumable checkpoint")
        if not safe_point.side_effects_committed:
            raise ValueError("worker may not acknowledge pause/cancel across uncommitted side effects")
    return WorkerControlRecord(
        operation_id=current.operation_id,
        state=selected_target,
        revision=current.revision + 1,
        fencing_token=current.fencing_token,
        updated_at=now,
        requested_by=requested_by,
        reason_code=reason_code,
        checkpoint_digest=safe_point.checkpoint_digest if safe_point is not None else current.checkpoint_digest,
        metadata=current.metadata,
    )


def persist_transition(
    store: WorkerControlStore,
    current: WorkerControlRecord,
    replacement: WorkerControlRecord,
) -> None:
    if replacement.operation_id != current.operation_id:
        raise ValueError("replacement operation_id differs")
    if replacement.fencing_token != current.fencing_token:
        raise ValueError("replacement fencing token differs")
    if replacement.revision != current.revision + 1:
        raise ValueError("replacement revision is not monotonic")
    if not store.compare_and_swap(
        current.operation_id,
        expected_revision=current.revision,
        expected_fencing_token=current.fencing_token,
        replacement=replacement,
    ):
        raise RuntimeError("worker control changed concurrently or fencing token is stale")


__all__ = [
    "WorkerControlRecord",
    "WorkerControlState",
    "WorkerControlStore",
    "WorkerDirective",
    "WorkerSafePoint",
    "canonical_digest",
    "directive_for",
    "persist_transition",
    "transition_control",
]
