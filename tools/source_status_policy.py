"""Effective source-status resolution for citation publication.

Source-status events are append-only. The latest event effective at or before the query
instant controls whether an existing indexed source may be published as current
authoritative evidence. Historical bytes/index rows are never deleted by this policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from tools.retraction_propagation import SourceStatusEvent

_BLOCKED = frozenset({"retracted", "withdrawn", "superseded", "corrected"})


class SourceStatusReader(Protocol):
    def source_status_events(self, owner_id: str, source_id: str) -> Sequence[SourceStatusEvent]: ...


@dataclass(frozen=True)
class SourceDisposition:
    source_id: str
    status: str
    allowed_as_current_evidence: bool
    event_sha256: str = ""
    effective_at: float | None = None
    replacement_source_id: str = ""
    reason: str = ""


def effective_source_status(
    reader: SourceStatusReader,
    owner_id: str,
    source_id: str,
    *,
    as_of: float | None = None,
) -> SourceDisposition:
    if not isinstance(source_id, str) or not source_id.strip() or len(source_id) > 1000:
        raise ValueError("source_id is invalid")
    instant = time.time() if as_of is None else float(as_of)
    if instant < 0:
        raise ValueError("as_of is invalid")
    events = tuple(reader.source_status_events(owner_id, source_id.strip()))
    applicable = [event for event in events if event.effective_at <= instant]
    if not applicable:
        return SourceDisposition(source_id.strip(), "active", True)
    event = max(applicable, key=lambda item: (item.effective_at, item.event_sha256))
    return SourceDisposition(
        source_id=event.source_id,
        status=event.status,
        allowed_as_current_evidence=event.status not in _BLOCKED,
        event_sha256=event.event_sha256,
        effective_at=event.effective_at,
        replacement_source_id=event.replacement_source_id,
        reason=event.reason,
    )


__all__ = ["SourceDisposition", "SourceStatusReader", "effective_source_status"]
