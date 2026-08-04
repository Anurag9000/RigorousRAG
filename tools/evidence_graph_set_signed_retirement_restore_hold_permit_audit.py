"""Read-only diagnostics for restore hold-placement permits."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_mutation import (
    _marker_row,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permits import (
    _ensure_table,
    _permit_digest,
)
from tools.security import normalize_owner_id

_CLASSIFICATIONS = frozenset(
    {
        "active_permit_with_active_hold",
        "active_permit_with_released_hold",
        "active_permit_without_hold_record",
        "released_permit_history",
    }
)


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RestoreHoldPermitAuditItem:
    hold_id: str
    restore_id: str
    permit_state: str
    hold_status: str | None
    marker_state: str | None
    created_at: float
    updated_at: float
    released_at: float | None
    permit_digest: str
    classification: str
    exact_hold_replay_recommended: bool


@dataclass(frozen=True)
class RestoreHoldPermitAuditReport:
    owner_id: str
    generated_at: float
    item_count: int
    classification_counts: dict[str, int]
    items: tuple[RestoreHoldPermitAuditItem, ...]
    report_digest: str
    mutation_performed: bool = False
    permit_released: bool = False
    hold_mutation_performed: bool = False
    restore_mutation_performed: bool = False
    source_text_returned: bool = False
    raw_paths_returned: bool = False


def audit_restore_hold_placement_permits(
    *,
    owner_id: str,
    restore_journal: Any,
    hold_store: Any,
    now: float | None = None,
    limit: int = 10_000,
) -> RestoreHoldPermitAuditReport:
    owner = normalize_owner_id(owner_id)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, 10_000)
    if not hasattr(restore_journal, "_lock") or not callable(
        getattr(restore_journal, "_connect", None)
    ):
        raise ValueError("restore journal lacks the permit read boundary.")
    if not callable(getattr(hold_store, "get", None)):
        raise ValueError("hold store lacks the get boundary.")
    with restore_journal._lock, restore_journal._connect() as connection:
        _ensure_table(connection)
        rows = connection.execute(
            "SELECT * FROM signed_retirement_restore_hold_placement_permits "
            "WHERE owner_id=? ORDER BY created_at DESC, hold_id DESC LIMIT ?",
            (owner, count),
        ).fetchall()
        if len(rows) >= count:
            raise RuntimeError("hold permit audit reached the bounded limit.")
        seen: set[str] = set()
        rendered: list[RestoreHoldPermitAuditItem] = []
        counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
        for row in rows:
            hold_id = _digest(row["hold_id"], "hold_id")
            if hold_id in seen:
                raise RuntimeError("hold permit store returned duplicate IDs.")
            seen.add(hold_id)
            restore_id = _digest(row["restore_id"], "restore_id")
            state = row["state"]
            if state not in {"active", "released"}:
                raise RuntimeError("stored hold permit state is corrupt.")
            created = _timestamp(row["created_at"], "created_at")
            updated = _timestamp(row["updated_at"], "updated_at")
            released = (
                None
                if row["released_at"] is None
                else _timestamp(row["released_at"], "released_at")
            )
            expected = _permit_digest(
                hold_id=hold_id,
                owner_id=owner,
                restore_id=restore_id,
                state=state,
                created_at=created,
                updated_at=updated,
                released_at=released,
            )
            permit_digest = _digest(row["permit_digest"], "permit_digest")
            if permit_digest != expected:
                raise RuntimeError("stored hold permit integrity differs.")
            try:
                hold = hold_store.get(hold_id)
            except KeyError:
                hold_status = None
            else:
                if hold.owner_id != owner or hold.restore_id != restore_id:
                    raise RuntimeError("hold permit escaped hold scope.")
                hold_status = hold.status
            marker = _marker_row(connection, restore_id)
            marker_state = None if marker is None else marker["state"]
            if state == "released":
                classification = "released_permit_history"
                replay = False
            elif hold_status == "active":
                classification = "active_permit_with_active_hold"
                replay = True
            elif hold_status == "released":
                classification = "active_permit_with_released_hold"
                replay = False
            else:
                classification = "active_permit_without_hold_record"
                replay = False
            counts[classification] += 1
            rendered.append(
                RestoreHoldPermitAuditItem(
                    hold_id=hold_id,
                    restore_id=restore_id,
                    permit_state=state,
                    hold_status=hold_status,
                    marker_state=marker_state,
                    created_at=created,
                    updated_at=updated,
                    released_at=released,
                    permit_digest=permit_digest,
                    classification=classification,
                    exact_hold_replay_recommended=replay,
                )
            )
    items = tuple(sorted(rendered, key=lambda item: item.hold_id))
    stable = {
        "scope": "rigorousrag-restore-hold-permit-audit-v1",
        "owner_id": owner,
        "generated_at": timestamp,
        "item_count": len(items),
        "classification_counts": counts,
        "items": [asdict(item) for item in items],
    }
    return RestoreHoldPermitAuditReport(
        owner_id=owner,
        generated_at=timestamp,
        item_count=len(items),
        classification_counts=counts,
        items=items,
        report_digest=_canonical_digest(stable),
    )


__all__ = [
    "RestoreHoldPermitAuditItem",
    "RestoreHoldPermitAuditReport",
    "audit_restore_hold_placement_permits",
]
