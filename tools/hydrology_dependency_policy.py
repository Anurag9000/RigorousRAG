"""Hydrology dependency identities and fail-closed freshness preflight helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from tools.dependency_invalidation import DependencyRef
from tools.retraction_propagation import latest_source_status

_KIND_MAP = {
    "topology": "hydrology_topology",
    "package": "hydrology_package",
    "plan": "hydrology_plan",
    "projection": "hydrology_projection",
    "report": "hydrology_report",
}
_MAX_SCAN = 10_000
_MAX_SOURCE_PREFLIGHT = 10_000


class StaleLedger(Protocol):
    def list_stale(self, owner_id: str, *, kind: str | None = None, limit: int = 1000): ...
    def source_status_events(self, owner_id: str, source_id: str): ...


@dataclass(frozen=True)
class HydrologyStaleReason:
    event_sha256: str
    reason: str
    stale_at: float
    replacement_id: str = ""

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "event_sha256": self.event_sha256,
            "reason": self.reason,
            "stale_at": self.stale_at,
            "replacement_id": self.replacement_id,
        }


class StaleHydrologyArtifactError(RuntimeError):
    def __init__(self, kind: str, fingerprint: str, reasons: tuple[HydrologyStaleReason, ...]) -> None:
        self.kind = kind
        self.fingerprint = fingerprint
        self.reasons = reasons
        detail = "; ".join(item.reason for item in reasons[:5]) or "stale dependency state"
        super().__init__(f"hydrology {kind} {fingerprint} is stale: {detail}")


class UnsafeHydrologySourceError(RuntimeError):
    def __init__(self, source_id: str, status: str, replacement_source_id: str = "") -> None:
        self.source_id = source_id
        self.status = status
        self.replacement_source_id = replacement_source_id
        suffix = f"; replacement {replacement_source_id}" if replacement_source_id else ""
        super().__init__(f"hydrology source {source_id} is not active ({status}){suffix}")


def dependency_ref(kind: str, fingerprint: str) -> DependencyRef:
    try:
        mapped = _KIND_MAP[str(kind).strip().lower()]
    except KeyError as exc:
        raise ValueError("unsupported hydrology dependency kind") from exc
    return DependencyRef(mapped, fingerprint)


def stale_reasons(
    store: StaleLedger | None,
    owner_id: str,
    *,
    kind: str,
    fingerprint: str,
) -> tuple[HydrologyStaleReason, ...]:
    if store is None:
        return ()
    ref = dependency_ref(kind, fingerprint)
    rows = tuple(store.list_stale(owner_id, kind=ref.kind, limit=_MAX_SCAN))
    output: list[HydrologyStaleReason] = []
    for row in rows:
        artifact = getattr(row, "artifact", None)
        if artifact != ref:
            continue
        output.append(
            HydrologyStaleReason(
                event_sha256=str(getattr(row, "triggering_event_sha256", "")),
                reason=str(getattr(row, "reason", "")),
                stale_at=float(getattr(row, "stale_at", 0.0)),
                replacement_id=str(getattr(row, "replacement_id", "")),
            )
        )
    if not output and len(rows) >= _MAX_SCAN:
        raise RuntimeError(f"hydrology {kind} freshness could not be proven because the stale ledger scan was truncated")
    return tuple(output)


def require_fresh(
    store: StaleLedger | None,
    owner_id: str,
    *,
    kind: str,
    fingerprint: str,
) -> None:
    reasons = stale_reasons(store, owner_id, kind=kind, fingerprint=fingerprint)
    if reasons:
        raise StaleHydrologyArtifactError(kind, fingerprint, reasons)


def require_sources_active(
    store: StaleLedger | None,
    owner_id: str,
    source_ids: Sequence[str],
) -> None:
    """Reject derivation from a source whose latest recorded state is non-active.

    No status history means the source has not been flagged by the governance ledger. A
    correction/supersession/retraction/withdrawal blocks new hydrology derivations until an
    explicit later ``active`` event records reconciliation.
    """
    if store is None:
        return
    unique = tuple(sorted(set(str(item).strip() for item in source_ids if str(item).strip())))
    if len(unique) > _MAX_SOURCE_PREFLIGHT:
        raise RuntimeError("hydrology source-status preflight exceeds the bounded source limit")
    for source_id in unique:
        events = tuple(store.source_status_events(owner_id, source_id))
        if not events:
            continue
        latest = latest_source_status(events).get(source_id)
        if latest is None or latest.status == "active":
            continue
        raise UnsafeHydrologySourceError(source_id, latest.status, latest.replacement_source_id)


__all__ = [
    "HydrologyStaleReason",
    "StaleHydrologyArtifactError",
    "StaleLedger",
    "UnsafeHydrologySourceError",
    "dependency_ref",
    "require_fresh",
    "require_sources_active",
    "stale_reasons",
]
