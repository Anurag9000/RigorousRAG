"""Hydrology dependency identities and fail-closed freshness preflight helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from tools.dependency_invalidation import DependencyRef

_KIND_MAP = {
    "topology": "hydrology_topology",
    "package": "hydrology_package",
    "plan": "hydrology_plan",
    "projection": "hydrology_projection",
    "report": "hydrology_report",
}
_MAX_SCAN = 10_000


class StaleLedger(Protocol):
    def list_stale(self, owner_id: str, *, kind: str | None = None, limit: int = 1000): ...


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
        # Absence cannot be proven once the bounded scan is saturated.
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


__all__ = [
    "HydrologyStaleReason",
    "StaleHydrologyArtifactError",
    "StaleLedger",
    "dependency_ref",
    "require_fresh",
    "stale_reasons",
]
