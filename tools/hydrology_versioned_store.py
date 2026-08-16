"""Version-transition decorator for governed hydrology artifact persistence.

The inner store owns freshness checks and dependency registration. This decorator makes a
successful current-generation transition invalidate the old fingerprint and queue only
those downstream hydrology artifacts that have deterministic derivation recipes. Raw
engineering packages are never automatically synthesized from a new topology/source.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence

from tools.dependency_invalidation import DependencyRef
from tools.hydrology_store import HydrologyArtifactEnvelope, HydrologyArtifactStore, HydrologyArtifactSummary

_RECOMPUTABLE = ("hydrology_plan", "hydrology_projection", "hydrology_report", "result", "report", "capsule")
_KIND_TO_DEP = {
    "topology": "hydrology_topology",
    "engineering_package": "hydrology_package",
    "retrieval_plan": "hydrology_plan",
    "evidence_projection": "hydrology_projection",
    "evidence_report": "hydrology_report",
}


class HydrologyInvalidationLedger(Protocol):
    def invalidate(
        self,
        owner_id: str,
        *,
        root: DependencyRef,
        reason: str,
        event_type: str,
        replacement_id: str = "",
        recomputable_kinds: Sequence[str] = (),
    ) -> Any: ...


class VersionedHydrologyArtifactStore(HydrologyArtifactStore):
    def __init__(self, inner: HydrologyArtifactStore, ledger: HydrologyInvalidationLedger) -> None:
        if inner is None or ledger is None:
            raise ValueError("inner hydrology store and invalidation ledger are required")
        self._inner = inner
        self._ledger = ledger

    def get(self, owner_id: str, project_id: str, kind: str, logical_id: str, *, fingerprint: str | None = None) -> HydrologyArtifactEnvelope:
        return self._inner.get(owner_id, project_id, kind, logical_id, fingerprint=fingerprint)

    def list(self, owner_id: str, project_id: str, *, kind: str | None = None, include_history: bool = False, limit: int = 200) -> tuple[HydrologyArtifactSummary, ...]:
        return self._inner.list(owner_id, project_id, kind=kind, include_history=include_history, limit=limit)

    def put(self, envelope: HydrologyArtifactEnvelope, *, expected_current_fingerprint: str | None = None) -> HydrologyArtifactSummary:
        if not isinstance(envelope, HydrologyArtifactEnvelope):
            raise TypeError("envelope must be HydrologyArtifactEnvelope")
        try:
            previous = self._inner.get(envelope.owner_id, envelope.project_id, envelope.kind, envelope.logical_id)
        except KeyError:
            previous = None
        stored = self._inner.put(envelope, expected_current_fingerprint=expected_current_fingerprint)
        if previous is None or previous.fingerprint == stored.fingerprint:
            return stored
        dependency_kind = _KIND_TO_DEP[envelope.kind]
        self._ledger.invalidate(
            envelope.owner_id,
            root=DependencyRef(dependency_kind, previous.fingerprint),
            reason=f"hydrology {envelope.kind} generation replaced",
            event_type="hydrology_generation_replaced",
            replacement_id=stored.fingerprint,
            recomputable_kinds=_RECOMPUTABLE,
        )
        return stored


__all__ = ["HydrologyInvalidationLedger", "VersionedHydrologyArtifactStore"]
