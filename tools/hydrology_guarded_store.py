"""Fail-closed hydrology persistence decorator with complete dependency lineage.

The underlying stores remain responsible for immutable/versioned persistence. This layer
owns a higher-level invariant: a newly-current hydrology artifact may only be written from
upstreams that are not known stale/withdrawn, and every successful write is connected to
its authoritative upstream dependency identities.

Reads are intentionally transparent so historical/stale artifacts remain inspectable.
"""
from __future__ import annotations

from typing import Any, Protocol

from tools.dependency_invalidation import DependencyRef
from tools.hydrology_dependency_policy import require_fresh, require_sources_active
from tools.hydrology_store import HydrologyArtifactEnvelope, HydrologyArtifactStore, HydrologyArtifactSummary, decode_artifact

_MAX_SOURCES = 10_000


class HydrologyGovernanceLedger(Protocol):
    def list_stale(self, owner_id: str, *, kind: str | None = None, limit: int = 1000): ...
    def source_status_events(self, owner_id: str, source_id: str): ...
    def register_dependency(self, owner_id: str, *, upstream: DependencyRef, downstream: DependencyRef, relation: str) -> Any: ...


def _dep(kind: str, fingerprint: str) -> DependencyRef:
    return DependencyRef(f"hydrology_{kind}", fingerprint)


def _sources(values: Any) -> tuple[str, ...]:
    unique = tuple(sorted({str(value).strip() for value in values if str(value).strip()}))
    if len(unique) > _MAX_SOURCES:
        raise RuntimeError(f"hydrology artifact exceeds the {_MAX_SOURCES} source dependency limit")
    return unique


def _topology_sources(network: Any) -> tuple[str, ...]:
    return _sources([item.source_id for item in network.nodes.values()] + [item.source_id for item in network.reaches.values()])


def _package_sources(package: Any) -> tuple[str, ...]:
    return _sources([item.source_id for item in package.records] + [item.source_id for item in package.objects])


def _row_sources(artifact: Any) -> tuple[str, ...]:
    return _sources(item.source_id for item in artifact.rows)


class GuardedHydrologyArtifactStore(HydrologyArtifactStore):
    """Hydrology store decorator enforcing freshness and source governance on writes."""

    def __init__(self, inner: HydrologyArtifactStore, ledger: HydrologyGovernanceLedger) -> None:
        if inner is None or ledger is None:
            raise ValueError("inner hydrology store and governance ledger are required")
        self._inner = inner
        self._ledger = ledger

    def get(self, owner_id: str, project_id: str, kind: str, logical_id: str, *, fingerprint: str | None = None) -> HydrologyArtifactEnvelope:
        return self._inner.get(owner_id, project_id, kind, logical_id, fingerprint=fingerprint)

    def list(self, owner_id: str, project_id: str, *, kind: str | None = None, include_history: bool = False, limit: int = 200) -> tuple[HydrologyArtifactSummary, ...]:
        return self._inner.list(owner_id, project_id, kind=kind, include_history=include_history, limit=limit)

    def _preflight(self, envelope: HydrologyArtifactEnvelope, artifact: Any) -> None:
        owner = envelope.owner_id
        kind = envelope.kind
        if kind == "topology":
            require_sources_active(self._ledger, owner, _topology_sources(artifact))
            return
        if kind == "engineering_package":
            require_fresh(self._ledger, owner, kind="topology", fingerprint=artifact.topology_fingerprint)
            require_sources_active(self._ledger, owner, _package_sources(artifact))
            return
        if kind == "retrieval_plan":
            require_fresh(self._ledger, owner, kind="topology", fingerprint=artifact.topology_fingerprint)
            if artifact.package_fingerprint:
                require_fresh(self._ledger, owner, kind="package", fingerprint=artifact.package_fingerprint)
            return
        if kind == "evidence_projection":
            require_fresh(self._ledger, owner, kind="topology", fingerprint=artifact.topology_fingerprint)
            require_fresh(self._ledger, owner, kind="package", fingerprint=artifact.package_fingerprint)
            require_fresh(self._ledger, owner, kind="plan", fingerprint=artifact.plan_fingerprint)
            require_sources_active(self._ledger, owner, _row_sources(artifact))
            return
        if kind == "evidence_report":
            require_fresh(self._ledger, owner, kind="topology", fingerprint=artifact.topology_fingerprint)
            require_fresh(self._ledger, owner, kind="package", fingerprint=artifact.package_fingerprint)
            require_fresh(self._ledger, owner, kind="plan", fingerprint=artifact.plan_fingerprint)
            require_fresh(self._ledger, owner, kind="projection", fingerprint=artifact.projection_fingerprint)
            require_sources_active(self._ledger, owner, _row_sources(artifact))
            return
        raise ValueError("unsupported hydrology artifact kind")

    def _register(self, owner: str, upstream: DependencyRef, downstream: DependencyRef, relation: str) -> None:
        self._ledger.register_dependency(owner, upstream=upstream, downstream=downstream, relation=relation)

    def _register_lineage(self, envelope: HydrologyArtifactEnvelope, artifact: Any) -> None:
        owner = envelope.owner_id
        downstream = _dep(
            {
                "topology": "topology",
                "engineering_package": "package",
                "retrieval_plan": "plan",
                "evidence_projection": "projection",
                "evidence_report": "report",
            }[envelope.kind],
            envelope.fingerprint,
        )
        if envelope.kind == "topology":
            for source_id in _topology_sources(artifact):
                self._register(owner, DependencyRef("source", source_id), downstream, "source_hydrology_topology")
            return
        if envelope.kind == "engineering_package":
            self._register(owner, _dep("topology", artifact.topology_fingerprint), downstream, "hydrology_topology_package")
            for source_id in _package_sources(artifact):
                self._register(owner, DependencyRef("source", source_id), downstream, "source_hydrology_package")
            return
        if envelope.kind == "retrieval_plan":
            self._register(owner, _dep("topology", artifact.topology_fingerprint), downstream, "hydrology_topology_plan")
            if artifact.package_fingerprint:
                self._register(owner, _dep("package", artifact.package_fingerprint), downstream, "hydrology_package_plan")
            return
        if envelope.kind == "evidence_projection":
            self._register(owner, _dep("topology", artifact.topology_fingerprint), downstream, "hydrology_topology_projection")
            self._register(owner, _dep("package", artifact.package_fingerprint), downstream, "hydrology_package_projection")
            self._register(owner, _dep("plan", artifact.plan_fingerprint), downstream, "hydrology_plan_projection")
            for source_id in _row_sources(artifact):
                self._register(owner, DependencyRef("source", source_id), downstream, "source_hydrology_projection")
            return
        if envelope.kind == "evidence_report":
            self._register(owner, _dep("topology", artifact.topology_fingerprint), downstream, "hydrology_topology_report")
            self._register(owner, _dep("package", artifact.package_fingerprint), downstream, "hydrology_package_report")
            self._register(owner, _dep("plan", artifact.plan_fingerprint), downstream, "hydrology_plan_report")
            self._register(owner, _dep("projection", artifact.projection_fingerprint), downstream, "hydrology_projection_report")
            self._register(owner, DependencyRef("project", envelope.project_id), downstream, "hydrology_report_project_scope")
            for source_id in _row_sources(artifact):
                self._register(owner, DependencyRef("source", source_id), downstream, "source_hydrology_report")

    def put(self, envelope: HydrologyArtifactEnvelope, *, expected_current_fingerprint: str | None = None) -> HydrologyArtifactSummary:
        if not isinstance(envelope, HydrologyArtifactEnvelope):
            raise TypeError("envelope must be HydrologyArtifactEnvelope")
        artifact = decode_artifact(envelope.kind, envelope.payload)
        self._preflight(envelope, artifact)
        stored = self._inner.put(envelope, expected_current_fingerprint=expected_current_fingerprint)
        self._register_lineage(envelope, artifact)
        return stored


__all__ = ["GuardedHydrologyArtifactStore", "HydrologyGovernanceLedger"]
