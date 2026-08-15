"""Owner-scoped artifact lineage, audit events, and dependency-aware retention.

The control plane links source bytes, parsed representations, chunks/regions, index
artifacts, retrieval outputs, model decisions, claims and citations without retaining
raw evidence text.  It is intentionally storage-neutral so SQLite/Postgres/object-store
implementations can persist the same immutable records later.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

from tools.security import normalize_owner_id

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_ALLOWED_RELATIONS = frozenset(
    {
        "derived_from",
        "parsed_from",
        "contains",
        "embedded_from",
        "indexed_from",
        "retrieved_from",
        "reranked_from",
        "supports",
        "contradicts",
        "cites",
        "generated_from",
        "calibrated_from",
        "trained_from",
        "promoted_from",
        "replaces",
        "summarizes",
        "references",
        "normalized_from",
    }
)
_MAX_ARTIFACTS = 200_000
_MAX_EDGES = 1_000_000
_MAX_METADATA_ITEMS = 64
_MAX_EVENT_FIELDS = 64


def _identifier(value: Any, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or not _IDENTIFIER_RE.fullmatch(cleaned):
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    digest = value.lower().strip()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return digest


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _safe_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        cleaned = " ".join(value.replace("\x00", " ").split())
        return cleaned[:500]
    return str(type(value).__name__)[:100]


def _safe_metadata(value: Mapping[str, Any]) -> dict[str, str | int | float | bool | None]:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping")
    if len(value) > _MAX_METADATA_ITEMS:
        raise ValueError("metadata exceeds the item limit")
    output: dict[str, str | int | float | bool | None] = {}
    for key, item in value.items():
        name = _identifier(str(key), "metadata key", 128)
        output[name] = _safe_scalar(item)
    return output


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite timestamp")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite timestamp") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be a finite non-negative timestamp")
    return parsed


@dataclass(frozen=True)
class ArtifactRef:
    """Content-addressed, owner-scoped artifact identity."""

    owner_id: str
    kind: str
    content_sha256: str
    generation: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "kind", _identifier(self.kind, "kind"))
        object.__setattr__(self, "content_sha256", _sha256(self.content_sha256, "content_sha256"))
        generation = self.generation.strip() if isinstance(self.generation, str) else ""
        if generation:
            generation = _identifier(generation, "generation")
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "metadata", _safe_metadata(self.metadata))
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))

    @property
    def artifact_id(self) -> str:
        payload = {
            "owner_id": self.owner_id,
            "kind": self.kind,
            "content_sha256": self.content_sha256,
            "generation": self.generation,
        }
        return hashlib.sha256(_canonical_json(payload)).hexdigest()

    @property
    def metadata_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(dict(self.metadata))).hexdigest()


@dataclass(frozen=True)
class LineageEdge:
    owner_id: str
    parent_artifact_id: str
    child_artifact_id: str
    relation: str
    operation_id: str
    created_at: float = field(default_factory=time.time)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "parent_artifact_id", _sha256(self.parent_artifact_id, "parent_artifact_id"))
        object.__setattr__(self, "child_artifact_id", _sha256(self.child_artifact_id, "child_artifact_id"))
        if self.parent_artifact_id == self.child_artifact_id:
            raise ValueError("lineage edges may not self-reference")
        relation = _identifier(self.relation, "relation").lower()
        if relation not in _ALLOWED_RELATIONS:
            raise ValueError("unsupported lineage relation")
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "operation_id", _identifier(self.operation_id, "operation_id"))
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        object.__setattr__(self, "attributes", _safe_metadata(self.attributes))

    @property
    def edge_id(self) -> str:
        payload = {
            "owner_id": self.owner_id,
            "parent": self.parent_artifact_id,
            "child": self.child_artifact_id,
            "relation": self.relation,
            "operation_id": self.operation_id,
        }
        return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class AuditEvent:
    """Privacy-safe event record containing identifiers/digests, not raw evidence."""

    owner_id: str
    event_type: str
    subject_id: str
    correlation_id: str
    fields: Mapping[str, Any] = field(default_factory=dict)
    occurred_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "event_type", _identifier(self.event_type, "event_type"))
        object.__setattr__(self, "subject_id", _identifier(self.subject_id, "subject_id"))
        object.__setattr__(self, "correlation_id", _identifier(self.correlation_id, "correlation_id"))
        if not isinstance(self.fields, Mapping) or len(self.fields) > _MAX_EVENT_FIELDS:
            raise ValueError("fields must be a bounded mapping")
        object.__setattr__(self, "fields", _safe_metadata(self.fields))
        object.__setattr__(self, "occurred_at", _timestamp(self.occurred_at, "occurred_at"))

    @property
    def event_id(self) -> str:
        payload = {
            "owner_id": self.owner_id,
            "event_type": self.event_type,
            "subject_id": self.subject_id,
            "correlation_id": self.correlation_id,
            "fields": dict(self.fields),
            "occurred_at": self.occurred_at,
        }
        return hashlib.sha256(_canonical_json(payload)).hexdigest()


class AuditSink(Protocol):
    def append(self, event: AuditEvent) -> None: ...


class InMemoryAuditSink:
    def __init__(self, *, max_events: int = 100_000) -> None:
        if isinstance(max_events, bool) or not isinstance(max_events, int) or not 1 <= max_events <= 1_000_000:
            raise ValueError("max_events is invalid")
        self.max_events = max_events
        self._events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        if not isinstance(event, AuditEvent):
            raise TypeError("event must be AuditEvent")
        if len(self._events) >= self.max_events:
            raise RuntimeError("audit sink capacity reached")
        self._events.append(event)

    def for_owner(self, owner_id: str) -> tuple[AuditEvent, ...]:
        owner = normalize_owner_id(owner_id)
        return tuple(event for event in self._events if event.owner_id == owner)


class ArtifactLineageGraph:
    """Bounded immutable-record lineage graph with tenant isolation."""

    def __init__(self, *, audit_sink: AuditSink | None = None) -> None:
        self._artifacts: dict[str, ArtifactRef] = {}
        self._edges: dict[str, LineageEdge] = {}
        self._children: defaultdict[str, set[str]] = defaultdict(set)
        self._parents: defaultdict[str, set[str]] = defaultdict(set)
        self._audit_sink = audit_sink

    def add_artifact(self, artifact: ArtifactRef, *, correlation_id: str = "lineage") -> str:
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("artifact must be ArtifactRef")
        existing = self._artifacts.get(artifact.artifact_id)
        if existing is not None and existing != artifact:
            raise ValueError("artifact identity collision")
        if existing is None and len(self._artifacts) >= _MAX_ARTIFACTS:
            raise RuntimeError("artifact lineage capacity reached")
        self._artifacts[artifact.artifact_id] = artifact
        self._emit(
            AuditEvent(
                artifact.owner_id,
                "artifact_registered",
                artifact.artifact_id,
                correlation_id,
                {"kind": artifact.kind, "metadata_sha256": artifact.metadata_sha256},
            )
        )
        return artifact.artifact_id

    def link(
        self,
        *,
        owner_id: str,
        parent_artifact_id: str,
        child_artifact_id: str,
        relation: str,
        operation_id: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> str:
        owner = normalize_owner_id(owner_id)
        parent_id = _sha256(parent_artifact_id, "parent_artifact_id")
        child_id = _sha256(child_artifact_id, "child_artifact_id")
        parent = self._artifacts.get(parent_id)
        child = self._artifacts.get(child_id)
        if parent is None or child is None:
            raise KeyError("both lineage artifacts must be registered before linking")
        if parent.owner_id != owner or child.owner_id != owner:
            raise PermissionError("cross-owner lineage is forbidden")
        if self._reachable(child_id, parent_id):
            raise ValueError("lineage derivation must remain acyclic")
        edge = LineageEdge(
            owner_id=owner,
            parent_artifact_id=parent_id,
            child_artifact_id=child_id,
            relation=relation,
            operation_id=operation_id,
            attributes=attributes or {},
        )
        existing = self._edges.get(edge.edge_id)
        if existing is not None and existing != edge:
            raise ValueError("lineage edge identity collision")
        if existing is None and len(self._edges) >= _MAX_EDGES:
            raise RuntimeError("lineage edge capacity reached")
        self._edges[edge.edge_id] = edge
        self._children[parent_id].add(child_id)
        self._parents[child_id].add(parent_id)
        self._emit(
            AuditEvent(
                owner,
                "lineage_linked",
                edge.edge_id,
                operation_id,
                {"relation": edge.relation, "parent": parent_id, "child": child_id},
            )
        )
        return edge.edge_id

    def get(self, owner_id: str, artifact_id: str) -> ArtifactRef:
        owner = normalize_owner_id(owner_id)
        identifier = _sha256(artifact_id, "artifact_id")
        artifact = self._artifacts[identifier]
        if artifact.owner_id != owner:
            raise PermissionError("artifact belongs to another owner")
        return artifact

    def ancestors(self, owner_id: str, artifact_id: str, *, max_depth: int = 64) -> tuple[ArtifactRef, ...]:
        return self._walk(owner_id, artifact_id, self._parents, max_depth=max_depth)

    def descendants(self, owner_id: str, artifact_id: str, *, max_depth: int = 64) -> tuple[ArtifactRef, ...]:
        return self._walk(owner_id, artifact_id, self._children, max_depth=max_depth)

    def _walk(
        self,
        owner_id: str,
        artifact_id: str,
        adjacency: Mapping[str, set[str]],
        *,
        max_depth: int,
    ) -> tuple[ArtifactRef, ...]:
        owner = normalize_owner_id(owner_id)
        start = self.get(owner, artifact_id)
        del start
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or not 1 <= max_depth <= 256:
            raise ValueError("max_depth must be between 1 and 256")
        queue: deque[tuple[str, int]] = deque([(artifact_id, 0)])
        seen = {artifact_id}
        output: list[ArtifactRef] = []
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for next_id in sorted(adjacency.get(current, ())):
                if next_id in seen:
                    continue
                seen.add(next_id)
                artifact = self.get(owner, next_id)
                output.append(artifact)
                if len(output) >= _MAX_ARTIFACTS:
                    raise RuntimeError("lineage traversal exceeded the artifact limit")
                queue.append((next_id, depth + 1))
        return tuple(output)

    def _reachable(self, start: str, target: str) -> bool:
        queue = deque([start])
        seen = {start}
        while queue:
            current = queue.popleft()
            if current == target:
                return True
            for child in self._children.get(current, ()):
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
        return False

    def edges_for_owner(self, owner_id: str) -> tuple[LineageEdge, ...]:
        owner = normalize_owner_id(owner_id)
        return tuple(sorted((edge for edge in self._edges.values() if edge.owner_id == owner), key=lambda item: item.edge_id))

    def _emit(self, event: AuditEvent) -> None:
        if self._audit_sink is not None:
            self._audit_sink.append(event)


@dataclass(frozen=True)
class RetentionRule:
    kind: str
    minimum_age_seconds: float
    keep_latest_generations: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _identifier(self.kind, "kind"))
        age = _timestamp(self.minimum_age_seconds, "minimum_age_seconds")
        object.__setattr__(self, "minimum_age_seconds", age)
        if isinstance(self.keep_latest_generations, bool) or not isinstance(self.keep_latest_generations, int) or not 0 <= self.keep_latest_generations <= 10_000:
            raise ValueError("keep_latest_generations is invalid")


@dataclass(frozen=True)
class RetentionPlan:
    owner_id: str
    created_at: float
    protected_artifact_ids: tuple[str, ...]
    delete_artifact_ids: tuple[str, ...]
    reason_by_artifact: Mapping[str, str]
    plan_sha256: str


class RetentionPlanner:
    """Plans deletion only when an artifact is old and not required by protected lineage."""

    def __init__(self, graph: ArtifactLineageGraph) -> None:
        if not isinstance(graph, ArtifactLineageGraph):
            raise TypeError("graph must be ArtifactLineageGraph")
        self.graph = graph

    def plan(
        self,
        *,
        owner_id: str,
        artifacts: Iterable[ArtifactRef],
        rules: Sequence[RetentionRule],
        protected_roots: Sequence[str] = (),
        now: float | None = None,
    ) -> RetentionPlan:
        owner = normalize_owner_id(owner_id)
        current = time.time() if now is None else _timestamp(now, "now")
        rule_map = {rule.kind: rule for rule in rules}
        rows = [item for item in artifacts if item.owner_id == owner]
        if len(rows) > _MAX_ARTIFACTS:
            raise ValueError("artifacts exceed the retention planning limit")

        protected: set[str] = set()
        for root in protected_roots:
            artifact = self.graph.get(owner, root)
            protected.add(artifact.artifact_id)
            protected.update(item.artifact_id for item in self.graph.ancestors(owner, root, max_depth=256))

        by_kind: defaultdict[str, list[ArtifactRef]] = defaultdict(list)
        for artifact in rows:
            by_kind[artifact.kind].append(artifact)
        keep_latest: set[str] = set()
        for kind, values in by_kind.items():
            rule = rule_map.get(kind)
            if rule is None or rule.keep_latest_generations <= 0:
                continue
            ordered = sorted(values, key=lambda item: (item.created_at, item.generation, item.artifact_id), reverse=True)
            keep_latest.update(item.artifact_id for item in ordered[: rule.keep_latest_generations])

        delete: list[str] = []
        reasons: dict[str, str] = {}
        for artifact in sorted(rows, key=lambda item: item.artifact_id):
            rule = rule_map.get(artifact.kind)
            if rule is None:
                reasons[artifact.artifact_id] = "no_retention_rule"
                continue
            if artifact.artifact_id in protected:
                reasons[artifact.artifact_id] = "lineage_protected"
                continue
            if artifact.artifact_id in keep_latest:
                reasons[artifact.artifact_id] = "latest_generation_protected"
                continue
            age = max(0.0, current - artifact.created_at)
            if age < rule.minimum_age_seconds:
                reasons[artifact.artifact_id] = "minimum_age_not_reached"
                continue
            delete.append(artifact.artifact_id)
            reasons[artifact.artifact_id] = "eligible_for_deletion"

        payload = {
            "owner_id": owner,
            "created_at": current,
            "protected": sorted(protected | keep_latest),
            "delete": delete,
            "reasons": reasons,
        }
        return RetentionPlan(
            owner_id=owner,
            created_at=current,
            protected_artifact_ids=tuple(sorted(protected | keep_latest)),
            delete_artifact_ids=tuple(delete),
            reason_by_artifact=reasons,
            plan_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
        )


__all__ = [
    "ArtifactLineageGraph",
    "ArtifactRef",
    "AuditEvent",
    "AuditSink",
    "InMemoryAuditSink",
    "LineageEdge",
    "RetentionPlan",
    "RetentionPlanner",
    "RetentionRule",
]
