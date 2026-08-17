"""Deterministic maintained-target population reconciliation.

The blue/green migration modules already make one target population operation durable,
fenced and crash resumable.  This module provides the fleet-level control plane around
those operations: compare the desired maintained target set with a physical inventory,
identify missing or drifted targets, choose already-valid populations when possible,
plan alias cutovers only to fully verified populations, and surface old populations as
GC *candidates* without deleting them.

The module intentionally owns no database, vector engine, timer or background thread.
Concrete stores implement the small protocols below.  All mutations are revalidated
under a caller-supplied fencing token and are suitable for invocation from
``orchestration.periodic_reconciliation``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Protocol, Sequence

from orchestration.periodic_reconciliation import ReconciliationResult
from tools.security import normalize_owner_id

_ALLOWED_KINDS = frozenset(
    {
        "dense",
        "sparse",
        "lexical",
        "late_interaction",
        "graph",
        "multimodal",
    }
)
_ALLOWED_STATES = frozenset({"building", "ready", "failed", "quarantined"})
_ALLOWED_FINDINGS = frozenset(
    {
        "healthy",
        "missing",
        "population_in_flight",
        "generation_drift",
        "profile_drift",
        "schema_drift",
        "source_drift",
        "count_drift",
        "failed_population",
        "alias_missing",
        "alias_stale",
        "orphan_candidate",
        "protected_orphan",
        "live_orphan",
    }
)
_ALLOWED_ACTIONS = frozenset({"populate", "bind_alias", "record_orphan_candidate"})
_MAX_ITEMS = 100_000


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 10**15:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_fence(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("fencing_token must be a positive integer")
    return value


@dataclass(frozen=True)
class DesiredTarget:
    owner_id: str
    kind: str
    logical_name: str
    generation_id: str
    profile_sha256: str
    schema_sha256: str
    source_sha256: str
    expected_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        kind = _text(self.kind, "kind", 64).lower()
        if kind not in _ALLOWED_KINDS:
            raise ValueError("unsupported target kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "logical_name", _text(self.logical_name, "logical_name", 300))
        object.__setattr__(self, "generation_id", _text(self.generation_id, "generation_id", 300))
        for name in ("profile_sha256", "schema_sha256", "source_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "expected_count", _non_negative_int(self.expected_count, "expected_count"))

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.logical_name)

    @property
    def signature_sha256(self) -> str:
        return _digest({"contract": "rigorousrag-maintained-target-v1", **asdict(self)})

    @property
    def deterministic_population_key(self) -> str:
        return _digest(
            {
                "owner_id": self.owner_id,
                "kind": self.kind,
                "logical_name": self.logical_name,
                "signature_sha256": self.signature_sha256,
            }
        )


@dataclass(frozen=True)
class PhysicalTarget:
    owner_id: str
    physical_id: str
    kind: str
    logical_name: str
    generation_id: str
    profile_sha256: str
    schema_sha256: str
    source_sha256: str
    observed_count: int
    state: str
    created_at: datetime
    population_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "physical_id", _text(self.physical_id, "physical_id", 500))
        kind = _text(self.kind, "kind", 64).lower()
        if kind not in _ALLOWED_KINDS:
            raise ValueError("unsupported target kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "logical_name", _text(self.logical_name, "logical_name", 300))
        object.__setattr__(self, "generation_id", _text(self.generation_id, "generation_id", 300))
        for name in ("profile_sha256", "schema_sha256", "source_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "observed_count", _non_negative_int(self.observed_count, "observed_count"))
        state = _text(self.state, "state", 32).lower()
        if state not in _ALLOWED_STATES:
            raise ValueError("unsupported physical target state")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        if self.population_key is not None:
            object.__setattr__(self, "population_key", _sha(self.population_key, "population_key"))

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.logical_name)

    @property
    def observation_sha256(self) -> str:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return _digest({"contract": "rigorousrag-physical-target-observation-v1", **payload})


@dataclass(frozen=True)
class AliasBinding:
    owner_id: str
    kind: str
    logical_name: str
    physical_id: str | None
    revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        kind = _text(self.kind, "kind", 64).lower()
        if kind not in _ALLOWED_KINDS:
            raise ValueError("unsupported alias kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "logical_name", _text(self.logical_name, "logical_name", 300))
        if self.physical_id is not None:
            object.__setattr__(self, "physical_id", _text(self.physical_id, "physical_id", 500))
        object.__setattr__(self, "revision", _non_negative_int(self.revision, "revision"))

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.logical_name)


@dataclass(frozen=True)
class PopulationSnapshot:
    owner_id: str
    desired: tuple[DesiredTarget, ...]
    physical: tuple[PhysicalTarget, ...]
    aliases: tuple[AliasBinding, ...]
    protected_physical_ids: frozenset[str] = frozenset()
    in_flight_physical_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        object.__setattr__(self, "owner_id", owner)
        if len(self.desired) > _MAX_ITEMS or len(self.physical) > _MAX_ITEMS or len(self.aliases) > _MAX_ITEMS:
            raise ValueError("population snapshot is too large")
        if any(item.owner_id != owner for item in (*self.desired, *self.physical, *self.aliases)):
            raise ValueError("population snapshot crosses owner boundary")
        if len({item.key for item in self.desired}) != len(self.desired):
            raise ValueError("desired targets must be unique by kind/logical_name")
        if len({item.physical_id for item in self.physical}) != len(self.physical):
            raise ValueError("physical target ids must be unique")
        if len({item.key for item in self.aliases}) != len(self.aliases):
            raise ValueError("aliases must be unique by kind/logical_name")
        known = {item.physical_id for item in self.physical}
        for name in ("protected_physical_ids", "in_flight_physical_ids"):
            selected = frozenset(_text(item, f"{name} item", 500) for item in getattr(self, name))
            if not selected.issubset(known):
                raise ValueError(f"{name} references unknown physical targets")
            object.__setattr__(self, name, selected)


@dataclass(frozen=True)
class PopulationFinding:
    finding_id: str
    owner_id: str
    kind: str
    logical_name: str
    status: str
    desired_signature_sha256: str | None = None
    physical_id: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "finding_id", _sha(self.finding_id, "finding_id"))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        kind = _text(self.kind, "kind", 64).lower()
        if kind not in _ALLOWED_KINDS:
            raise ValueError("unsupported finding kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "logical_name", _text(self.logical_name, "logical_name", 300))
        status = _text(self.status, "status", 64).lower()
        if status not in _ALLOWED_FINDINGS:
            raise ValueError("unsupported population finding")
        object.__setattr__(self, "status", status)
        if self.desired_signature_sha256 is not None:
            object.__setattr__(self, "desired_signature_sha256", _sha(self.desired_signature_sha256, "desired_signature_sha256"))
        if self.physical_id is not None:
            object.__setattr__(self, "physical_id", _text(self.physical_id, "physical_id", 500))
        object.__setattr__(self, "detail", _text(self.detail, "detail", 2000) if self.detail else "")


@dataclass(frozen=True)
class PopulationAction:
    action_id: str
    action: str
    desired: DesiredTarget | None
    physical_id: str | None
    expected_alias_physical_id: str | None = None
    expected_alias_revision: int | None = None
    expected_observation_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _sha(self.action_id, "action_id"))
        action = _text(self.action, "action", 64).lower()
        if action not in _ALLOWED_ACTIONS:
            raise ValueError("unsupported population action")
        object.__setattr__(self, "action", action)
        if self.desired is not None and not isinstance(self.desired, DesiredTarget):
            raise ValueError("desired must be DesiredTarget")
        if self.physical_id is not None:
            object.__setattr__(self, "physical_id", _text(self.physical_id, "physical_id", 500))
        if self.expected_alias_physical_id is not None:
            object.__setattr__(self, "expected_alias_physical_id", _text(self.expected_alias_physical_id, "expected_alias_physical_id", 500))
        if self.expected_alias_revision is not None:
            object.__setattr__(self, "expected_alias_revision", _non_negative_int(self.expected_alias_revision, "expected_alias_revision"))
        if self.expected_observation_sha256 is not None:
            object.__setattr__(self, "expected_observation_sha256", _sha(self.expected_observation_sha256, "expected_observation_sha256"))
        if action in {"populate", "bind_alias"} and self.desired is None:
            raise ValueError(f"{action} requires desired target")
        if action == "bind_alias" and (self.physical_id is None or self.expected_alias_revision is None):
            raise ValueError("bind_alias requires physical target and expected alias revision")
        if action == "record_orphan_candidate" and (self.physical_id is None or self.expected_observation_sha256 is None):
            raise ValueError("orphan candidate requires physical observation identity")


@dataclass(frozen=True)
class PopulationPlan:
    owner_id: str
    findings: tuple[PopulationFinding, ...]
    actions: tuple[PopulationAction, ...]
    observed_at: datetime
    plan_sha256: str

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        object.__setattr__(self, "owner_id", owner)
        if any(item.owner_id != owner for item in self.findings):
            raise ValueError("population plan findings cross owner boundary")
        object.__setattr__(self, "observed_at", _utc(self.observed_at, "observed_at"))
        object.__setattr__(self, "plan_sha256", _sha(self.plan_sha256, "plan_sha256"))


@dataclass(frozen=True)
class PopulationReceipt:
    action_id: str
    status: str
    physical_id: str | None
    evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _sha(self.action_id, "action_id"))
        object.__setattr__(self, "status", _text(self.status, "status", 64).lower())
        if self.physical_id is not None:
            object.__setattr__(self, "physical_id", _text(self.physical_id, "physical_id", 500))
        object.__setattr__(self, "evidence_sha256", _sha(self.evidence_sha256, "evidence_sha256"))


def _matches(desired: DesiredTarget, physical: PhysicalTarget) -> bool:
    return (
        physical.owner_id == desired.owner_id
        and physical.key == desired.key
        and physical.state == "ready"
        and physical.generation_id == desired.generation_id
        and physical.profile_sha256 == desired.profile_sha256
        and physical.schema_sha256 == desired.schema_sha256
        and physical.source_sha256 == desired.source_sha256
        and physical.observed_count == desired.expected_count
    )


def _drift_status(desired: DesiredTarget, candidates: Sequence[PhysicalTarget]) -> tuple[str, str]:
    if not candidates:
        return "missing", "no physical population exists for this maintained target"
    if any(item.state == "failed" for item in candidates):
        return "failed_population", "at least one physical population is failed and no exact ready target exists"
    if any(item.generation_id != desired.generation_id for item in candidates):
        return "generation_drift", "physical target generation differs from desired generation"
    if any(item.profile_sha256 != desired.profile_sha256 for item in candidates):
        return "profile_drift", "physical target retrieval/model profile differs from desired profile"
    if any(item.schema_sha256 != desired.schema_sha256 for item in candidates):
        return "schema_drift", "physical target schema differs from desired schema"
    if any(item.source_sha256 != desired.source_sha256 for item in candidates):
        return "source_drift", "physical target source corpus identity differs from desired source"
    return "count_drift", "physical target count differs from desired complete population"


def _finding(owner: str, kind: str, logical_name: str, status: str, *, desired: DesiredTarget | None = None, physical_id: str | None = None, detail: str = "") -> PopulationFinding:
    payload = {
        "owner_id": owner,
        "kind": kind,
        "logical_name": logical_name,
        "status": status,
        "desired_signature_sha256": desired.signature_sha256 if desired else None,
        "physical_id": physical_id,
        "detail": detail,
    }
    return PopulationFinding(_digest({"contract": "rigorousrag-population-finding-v1", **payload}), **payload)


def _action(action: str, *, desired: DesiredTarget | None, physical_id: str | None, expected_alias_physical_id: str | None = None, expected_alias_revision: int | None = None, expected_observation_sha256: str | None = None) -> PopulationAction:
    payload = {
        "action": action,
        "desired_signature_sha256": desired.signature_sha256 if desired else None,
        "physical_id": physical_id,
        "expected_alias_physical_id": expected_alias_physical_id,
        "expected_alias_revision": expected_alias_revision,
        "expected_observation_sha256": expected_observation_sha256,
    }
    return PopulationAction(
        _digest({"contract": "rigorousrag-population-action-v1", **payload}),
        action,
        desired,
        physical_id,
        expected_alias_physical_id,
        expected_alias_revision,
        expected_observation_sha256,
    )


def build_population_plan(
    snapshot: PopulationSnapshot,
    *,
    observed_at: datetime,
    orphan_grace_seconds: int = 86_400,
) -> PopulationPlan:
    """Build a deterministic non-destructive reconciliation plan."""

    if not isinstance(snapshot, PopulationSnapshot):
        raise ValueError("snapshot must be PopulationSnapshot")
    instant = _utc(observed_at, "observed_at")
    if isinstance(orphan_grace_seconds, bool) or not isinstance(orphan_grace_seconds, int) or orphan_grace_seconds < 0:
        raise ValueError("orphan_grace_seconds must be non-negative")

    physical_by_key: dict[tuple[str, str], list[PhysicalTarget]] = {}
    physical_by_id = {item.physical_id: item for item in snapshot.physical}
    for item in snapshot.physical:
        physical_by_key.setdefault(item.key, []).append(item)
    alias_by_key = {item.key: item for item in snapshot.aliases}
    aliased_ids = {item.physical_id for item in snapshot.aliases if item.physical_id is not None}
    selected_ids: set[str] = set()
    findings: list[PopulationFinding] = []
    actions: list[PopulationAction] = []

    for desired in sorted(snapshot.desired, key=lambda item: item.key):
        candidates = sorted(physical_by_key.get(desired.key, ()), key=lambda item: item.physical_id)
        exact = [item for item in candidates if _matches(desired, item)]
        alias = alias_by_key.get(desired.key, AliasBinding(snapshot.owner_id, desired.kind, desired.logical_name, None, 0))
        if exact:
            exact_ids = {item.physical_id for item in exact}
            chosen = physical_by_id[alias.physical_id] if alias.physical_id in exact_ids else exact[0]
            selected_ids.add(chosen.physical_id)
            if alias.physical_id == chosen.physical_id:
                findings.append(_finding(snapshot.owner_id, desired.kind, desired.logical_name, "healthy", desired=desired, physical_id=chosen.physical_id, detail="exact ready population is the live alias target"))
            else:
                status = "alias_missing" if alias.physical_id is None else "alias_stale"
                findings.append(_finding(snapshot.owner_id, desired.kind, desired.logical_name, status, desired=desired, physical_id=chosen.physical_id, detail="an exact ready population exists but the live alias does not select it"))
                actions.append(_action("bind_alias", desired=desired, physical_id=chosen.physical_id, expected_alias_physical_id=alias.physical_id, expected_alias_revision=alias.revision))
            continue

        in_flight = [
            item
            for item in candidates
            if item.state == "building"
            and (
                item.population_key == desired.deterministic_population_key
                or item.physical_id in snapshot.in_flight_physical_ids
            )
        ]
        if in_flight:
            chosen = in_flight[0]
            selected_ids.add(chosen.physical_id)
            findings.append(_finding(snapshot.owner_id, desired.kind, desired.logical_name, "population_in_flight", desired=desired, physical_id=chosen.physical_id, detail="a matching staged population is already in flight; no duplicate build is scheduled"))
            continue

        status, detail = _drift_status(desired, candidates)
        findings.append(_finding(snapshot.owner_id, desired.kind, desired.logical_name, status, desired=desired, detail=detail))
        actions.append(_action("populate", desired=desired, physical_id=None))

    desired_keys = {item.key for item in snapshot.desired}
    grace = timedelta(seconds=orphan_grace_seconds)
    for physical in sorted(snapshot.physical, key=lambda item: item.physical_id):
        if physical.physical_id in selected_ids:
            continue
        is_old_or_unmaintained = physical.key not in desired_keys or not any(
            desired.key == physical.key and _matches(desired, physical) for desired in snapshot.desired
        )
        if not is_old_or_unmaintained:
            continue
        if physical.physical_id in aliased_ids:
            findings.append(_finding(snapshot.owner_id, physical.kind, physical.logical_name, "live_orphan", physical_id=physical.physical_id, detail="physical population is not selected by desired state but is still referenced by a live alias"))
            continue
        if physical.physical_id in snapshot.protected_physical_ids or physical.physical_id in snapshot.in_flight_physical_ids:
            findings.append(_finding(snapshot.owner_id, physical.kind, physical.logical_name, "protected_orphan", physical_id=physical.physical_id, detail="unselected population is protected or still in flight"))
            continue
        if instant - physical.created_at < grace:
            findings.append(_finding(snapshot.owner_id, physical.kind, physical.logical_name, "protected_orphan", physical_id=physical.physical_id, detail="unselected population remains inside the orphan grace window"))
            continue
        findings.append(_finding(snapshot.owner_id, physical.kind, physical.logical_name, "orphan_candidate", physical_id=physical.physical_id, detail="unselected population is old, unaliased, unprotected and eligible only for downstream GC review"))
        actions.append(_action("record_orphan_candidate", desired=None, physical_id=physical.physical_id, expected_observation_sha256=physical.observation_sha256))

    findings_tuple = tuple(sorted(findings, key=lambda item: (item.kind, item.logical_name, item.status, item.physical_id or "")))
    actions_tuple = tuple(sorted(actions, key=lambda item: (item.action, item.action_id)))
    payload = {
        "contract": "rigorousrag-population-plan-v1",
        "owner_id": snapshot.owner_id,
        "findings": [asdict(item) for item in findings_tuple],
        "actions": [
            {
                **asdict(item),
                "desired": asdict(item.desired) if item.desired is not None else None,
            }
            for item in actions_tuple
        ],
        "observed_at": instant.isoformat(),
    }
    return PopulationPlan(snapshot.owner_id, findings_tuple, actions_tuple, instant, _digest(payload))


class PopulationInventory(Protocol):
    def snapshot(self, owner_id: str) -> PopulationSnapshot: ...


class PopulationMutationBackend(Protocol):
    def assert_fence(self, fencing_token: int) -> None: ...

    def begin_population(self, desired: DesiredTarget, *, population_key: str, fencing_token: int) -> PhysicalTarget: ...

    def inspect_physical(self, owner_id: str, physical_id: str) -> PhysicalTarget | None: ...

    def current_alias(self, owner_id: str, kind: str, logical_name: str) -> AliasBinding: ...

    def compare_and_swap_alias(
        self,
        desired: DesiredTarget,
        *,
        expected_physical_id: str | None,
        expected_revision: int,
        new_physical_id: str,
        fencing_token: int,
    ) -> AliasBinding: ...

    def aliases_for_physical(self, owner_id: str, physical_id: str) -> Sequence[AliasBinding]: ...

    def is_protected(self, owner_id: str, physical_id: str) -> bool: ...

    def record_orphan_candidate(self, physical: PhysicalTarget, *, fencing_token: int, plan_sha256: str) -> str: ...


def execute_population_plan(
    plan: PopulationPlan,
    *,
    backend: PopulationMutationBackend,
    fencing_token: int,
    max_actions: int = 100,
) -> tuple[PopulationReceipt, ...]:
    """Execute a bounded plan with pre-mutation fencing and identity revalidation.

    Population submission never changes an alias.  A later reconciliation pass may bind
    an alias only after the new physical target is independently observed as an exact
    ready match.  Orphan handling records candidates only; deletion belongs to the
    repository's retention/legal-hold governed lifecycle.
    """

    fence = _positive_fence(fencing_token)
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or not 1 <= max_actions <= _MAX_ITEMS:
        raise ValueError("max_actions is invalid")
    backend.assert_fence(fence)
    receipts: list[PopulationReceipt] = []
    for action in plan.actions[:max_actions]:
        backend.assert_fence(fence)
        if action.action == "populate":
            assert action.desired is not None
            created = backend.begin_population(action.desired, population_key=action.desired.deterministic_population_key, fencing_token=fence)
            if created.owner_id != plan.owner_id or created.key != action.desired.key:
                raise RuntimeError("population backend returned a cross-target physical population")
            if created.population_key != action.desired.deterministic_population_key:
                raise RuntimeError("population backend did not preserve idempotency key")
            evidence = _digest({"action_id": action.action_id, "physical": created.observation_sha256, "state": created.state})
            receipts.append(PopulationReceipt(action.action_id, "population_submitted", created.physical_id, evidence))
            continue

        if action.action == "bind_alias":
            assert action.desired is not None and action.physical_id is not None and action.expected_alias_revision is not None
            physical = backend.inspect_physical(plan.owner_id, action.physical_id)
            if physical is None or not _matches(action.desired, physical):
                raise RuntimeError("alias target is no longer an exact ready population")
            current = backend.current_alias(plan.owner_id, action.desired.kind, action.desired.logical_name)
            if current.revision != action.expected_alias_revision or current.physical_id != action.expected_alias_physical_id:
                raise RuntimeError("alias changed since population plan was observed")
            backend.assert_fence(fence)
            updated = backend.compare_and_swap_alias(
                action.desired,
                expected_physical_id=action.expected_alias_physical_id,
                expected_revision=action.expected_alias_revision,
                new_physical_id=action.physical_id,
                fencing_token=fence,
            )
            if updated.owner_id != plan.owner_id or updated.key != action.desired.key or updated.physical_id != action.physical_id or updated.revision <= current.revision:
                raise RuntimeError("alias backend returned an invalid cutover result")
            evidence = _digest({"action_id": action.action_id, "physical": physical.observation_sha256, "alias_revision": updated.revision})
            receipts.append(PopulationReceipt(action.action_id, "alias_bound", action.physical_id, evidence))
            continue

        assert action.action == "record_orphan_candidate" and action.physical_id is not None and action.expected_observation_sha256 is not None
        physical = backend.inspect_physical(plan.owner_id, action.physical_id)
        if physical is None:
            evidence = _digest({"action_id": action.action_id, "physical_id": action.physical_id, "state": "already_absent"})
            receipts.append(PopulationReceipt(action.action_id, "already_absent", action.physical_id, evidence))
            continue
        if physical.observation_sha256 != action.expected_observation_sha256:
            raise RuntimeError("orphan candidate changed since population plan was observed")
        if backend.aliases_for_physical(plan.owner_id, action.physical_id):
            raise RuntimeError("orphan candidate became live before candidate recording")
        if backend.is_protected(plan.owner_id, action.physical_id):
            raise RuntimeError("orphan candidate became protected before candidate recording")
        backend.assert_fence(fence)
        candidate_digest = _sha(backend.record_orphan_candidate(physical, fencing_token=fence, plan_sha256=plan.plan_sha256), "orphan candidate receipt")
        receipts.append(PopulationReceipt(action.action_id, "orphan_candidate_recorded", action.physical_id, candidate_digest))
    return tuple(receipts)


class PopulationReconciliationJob:
    """Owner-scoped periodic job adapter for ``run_due_reconciliations``."""

    def __init__(
        self,
        *,
        owner_id: str,
        inventory: PopulationInventory,
        backend: PopulationMutationBackend,
        clock: callable,
        orphan_grace_seconds: int = 86_400,
        max_actions: int = 100,
    ) -> None:
        self.owner_id = normalize_owner_id(owner_id)
        self.inventory = inventory
        self.backend = backend
        if not callable(clock):
            raise ValueError("clock must be callable")
        self.clock = clock
        if isinstance(orphan_grace_seconds, bool) or not isinstance(orphan_grace_seconds, int) or orphan_grace_seconds < 0:
            raise ValueError("orphan_grace_seconds must be non-negative")
        self.orphan_grace_seconds = orphan_grace_seconds
        if isinstance(max_actions, bool) or not isinstance(max_actions, int) or not 1 <= max_actions <= _MAX_ITEMS:
            raise ValueError("max_actions is invalid")
        self.max_actions = max_actions

    def __call__(self, *, fencing_token: int, continuation_token: str | None) -> ReconciliationResult:
        del continuation_token
        snapshot = self.inventory.snapshot(self.owner_id)
        if snapshot.owner_id != self.owner_id:
            raise RuntimeError("inventory returned a cross-owner snapshot")
        plan = build_population_plan(snapshot, observed_at=self.clock(), orphan_grace_seconds=self.orphan_grace_seconds)
        receipts = execute_population_plan(plan, backend=self.backend, fencing_token=fencing_token, max_actions=self.max_actions) if plan.actions else ()
        actionable = min(len(plan.actions), self.max_actions)
        unchanged = sum(1 for item in plan.findings if item.status in {"healthy", "population_in_flight", "protected_orphan", "live_orphan"})
        continuation = plan.plan_sha256 if len(plan.actions) > actionable else None
        return ReconciliationResult(
            examined=len(plan.findings),
            repaired=len(receipts),
            unchanged=unchanged,
            failed_items=0,
            continuation_token=continuation,
        )


__all__ = [
    "AliasBinding",
    "DesiredTarget",
    "PhysicalTarget",
    "PopulationAction",
    "PopulationFinding",
    "PopulationInventory",
    "PopulationMutationBackend",
    "PopulationPlan",
    "PopulationReceipt",
    "PopulationReconciliationJob",
    "PopulationSnapshot",
    "build_population_plan",
    "execute_population_plan",
]
