"""Deterministic cross-store repair/adoption/reindex planning.

This complements ``operator_repair.py`` (durable-job row recovery). It targets the
vector/sparse/manifest/retained/graph/object lifecycle and strictly separates scan,
planning, exact confirmation and store-specific execution.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from tools.security import normalize_owner_id

_ALLOWED_STORES = frozenset({"vector", "sparse", "manifest", "retained", "graph", "object", "generation", "lifecycle"})
_ALLOWED_ACTIONS = frozenset({"reindex", "rebuild_manifest", "remove_orphan", "adopt_legacy", "restore_registry", "rebuild_graph", "reconcile", "no_op"})


def _text(value: Any, label: str, maximum: int = 500, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class StoreObservation:
    store: str
    present: bool
    identity_sha256: str = ""
    generation: int = 0
    content_sha256: str = ""
    profile_sha256: str = ""
    count: int = 0

    def __post_init__(self) -> None:
        store = _text(self.store, "store", 32).lower()
        if store not in _ALLOWED_STORES:
            raise ValueError("unsupported store")
        object.__setattr__(self, "store", store)
        if not isinstance(self.present, bool):
            raise ValueError("present must be boolean")
        for name in ("identity_sha256", "content_sha256", "profile_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name, allow_empty=True))
        for name in ("generation", "count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 10**15:
                raise ValueError(f"{name} is invalid")


@dataclass(frozen=True)
class RepairFinding:
    finding_id: str
    owner_id: str
    doc_id: str
    category: str
    observations: tuple[StoreObservation, ...]
    severity: str
    recoverable_from_retained_source: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "finding_id", _text(self.finding_id, "finding_id", 256))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _text(self.doc_id, "doc_id", 200))
        object.__setattr__(self, "category", _text(self.category, "category", 128))
        if not self.observations or len(self.observations) > len(_ALLOWED_STORES) or any(not isinstance(item, StoreObservation) for item in self.observations):
            raise ValueError("observations are invalid")
        if len({item.store for item in self.observations}) != len(self.observations):
            raise ValueError("observations contain duplicate stores")
        severity = _text(self.severity, "severity", 16).lower()
        if severity not in {"info", "warning", "error", "critical"}:
            raise ValueError("severity is invalid")
        object.__setattr__(self, "severity", severity)
        if not isinstance(self.recoverable_from_retained_source, bool):
            raise ValueError("recoverable_from_retained_source must be boolean")


@dataclass(frozen=True)
class RepairAction:
    action_id: str
    finding_id: str
    action: str
    target_stores: tuple[str, ...]
    expected_source_sha256: str = ""
    expected_generation: int = 0
    destructive: bool = False
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _text(self.action_id, "action_id", 256))
        object.__setattr__(self, "finding_id", _text(self.finding_id, "finding_id", 256))
        action = _text(self.action, "action", 64).lower()
        if action not in _ALLOWED_ACTIONS:
            raise ValueError("unsupported repair action")
        object.__setattr__(self, "action", action)
        stores = tuple(dict.fromkeys(_text(item, "target store", 32).lower() for item in self.target_stores))
        if not stores or any(item not in _ALLOWED_STORES for item in stores):
            raise ValueError("target_stores are invalid")
        object.__setattr__(self, "target_stores", stores)
        object.__setattr__(self, "expected_source_sha256", _sha(self.expected_source_sha256, "expected_source_sha256", allow_empty=True))
        if isinstance(self.expected_generation, bool) or not isinstance(self.expected_generation, int) or self.expected_generation < 0:
            raise ValueError("expected_generation is invalid")
        if not isinstance(self.destructive, bool):
            raise ValueError("destructive must be boolean")
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale", 2000, allow_empty=True))


@dataclass(frozen=True)
class RepairPlan:
    owner_id: str
    findings: tuple[RepairFinding, ...]
    actions: tuple[RepairAction, ...]
    created_at: float
    plan_sha256: str

    @property
    def confirmation_token(self) -> str:
        return f"CONFIRM-STORE-REPAIR-{self.plan_sha256}"


def build_repair_plan(owner_id: str, findings: Sequence[RepairFinding]) -> RepairPlan:
    owner = normalize_owner_id(owner_id)
    if len(findings) > 100_000 or any(item.owner_id != owner for item in findings):
        raise ValueError("findings are invalid for this owner")
    actions: list[RepairAction] = []
    for finding in sorted(findings, key=lambda item: item.finding_id):
        by_store = {item.store: item for item in finding.observations}
        retained = by_store.get("retained")
        present = {name for name, item in by_store.items() if item.present}
        missing = set(by_store) - present
        source_hash = retained.content_sha256 if retained and retained.present else ""
        generation = max((item.generation for item in finding.observations), default=0)
        reconstructible = missing & {"vector", "sparse", "manifest", "graph"}
        if finding.recoverable_from_retained_source and retained and retained.present and reconstructible:
            action = "rebuild_manifest" if reconstructible == {"manifest"} else "reindex"
            actions.append(RepairAction(f"repair-{finding.finding_id}", finding.finding_id, action, tuple(sorted(reconstructible)), source_hash, generation, False, "reconstruct missing derivatives from authenticated retained source into a new generation"))
            continue
        if finding.category == "verified_aligned_legacy" and {"vector", "sparse"}.issubset(present) and "manifest" in missing:
            vector = by_store["vector"]
            sparse = by_store["sparse"]
            aligned = bool(source_hash and vector.content_sha256 == sparse.content_sha256 == source_hash and vector.profile_sha256 == sparse.profile_sha256 and vector.count == sparse.count)
            if aligned:
                actions.append(RepairAction(f"adopt-{finding.finding_id}", finding.finding_id, "adopt_legacy", ("manifest", "generation"), source_hash, generation, False, "adopt only an exactly aligned legacy vector/sparse pair with retained-source identity"))
            else:
                actions.append(RepairAction(f"noop-{finding.finding_id}", finding.finding_id, "no_op", ("manifest",), source_hash, generation, False, "legacy state is not sufficiently aligned for adoption"))
            continue
        if finding.category == "orphan" and finding.severity in {"error", "critical"}:
            orphans = tuple(sorted(name for name, item in by_store.items() if item.present and name not in {"retained", "generation"}))
            if orphans:
                actions.append(RepairAction(f"remove-{finding.finding_id}", finding.finding_id, "remove_orphan", orphans, source_hash, generation, True, "remove only after exact confirmation and identity revalidation"))
                continue
        actions.append(RepairAction(f"reconcile-{finding.finding_id}", finding.finding_id, "reconcile", tuple(sorted(by_store)), source_hash, generation, False, "requires store-specific reconciliation"))
    created = time.time()
    payload = {"owner_id": owner, "findings": [asdict(item) for item in findings], "actions": [asdict(item) for item in actions], "created_at": created}
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    return RepairPlan(owner, tuple(findings), tuple(actions), created, digest)


@dataclass(frozen=True)
class RepairReceipt:
    action_id: str
    status: str
    before_sha256: str
    after_sha256: str
    operation_id: str
    completed_at: float
    details: Mapping[str, str] = field(default_factory=dict)


class RepairHandler(Protocol):
    def preflight(self, action: RepairAction) -> str: ...
    def execute(self, action: RepairAction) -> tuple[str, Mapping[str, str]]: ...


def execute_repair_plan(plan: RepairPlan, *, confirmation_token: str, handlers: Mapping[str, RepairHandler], operation_id: str) -> tuple[RepairReceipt, ...]:
    if confirmation_token != plan.confirmation_token:
        raise PermissionError("confirmation token does not match the exact repair plan")
    op_id = _text(operation_id, "operation_id", 256)
    receipts: list[RepairReceipt] = []
    for action in plan.actions:
        if action.action == "no_op":
            continue
        handler = handlers.get(action.action)
        if handler is None:
            raise RuntimeError(f"no repair handler registered for {action.action}")
        before = _sha(handler.preflight(action), "preflight digest")
        if action.expected_source_sha256 and action.action in {"reindex", "adopt_legacy", "restore_registry", "rebuild_graph"} and before != action.expected_source_sha256:
            raise RuntimeError("repair preflight identity changed since planning")
        after_raw, details = handler.execute(action)
        after = _sha(after_raw, "after digest")
        receipts.append(RepairReceipt(action.action_id, "completed", before, after, op_id, time.time(), {str(k)[:100]: str(v)[:500] for k, v in details.items()}))
    return tuple(receipts)


__all__ = ["RepairAction", "RepairFinding", "RepairHandler", "RepairPlan", "RepairReceipt", "StoreObservation", "build_repair_plan", "execute_repair_plan"]
