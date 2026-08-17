"""Durable disaster-recovery rehearsal orchestration with RTO/RPO evidence.

This module deliberately cannot promote a rehearsal target into production.  It wraps
recovery backends with an isolated, idempotent, fenced workflow that records enough
evidence to answer a narrower operational question: can a declared recovery point be
restored and verified within the configured recovery objectives?

Actual backup acquisition and production restore remain delegated to governed backend
implementations.  The included local-file backend adapts ``tools.disaster_recovery``
for offline drills only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import stat
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol, Sequence

from tools.disaster_recovery import BackupManifest, manifest_sha256, restore_backup, verify_backup


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _digest(value: str, *, label: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return text


def _text(value: str, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty.")
    return text


def _seconds(value: float, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0 or (positive and selected <= 0):
        raise ValueError(f"{label} is invalid.")
    return selected


@dataclass(frozen=True)
class RecoveryObjective:
    max_rto_seconds: float
    max_rpo_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_rto_seconds", _seconds(self.max_rto_seconds, label="max_rto_seconds", positive=True))
        object.__setattr__(self, "max_rpo_seconds", _seconds(self.max_rpo_seconds, label="max_rpo_seconds"))


@dataclass(frozen=True)
class RecoveryPoint:
    component: str
    recovery_point_id: str
    backup_manifest_sha256: str
    source_watermark_at: float
    custody_evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", _text(self.component, label="component"))
        object.__setattr__(self, "recovery_point_id", _text(self.recovery_point_id, label="recovery_point_id"))
        object.__setattr__(self, "backup_manifest_sha256", _digest(self.backup_manifest_sha256, label="backup_manifest_sha256"))
        object.__setattr__(self, "custody_evidence_sha256", _digest(self.custody_evidence_sha256, label="custody_evidence_sha256"))
        object.__setattr__(self, "source_watermark_at", _seconds(self.source_watermark_at, label="source_watermark_at"))


@dataclass(frozen=True)
class RecoveryRehearsalSpec:
    owner_id: str
    incident_at: float
    objective: RecoveryObjective
    recovery_points: tuple[RecoveryPoint, ...]
    policy_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, label="owner_id"))
        object.__setattr__(self, "incident_at", _seconds(self.incident_at, label="incident_at"))
        object.__setattr__(self, "policy_sha256", _digest(self.policy_sha256, label="policy_sha256"))
        points = tuple(self.recovery_points)
        if not points:
            raise ValueError("at least one recovery point is required.")
        components = [point.component for point in points]
        if len(set(components)) != len(components):
            raise ValueError("recovery-point components must be unique.")
        if any(point.source_watermark_at > self.incident_at for point in points):
            raise ValueError("source watermark cannot be later than the simulated incident.")
        object.__setattr__(self, "recovery_points", tuple(sorted(points, key=lambda item: item.component)))

    @property
    def drill_id(self) -> str:
        return _sha256_json(
            {
                "schema": "rigorousrag-dr-rehearsal-spec/v1",
                "owner_id": self.owner_id,
                "incident_at": self.incident_at,
                "objective": asdict(self.objective),
                "recovery_points": [asdict(point) for point in self.recovery_points],
                "policy_sha256": self.policy_sha256,
            }
        )


@dataclass(frozen=True)
class RestoreEvidence:
    component: str
    recovery_point_id: str
    restored_manifest_sha256: str
    restored_at: float
    target_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", _text(self.component, label="component"))
        object.__setattr__(self, "recovery_point_id", _text(self.recovery_point_id, label="recovery_point_id"))
        object.__setattr__(self, "restored_manifest_sha256", _digest(self.restored_manifest_sha256, label="restored_manifest_sha256"))
        object.__setattr__(self, "restored_at", _seconds(self.restored_at, label="restored_at"))
        object.__setattr__(self, "target_digest", _digest(self.target_digest, label="target_digest"))


@dataclass(frozen=True)
class VerificationEvidence:
    component: str
    recovery_point_id: str
    restored_manifest_sha256: str
    semantic_evidence_sha256: str
    verified_at: float
    ready: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", _text(self.component, label="component"))
        object.__setattr__(self, "recovery_point_id", _text(self.recovery_point_id, label="recovery_point_id"))
        object.__setattr__(self, "restored_manifest_sha256", _digest(self.restored_manifest_sha256, label="restored_manifest_sha256"))
        object.__setattr__(self, "semantic_evidence_sha256", _digest(self.semantic_evidence_sha256, label="semantic_evidence_sha256"))
        object.__setattr__(self, "verified_at", _seconds(self.verified_at, label="verified_at"))
        if not isinstance(self.ready, bool):
            raise ValueError("ready must be boolean.")


@dataclass(frozen=True)
class CleanupEvidence:
    cleaned_at: float
    target_digest: str
    removed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "cleaned_at", _seconds(self.cleaned_at, label="cleaned_at"))
        object.__setattr__(self, "target_digest", _digest(self.target_digest, label="target_digest"))
        if not isinstance(self.removed, bool):
            raise ValueError("removed must be boolean.")


@dataclass(frozen=True)
class RehearsalReceipt:
    drill_id: str
    owner_id: str
    incident_at: float
    completed_at: float
    max_observed_rpo_seconds: float
    observed_rto_seconds: float
    objective: RecoveryObjective
    verifications: tuple[VerificationEvidence, ...]
    cleanup: CleanupEvidence
    objective_met: bool
    reason_codes: tuple[str, ...]
    receipt_sha256: str

    @classmethod
    def build(
        cls,
        *,
        spec: RecoveryRehearsalSpec,
        completed_at: float,
        verifications: Sequence[VerificationEvidence],
        cleanup: CleanupEvidence,
    ) -> "RehearsalReceipt":
        completed = _seconds(completed_at, label="completed_at")
        if completed < spec.incident_at:
            raise ValueError("completed_at cannot precede incident_at.")
        ordered = tuple(sorted(verifications, key=lambda item: item.component))
        expected = {point.component: point for point in spec.recovery_points}
        if set(item.component for item in ordered) != set(expected):
            raise ValueError("verification evidence does not cover the recovery specification.")
        reasons: list[str] = []
        for item in ordered:
            point = expected[item.component]
            if item.recovery_point_id != point.recovery_point_id:
                raise ValueError("verification recovery-point identity mismatch.")
            if item.restored_manifest_sha256 != point.backup_manifest_sha256:
                raise ValueError("verification manifest identity mismatch.")
            if not item.ready:
                reasons.append(f"component_not_ready:{item.component}")
        max_rpo = max(spec.incident_at - point.source_watermark_at for point in spec.recovery_points)
        verified_ready_at = max(item.verified_at for item in ordered)
        observed_rto = max(0.0, verified_ready_at - spec.incident_at)
        if max_rpo > spec.objective.max_rpo_seconds:
            reasons.append("rpo_objective_exceeded")
        if observed_rto > spec.objective.max_rto_seconds:
            reasons.append("rto_objective_exceeded")
        if not cleanup.removed:
            reasons.append("rehearsal_target_not_removed")
        payload = {
            "schema": "rigorousrag-dr-rehearsal-receipt/v1",
            "drill_id": spec.drill_id,
            "owner_id": spec.owner_id,
            "incident_at": spec.incident_at,
            "completed_at": completed,
            "max_observed_rpo_seconds": max_rpo,
            "observed_rto_seconds": observed_rto,
            "objective": asdict(spec.objective),
            "verifications": [asdict(item) for item in ordered],
            "cleanup": asdict(cleanup),
            "objective_met": not reasons,
            "reason_codes": sorted(set(reasons)),
        }
        return cls(
            drill_id=spec.drill_id,
            owner_id=spec.owner_id,
            incident_at=spec.incident_at,
            completed_at=completed,
            max_observed_rpo_seconds=max_rpo,
            observed_rto_seconds=observed_rto,
            objective=spec.objective,
            verifications=ordered,
            cleanup=cleanup,
            objective_met=not reasons,
            reason_codes=tuple(sorted(set(reasons))),
            receipt_sha256=_sha256_json(payload),
        )


class RecoveryRehearsalBackend(Protocol):
    """Backend contract intentionally limited to isolated rehearsal targets."""

    def prepare_isolated_target(self, spec: RecoveryRehearsalSpec, *, idempotency_key: str) -> str: ...

    def restore(
        self,
        point: RecoveryPoint,
        *,
        target_ref: str,
        idempotency_key: str,
        now: float,
    ) -> RestoreEvidence: ...

    def verify(
        self,
        point: RecoveryPoint,
        *,
        target_ref: str,
        restore: RestoreEvidence,
        idempotency_key: str,
        now: float,
    ) -> VerificationEvidence: ...

    def cleanup(self, *, target_ref: str, idempotency_key: str, now: float) -> CleanupEvidence: ...


@dataclass(frozen=True)
class DrillLease:
    drill_id: str
    worker_id: str
    fencing_token: int
    expires_at: float


@dataclass(frozen=True)
class DrillRecord:
    spec: RecoveryRehearsalSpec
    state: str
    revision: int
    target_ref: str | None
    component_index: int
    restores: tuple[RestoreEvidence, ...]
    verifications: tuple[VerificationEvidence, ...]
    cleanup: CleanupEvidence | None
    receipt: RehearsalReceipt | None
    last_error: str | None
    updated_at: float


_STATES = {
    "planned",
    "prepare_requested",
    "restoring",
    "restore_requested",
    "verify_requested",
    "cleanup_requested",
    "completed",
    "failed",
}


class SQLiteRecoveryRehearsalStore:
    """SQLite journal with monotonic fencing and transaction-bound CAS transitions."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS dr_rehearsals (
                    drill_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    lease_worker TEXT,
                    lease_token INTEGER NOT NULL DEFAULT 0,
                    lease_expires REAL NOT NULL DEFAULT 0
                )"""
            )

    @staticmethod
    def _payload(record: DrillRecord) -> str:
        value = asdict(record)
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _record(payload: str) -> DrillRecord:
        raw = json.loads(payload)
        spec_raw = raw["spec"]
        objective = RecoveryObjective(**spec_raw["objective"])
        points = tuple(RecoveryPoint(**item) for item in spec_raw["recovery_points"])
        spec = RecoveryRehearsalSpec(
            owner_id=spec_raw["owner_id"],
            incident_at=spec_raw["incident_at"],
            objective=objective,
            recovery_points=points,
            policy_sha256=spec_raw["policy_sha256"],
        )
        restores = tuple(RestoreEvidence(**item) for item in raw["restores"])
        verifications = tuple(VerificationEvidence(**item) for item in raw["verifications"])
        cleanup = CleanupEvidence(**raw["cleanup"]) if raw["cleanup"] is not None else None
        receipt = None
        if raw["receipt"] is not None:
            item = raw["receipt"]
            receipt = RehearsalReceipt(
                drill_id=item["drill_id"],
                owner_id=item["owner_id"],
                incident_at=item["incident_at"],
                completed_at=item["completed_at"],
                max_observed_rpo_seconds=item["max_observed_rpo_seconds"],
                observed_rto_seconds=item["observed_rto_seconds"],
                objective=RecoveryObjective(**item["objective"]),
                verifications=tuple(VerificationEvidence(**value) for value in item["verifications"]),
                cleanup=CleanupEvidence(**item["cleanup"]),
                objective_met=item["objective_met"],
                reason_codes=tuple(item["reason_codes"]),
                receipt_sha256=item["receipt_sha256"],
            )
        return DrillRecord(
            spec=spec,
            state=raw["state"],
            revision=int(raw["revision"]),
            target_ref=raw["target_ref"],
            component_index=int(raw["component_index"]),
            restores=restores,
            verifications=verifications,
            cleanup=cleanup,
            receipt=receipt,
            last_error=raw["last_error"],
            updated_at=float(raw["updated_at"]),
        )

    def ensure(self, spec: RecoveryRehearsalSpec, *, now: float) -> DrillRecord:
        timestamp = _seconds(now, label="now")
        created = DrillRecord(spec, "planned", 0, None, 0, (), (), None, None, None, timestamp)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT payload FROM dr_rehearsals WHERE drill_id = ?", (spec.drill_id,)).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO dr_rehearsals(drill_id,payload,revision,updated_at) VALUES(?,?,?,?)",
                    (spec.drill_id, self._payload(created), 0, timestamp),
                )
                return created
            existing = self._record(row["payload"])
            if existing.spec != spec:
                raise RuntimeError("drill identity collision with different specification.")
            return existing

    def get(self, drill_id: str) -> DrillRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM dr_rehearsals WHERE drill_id = ?", (drill_id,)).fetchone()
        if row is None:
            raise KeyError(drill_id)
        return self._record(row["payload"])

    def claim(self, drill_id: str, *, worker_id: str, now: float, lease_seconds: float) -> DrillLease:
        worker = _text(worker_id, label="worker_id")
        timestamp = _seconds(now, label="now")
        duration = _seconds(lease_seconds, label="lease_seconds", positive=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT lease_worker,lease_token,lease_expires FROM dr_rehearsals WHERE drill_id = ?",
                (drill_id,),
            ).fetchone()
            if row is None:
                raise KeyError(drill_id)
            live = row["lease_worker"] is not None and float(row["lease_expires"]) > timestamp
            if live and row["lease_worker"] != worker:
                raise RuntimeError("drill is claimed by another worker.")
            token = int(row["lease_token"])
            if not live or row["lease_worker"] != worker:
                token += 1
            expires = timestamp + duration
            connection.execute(
                "UPDATE dr_rehearsals SET lease_worker=?,lease_token=?,lease_expires=? WHERE drill_id=?",
                (worker, token, expires, drill_id),
            )
        return DrillLease(drill_id, worker, token, expires)

    def transition(
        self,
        lease: DrillLease,
        *,
        expected_state: str,
        expected_revision: int,
        state: str,
        now: float,
        **changes: object,
    ) -> DrillRecord:
        if expected_state not in _STATES or state not in _STATES:
            raise ValueError("unknown drill state.")
        timestamp = _seconds(now, label="now")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload,revision,lease_worker,lease_token,lease_expires FROM dr_rehearsals WHERE drill_id=?",
                (lease.drill_id,),
            ).fetchone()
            if row is None:
                raise KeyError(lease.drill_id)
            if (
                row["lease_worker"] != lease.worker_id
                or int(row["lease_token"]) != lease.fencing_token
                or float(row["lease_expires"]) <= timestamp
            ):
                raise RuntimeError("drill lease is expired or fenced.")
            current = self._record(row["payload"])
            if current.state != expected_state or current.revision != expected_revision or int(row["revision"]) != expected_revision:
                raise RuntimeError("drill state changed concurrently.")
            permitted = {
                "target_ref",
                "component_index",
                "restores",
                "verifications",
                "cleanup",
                "receipt",
                "last_error",
            }
            unknown = set(changes) - permitted
            if unknown:
                raise ValueError(f"unsupported transition fields: {sorted(unknown)}")
            next_record = replace(current, state=state, revision=current.revision + 1, updated_at=timestamp, **changes)
            cursor = connection.execute(
                "UPDATE dr_rehearsals SET payload=?,revision=?,updated_at=? WHERE drill_id=? AND revision=?",
                (self._payload(next_record), next_record.revision, timestamp, lease.drill_id, expected_revision),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("drill state CAS failed.")
            return next_record


@dataclass(frozen=True)
class LocalFileBackupAsset:
    component: str
    source: Path
    manifest: BackupManifest
    semantic_evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", _text(self.component, label="component"))
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "semantic_evidence_sha256", _digest(self.semantic_evidence_sha256, label="semantic_evidence_sha256"))


class LocalFileRecoveryRehearsalBackend:
    """Offline adapter over ``tools.disaster_recovery`` confined to one isolation root."""

    def __init__(self, isolation_root: str | Path, assets: Sequence[LocalFileBackupAsset]) -> None:
        root = Path(isolation_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        info = root.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("isolation_root must be a real directory, not a symlink.")
        self.root = root
        self.assets = {asset.component: asset for asset in assets}
        if len(self.assets) != len(tuple(assets)):
            raise ValueError("local rehearsal assets must have unique components.")

    def _target(self, target_ref: str) -> Path:
        candidate = Path(target_ref).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("rehearsal target escaped the isolation root.") from exc
        if candidate == self.root:
            raise ValueError("the isolation root itself cannot be a drill target.")
        return candidate

    @staticmethod
    def _target_digest(path: Path) -> str:
        return hashlib.sha256(str(path).encode("utf-8")).hexdigest()

    def prepare_isolated_target(self, spec: RecoveryRehearsalSpec, *, idempotency_key: str) -> str:
        _text(idempotency_key, label="idempotency_key")
        target = self._target(str(self.root / spec.drill_id))
        target.mkdir(parents=False, exist_ok=True)
        return str(target)

    def restore(self, point: RecoveryPoint, *, target_ref: str, idempotency_key: str, now: float) -> RestoreEvidence:
        _text(idempotency_key, label="idempotency_key")
        asset = self.assets.get(point.component)
        if asset is None:
            raise KeyError(f"no local backup asset for {point.component}")
        actual_manifest = manifest_sha256(asset.manifest)
        if actual_manifest != point.backup_manifest_sha256:
            raise RuntimeError("registered backup manifest differs from the governed recovery point.")
        if not verify_backup(source=asset.source, manifest=asset.manifest):
            raise RuntimeError("backup bytes failed checksum verification before rehearsal restore.")
        target = self._target(target_ref)
        component_target = target / point.component
        component_target.mkdir(parents=False, exist_ok=True)
        report = restore_backup(source=asset.source, destination=component_target, manifest=asset.manifest)
        if report.manifest_sha256 != point.backup_manifest_sha256:
            raise RuntimeError("restore report manifest mismatch.")
        return RestoreEvidence(
            component=point.component,
            recovery_point_id=point.recovery_point_id,
            restored_manifest_sha256=report.manifest_sha256,
            restored_at=_seconds(now, label="now"),
            target_digest=self._target_digest(component_target.resolve()),
        )

    def verify(
        self,
        point: RecoveryPoint,
        *,
        target_ref: str,
        restore: RestoreEvidence,
        idempotency_key: str,
        now: float,
    ) -> VerificationEvidence:
        _text(idempotency_key, label="idempotency_key")
        asset = self.assets.get(point.component)
        if asset is None:
            raise KeyError(point.component)
        target = self._target(target_ref) / point.component
        ready = (
            restore.component == point.component
            and restore.recovery_point_id == point.recovery_point_id
            and restore.restored_manifest_sha256 == point.backup_manifest_sha256
            and verify_backup(source=target, manifest=asset.manifest)
        )
        return VerificationEvidence(
            component=point.component,
            recovery_point_id=point.recovery_point_id,
            restored_manifest_sha256=point.backup_manifest_sha256,
            semantic_evidence_sha256=asset.semantic_evidence_sha256,
            verified_at=_seconds(now, label="now"),
            ready=ready,
        )

    def cleanup(self, *, target_ref: str, idempotency_key: str, now: float) -> CleanupEvidence:
        _text(idempotency_key, label="idempotency_key")
        target = self._target(target_ref)
        digest = self._target_digest(target)
        if target.exists():
            info = target.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("rehearsal target changed type before cleanup.")
            shutil.rmtree(target)
        return CleanupEvidence(_seconds(now, label="now"), digest, not target.exists())


def _point(record: DrillRecord) -> RecoveryPoint:
    if record.component_index < 0 or record.component_index >= len(record.spec.recovery_points):
        raise RuntimeError("drill component cursor is out of range.")
    return record.spec.recovery_points[record.component_index]


def advance_recovery_rehearsal(
    *,
    store: SQLiteRecoveryRehearsalStore,
    spec: RecoveryRehearsalSpec,
    backend: RecoveryRehearsalBackend,
    worker_id: str,
    now: float,
    lease_seconds: float = 60.0,
) -> DrillRecord:
    """Advance at most one external side-effect boundary for a crash-resumable drill.

    Request states are persisted *before* backend calls.  Re-entry therefore reissues the
    same deterministic idempotency key instead of guessing whether an interrupted call
    completed.  Backends used here must honor that key or otherwise make the operation
    naturally idempotent.
    """

    timestamp = _seconds(now, label="now")
    store.ensure(spec, now=timestamp)
    lease = store.claim(spec.drill_id, worker_id=worker_id, now=timestamp, lease_seconds=lease_seconds)
    record = store.get(spec.drill_id)
    if record.state in {"completed", "failed"}:
        return record

    try:
        if record.state == "planned":
            return store.transition(
                lease,
                expected_state="planned",
                expected_revision=record.revision,
                state="prepare_requested",
                now=timestamp,
                last_error=None,
            )

        if record.state == "prepare_requested":
            target_ref = backend.prepare_isolated_target(spec, idempotency_key=f"{spec.drill_id}:prepare")
            return store.transition(
                lease,
                expected_state="prepare_requested",
                expected_revision=record.revision,
                state="restoring",
                now=timestamp,
                target_ref=_text(target_ref, label="target_ref"),
                component_index=0,
                last_error=None,
            )

        if record.state == "restoring":
            return store.transition(
                lease,
                expected_state="restoring",
                expected_revision=record.revision,
                state="restore_requested",
                now=timestamp,
                last_error=None,
            )

        if record.state == "restore_requested":
            if record.target_ref is None:
                raise RuntimeError("rehearsal target is missing.")
            point = _point(record)
            evidence = backend.restore(
                point,
                target_ref=record.target_ref,
                idempotency_key=f"{spec.drill_id}:{point.component}:restore",
                now=timestamp,
            )
            if evidence.component != point.component or evidence.recovery_point_id != point.recovery_point_id:
                raise RuntimeError("restore backend returned evidence for the wrong recovery point.")
            if evidence.restored_manifest_sha256 != point.backup_manifest_sha256:
                raise RuntimeError("restore evidence is not bound to the governed manifest.")
            return store.transition(
                lease,
                expected_state="restore_requested",
                expected_revision=record.revision,
                state="verify_requested",
                now=timestamp,
                restores=record.restores + (evidence,),
                last_error=None,
            )

        if record.state == "verify_requested":
            if record.target_ref is None or not record.restores:
                raise RuntimeError("restore evidence is missing before verification.")
            point = _point(record)
            restore = record.restores[-1]
            evidence = backend.verify(
                point,
                target_ref=record.target_ref,
                restore=restore,
                idempotency_key=f"{spec.drill_id}:{point.component}:verify",
                now=timestamp,
            )
            if evidence.component != point.component or evidence.recovery_point_id != point.recovery_point_id:
                raise RuntimeError("verification backend returned evidence for the wrong recovery point.")
            if evidence.restored_manifest_sha256 != point.backup_manifest_sha256:
                raise RuntimeError("verification evidence is not bound to the governed manifest.")
            verifications = record.verifications + (evidence,)
            next_index = record.component_index + 1
            if next_index < len(spec.recovery_points):
                return store.transition(
                    lease,
                    expected_state="verify_requested",
                    expected_revision=record.revision,
                    state="restore_requested",
                    now=timestamp,
                    verifications=verifications,
                    component_index=next_index,
                    last_error=None,
                )
            return store.transition(
                lease,
                expected_state="verify_requested",
                expected_revision=record.revision,
                state="cleanup_requested",
                now=timestamp,
                verifications=verifications,
                component_index=next_index,
                last_error=None,
            )

        if record.state == "cleanup_requested":
            if record.target_ref is None:
                raise RuntimeError("rehearsal target is missing before cleanup.")
            cleanup = backend.cleanup(
                target_ref=record.target_ref,
                idempotency_key=f"{spec.drill_id}:cleanup",
                now=timestamp,
            )
            receipt = RehearsalReceipt.build(
                spec=spec,
                completed_at=timestamp,
                verifications=record.verifications,
                cleanup=cleanup,
            )
            return store.transition(
                lease,
                expected_state="cleanup_requested",
                expected_revision=record.revision,
                state="completed",
                now=timestamp,
                cleanup=cleanup,
                receipt=receipt,
                last_error=None,
            )

        raise RuntimeError(f"unsupported drill state: {record.state}")
    except Exception as exc:
        # External-call failures remain retryable in their persisted request state.  A
        # deterministic source/evidence violation is surfaced to the operator without
        # silently advancing the workflow.
        current = store.get(spec.drill_id)
        if current.state not in {"completed", "failed"}:
            try:
                return store.transition(
                    lease,
                    expected_state=current.state,
                    expected_revision=current.revision,
                    state=current.state,
                    now=timestamp,
                    last_error=f"{type(exc).__name__}:{exc}",
                )
            except RuntimeError:
                pass
        raise


__all__ = [
    "CleanupEvidence",
    "DrillLease",
    "DrillRecord",
    "LocalFileBackupAsset",
    "LocalFileRecoveryRehearsalBackend",
    "RecoveryObjective",
    "RecoveryPoint",
    "RecoveryRehearsalBackend",
    "RecoveryRehearsalSpec",
    "RehearsalReceipt",
    "RestoreEvidence",
    "SQLiteRecoveryRehearsalStore",
    "VerificationEvidence",
    "advance_recovery_rehearsal",
]
