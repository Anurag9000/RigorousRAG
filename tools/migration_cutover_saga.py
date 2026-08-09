"""Cutover saga and verified compensation contract.

The concrete single-host, same-dimension adapter lives in
``tools.migration_cutover_local``. Dimension-changing blue/green and distributed
adapters remain separate deployment concerns. This module owns mutation/recovery
ordering and the adapter protocol only.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from tools.migration_cutover_control import CutoverOperation, CutoverPreparation
from tools.migration_types import digest, exact_integer, identifier

_PHASES = {
    "lock_acquired",
    "source_revalidated",
    "hidden_target_written",
    "hidden_target_validated",
    "visibility_committed",
    "visible_target_validated",
    "hidden_target_discarded",
    "rollback_restored",
    "rollback_validated",
}


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BackendStateIdentity:
    source_sequence: int
    profile_fingerprint: str
    content_sha256: str
    vector_snapshot_digest: str
    sparse_snapshot_digest: str
    vector_rows: int
    sparse_generation: int
    sparse_fields: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_sequence",
            exact_integer(self.source_sequence, "source_sequence", 1, 2**63 - 1),
        )
        for name in (
            "profile_fingerprint",
            "content_sha256",
            "vector_snapshot_digest",
            "sparse_snapshot_digest",
        ):
            object.__setattr__(self, name, digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "vector_rows",
            exact_integer(self.vector_rows, "vector_rows", 1, 100_000_000),
        )
        object.__setattr__(
            self,
            "sparse_generation",
            exact_integer(self.sparse_generation, "sparse_generation", 1, 2**63 - 1),
        )
        object.__setattr__(
            self,
            "sparse_fields",
            exact_integer(self.sparse_fields, "sparse_fields", 1, 100_000_000),
        )

    @classmethod
    def from_preparation(cls, value: CutoverPreparation) -> "BackendStateIdentity":
        if not isinstance(value, CutoverPreparation):
            raise ValueError("value must be CutoverPreparation.")
        return cls(
            source_sequence=value.source_sequence,
            profile_fingerprint=value.source_profile_fingerprint,
            content_sha256=value.source_content_sha256,
            vector_snapshot_digest=value.vector_snapshot_digest,
            sparse_snapshot_digest=value.sparse_snapshot_digest,
            vector_rows=value.source_vector_rows,
            sparse_generation=value.source_sparse_generation,
            sparse_fields=value.source_sparse_fields,
        )


@dataclass(frozen=True)
class TargetPublication:
    publication_id: str
    target_artifact_digest: str
    target_profile_fingerprint: str
    content_sha256: str
    vector_rows: int
    sparse_rows: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "publication_id",
            digest(self.publication_id, "publication_id"),
        )
        for name in (
            "target_artifact_digest",
            "target_profile_fingerprint",
            "content_sha256",
        ):
            object.__setattr__(self, name, digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "vector_rows",
            exact_integer(self.vector_rows, "vector_rows", 1, 100_000_000),
        )
        object.__setattr__(
            self,
            "sparse_rows",
            exact_integer(self.sparse_rows, "sparse_rows", 1, 100_000_000),
        )

    @classmethod
    def expected(cls, value: CutoverPreparation) -> "TargetPublication":
        if not isinstance(value, CutoverPreparation):
            raise ValueError("value must be CutoverPreparation.")
        publication_id = _sha256(
            {
                "contract": "rigorousrag-hidden-target-publication-v1",
                "operation_id": value.operation_id,
                "target_artifact_digest": value.target_artifact_digest,
            }
        )
        return cls(
            publication_id=publication_id,
            target_artifact_digest=value.target_artifact_digest,
            target_profile_fingerprint=value.target_profile_fingerprint,
            content_sha256=value.source_content_sha256,
            vector_rows=value.target_vector_rows,
            sparse_rows=value.target_sparse_rows,
        )


@dataclass(frozen=True)
class CutoverSagaResult:
    operation_id: str
    outcome: str
    phases: tuple[str, ...]
    failure_type: str | None
    publication_id: str | None
    rollback_verified: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            digest(self.operation_id, "operation_id"),
        )
        if self.outcome not in {"published", "aborted", "rolled_back"}:
            raise ValueError("cutover saga outcome is invalid.")
        phases = tuple(identifier(item, "phase", 100) for item in self.phases)
        if any(item not in _PHASES for item in phases):
            raise ValueError("cutover saga contains an unknown phase.")
        object.__setattr__(self, "phases", phases)
        if self.failure_type is not None:
            object.__setattr__(
                self,
                "failure_type",
                identifier(self.failure_type, "failure_type", 200),
            )
        if self.publication_id is not None:
            object.__setattr__(
                self,
                "publication_id",
                digest(self.publication_id, "publication_id"),
            )
        if not isinstance(self.rollback_verified, bool):
            raise ValueError("rollback_verified must be boolean.")
        if self.outcome == "published" and (
            self.failure_type is not None
            or self.publication_id is None
            or self.rollback_verified
        ):
            raise ValueError("published cutover saga result is inconsistent.")
        if self.outcome == "aborted" and self.rollback_verified:
            raise ValueError("aborted pre-publication result may not claim rollback.")
        if self.outcome == "rolled_back" and (
            self.failure_type is None or not self.rollback_verified
        ):
            raise ValueError("rolled-back result requires verified compensation.")

    @property
    def trace_digest(self) -> str:
        return _sha256(asdict(self))


class CutoverRecoveryError(RuntimeError):
    """Raised when compensation cannot restore and verify the prior generation."""

    def __init__(self, phase: str, failure_type: str, recovery_failure_type: str) -> None:
        self.phase = identifier(phase, "phase", 100)
        self.failure_type = identifier(failure_type, "failure_type", 200)
        self.recovery_failure_type = identifier(
            recovery_failure_type,
            "recovery_failure_type",
            200,
        )
        super().__init__("cutover recovery failed after a bounded adapter error.")


@runtime_checkable
class CutoverBackendAdapter(Protocol):
    def exclusive_lock(self, operation: CutoverOperation) -> AbstractContextManager[Any]: ...
    def current_identity(self, operation: CutoverOperation) -> BackendStateIdentity: ...
    def write_hidden_target(self, operation: CutoverOperation) -> TargetPublication: ...
    def validate_hidden_target(
        self, operation: CutoverOperation, publication: TargetPublication
    ) -> TargetPublication: ...
    def commit_visibility(
        self, operation: CutoverOperation, publication: TargetPublication
    ) -> None: ...
    def validate_visible_target(
        self, operation: CutoverOperation, publication: TargetPublication
    ) -> None: ...
    def discard_hidden_target(
        self, operation: CutoverOperation, publication: TargetPublication
    ) -> None: ...
    def restore_rollback(self, operation: CutoverOperation) -> None: ...
    def validate_rollback(self, operation: CutoverOperation) -> None: ...


def _require_source(
    operation: CutoverOperation,
    actual: BackendStateIdentity,
) -> None:
    if not isinstance(actual, BackendStateIdentity):
        raise RuntimeError("cutover adapter returned invalid source identity.")
    if actual != BackendStateIdentity.from_preparation(operation.preparation):
        raise RuntimeError("authoritative source identity changed before publication.")


def _require_target(
    operation: CutoverOperation,
    actual: TargetPublication,
) -> TargetPublication:
    if not isinstance(actual, TargetPublication):
        raise RuntimeError("cutover adapter returned invalid target publication.")
    expected = TargetPublication.expected(operation.preparation)
    if actual != expected:
        raise RuntimeError("hidden target publication does not match preparation.")
    return actual


def execute_cutover_saga(
    operation: CutoverOperation,
    adapter: CutoverBackendAdapter,
    *,
    fault_hook: Callable[[str], None] | None = None,
) -> CutoverSagaResult:
    """Execute a cutover adapter with mandatory verified compensation.

    Callers select an adapter explicitly. The repository includes a concrete local
    same-dimension adapter in ``tools.migration_cutover_local``; alternative
    blue/green or distributed adapters must satisfy the same protocol. Publication
    and compensation execute inside the adapter-provided exclusive lock.
    """

    if not isinstance(operation, CutoverOperation) or operation.state != "ready":
        raise ValueError("cutover saga requires one ready cutover operation.")
    required_methods = (
        "exclusive_lock",
        "current_identity",
        "write_hidden_target",
        "validate_hidden_target",
        "commit_visibility",
        "validate_visible_target",
        "discard_hidden_target",
        "restore_rollback",
        "validate_rollback",
    )
    if any(not callable(getattr(adapter, name, None)) for name in required_methods):
        raise ValueError("cutover adapter does not implement the required contract.")
    hook = fault_hook or (lambda phase: None)
    phases: list[str] = []

    try:
        lock = adapter.exclusive_lock(operation)
        if not isinstance(lock, AbstractContextManager):
            if not (hasattr(lock, "__enter__") and hasattr(lock, "__exit__")):
                raise ValueError("cutover adapter returned an invalid exclusive lock.")
        with lock:
            phases.append("lock_acquired")
            hook("lock_acquired")
            publication: TargetPublication | None = None
            visible = False
            failure_phase = "lock_acquired"
            try:
                _require_source(operation, adapter.current_identity(operation))
                failure_phase = "source_revalidated"
                phases.append("source_revalidated")
                hook("source_revalidated")

                publication = adapter.write_hidden_target(operation)
                publication = _require_target(operation, publication)
                failure_phase = "hidden_target_written"
                phases.append("hidden_target_written")
                hook("hidden_target_written")

                validated = _require_target(
                    operation,
                    adapter.validate_hidden_target(operation, publication),
                )
                if validated != publication:
                    raise RuntimeError(
                        "hidden target validation changed publication identity."
                    )
                failure_phase = "hidden_target_validated"
                phases.append("hidden_target_validated")
                hook("hidden_target_validated")

                adapter.commit_visibility(operation, publication)
                visible = True
                failure_phase = "visibility_committed"
                phases.append("visibility_committed")
                hook("visibility_committed")

                adapter.validate_visible_target(operation, publication)
                failure_phase = "visible_target_validated"
                phases.append("visible_target_validated")
                hook("visible_target_validated")
                return CutoverSagaResult(
                    operation_id=operation.operation_id,
                    outcome="published",
                    phases=tuple(phases),
                    failure_type=None,
                    publication_id=publication.publication_id,
                    rollback_verified=False,
                )
            except Exception as exc:
                failure_type = type(exc).__name__
                try:
                    if visible:
                        adapter.restore_rollback(operation)
                        phases.append("rollback_restored")
                        hook("rollback_restored")
                        adapter.validate_rollback(operation)
                        phases.append("rollback_validated")
                        hook("rollback_validated")
                        return CutoverSagaResult(
                            operation_id=operation.operation_id,
                            outcome="rolled_back",
                            phases=tuple(phases),
                            failure_type=failure_type,
                            publication_id=(
                                publication.publication_id
                                if publication is not None
                                else None
                            ),
                            rollback_verified=True,
                        )
                    if publication is not None:
                        adapter.discard_hidden_target(operation, publication)
                        phases.append("hidden_target_discarded")
                        hook("hidden_target_discarded")
                        _require_source(operation, adapter.current_identity(operation))
                    return CutoverSagaResult(
                        operation_id=operation.operation_id,
                        outcome="aborted",
                        phases=tuple(phases),
                        failure_type=failure_type,
                        publication_id=(
                            publication.publication_id
                            if publication is not None
                            else None
                        ),
                        rollback_verified=False,
                    )
                except Exception as recovery_exc:
                    raise CutoverRecoveryError(
                        failure_phase,
                        failure_type,
                        type(recovery_exc).__name__,
                    ) from recovery_exc
    except CutoverRecoveryError:
        raise
    except Exception as exc:
        return CutoverSagaResult(
            operation_id=operation.operation_id,
            outcome="aborted",
            phases=tuple(phases),
            failure_type=type(exc).__name__,
            publication_id=None,
            rollback_verified=False,
        )


__all__ = [
    "BackendStateIdentity",
    "CutoverBackendAdapter",
    "CutoverRecoveryError",
    "CutoverSagaResult",
    "TargetPublication",
    "execute_cutover_saga",
]
