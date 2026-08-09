"""Crash-resumable and cross-process-fenced blue/green cutover adapter."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any

from tools.migration_cutover_blue_green import BlueGreenCutoverBackendAdapter
from tools.migration_cutover_control import CutoverOperation
from tools.migration_cutover_saga import TargetPublication
from tools.migration_target_population import (
    TargetPopulationClaim,
    TargetPopulationIdentity,
    TargetPopulationJournal,
    TargetPopulationReconciliation,
    reconcile_target_population,
)


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class DurableBlueGreenCutoverBackendAdapter(BlueGreenCutoverBackendAdapter):
    """Blue/green adapter with durable hidden-population recovery evidence.

    The existing adapter remains the mutation implementation. This subclass wraps it
    with a journal intent written before physical target mutation, a monotonic-fenced
    cross-process executor lease, deterministic readback receipts, retry/resume of an
    already-complete hidden population, and reconciliation/finalization entry points
    for process death after physical publication or route visibility.
    """

    def __init__(
        self,
        *,
        population_journal: TargetPopulationJournal,
        worker_id: str | None = None,
        lease_seconds: float = 3_600.0,
        **kwargs: Any,
    ) -> None:
        if not isinstance(population_journal, TargetPopulationJournal):
            raise ValueError("population_journal must be TargetPopulationJournal.")
        selected_worker = worker_id or f"blue-green-{uuid.uuid4().hex}"
        if not isinstance(selected_worker, str) or not selected_worker.strip():
            raise ValueError("worker_id must be a non-empty string.")
        if isinstance(lease_seconds, bool):
            raise ValueError("lease_seconds must be finite and positive.")
        try:
            selected_lease = float(lease_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("lease_seconds must be finite and positive.") from exc
        if not 0.0 < selected_lease <= 86_400.0:
            raise ValueError("lease_seconds must be between 0 and 86400.")
        super().__init__(**kwargs)
        self.population_journal = population_journal
        self.worker_id = selected_worker.strip()
        self.lease_seconds = selected_lease
        self._population_claim: TargetPopulationClaim | None = None
        self._population_identity_value: TargetPopulationIdentity | None = None

    def _population_identity(self, operation: CutoverOperation) -> TargetPopulationIdentity:
        self._bind(operation)
        if self._population_identity_value is not None:
            return self._population_identity_value
        preparation = operation.preparation
        target_spec = self.registry.collection_for_profile(
            preparation.target_profile_fingerprint
        )
        if target_spec is None or target_spec.state != "ready":
            raise RuntimeError("target physical vector collection is not registered and ready.")
        self._target_spec = target_spec
        identity = TargetPopulationIdentity(
            operation_id=operation.operation_id,
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
            target_collection_id=target_spec.collection_id,
            target_profile_fingerprint=preparation.target_profile_fingerprint,
            content_sha256=preparation.source_content_sha256,
            target_artifact_digest=preparation.target_artifact_digest,
            expected_vector_rows=preparation.target_vector_rows,
        )
        self._population_identity_value = identity
        return identity

    def _renew_execution(self) -> TargetPopulationClaim:
        claim = self._population_claim
        if claim is None:
            raise RuntimeError("durable blue-green executor lease is not held.")
        renewed = self.population_journal.renew(
            claim,
            now=self._now(),
            lease_seconds=self.lease_seconds,
        )
        self._population_claim = renewed
        return renewed

    def _target_rows(self, operation: CutoverOperation) -> tuple[Any, ...]:
        if self._target_spec is None:
            self._population_identity(operation)
        if self._target_spec is None:
            raise RuntimeError("target physical collection is unresolved.")
        return self._collection_rows(self.provider.collection(self._target_spec), operation)

    def _target_rows_match(self, operation: CutoverOperation) -> bool:
        if self._target is None:
            self._target = self._load_target(operation)
        actual = self._target_rows(operation)
        expected = tuple(sorted(self._target.vectors, key=lambda row: row.row_id))
        if len(actual) != len(expected):
            return False
        preparation = operation.preparation
        for current, planned in zip(actual, expected, strict=True):
            if (
                current.row_id != planned.row_id
                or current.text != planned.text
                or current.metadata.get("owner_id") != preparation.owner_id
                or current.metadata.get("doc_id") != preparation.doc_id
                or current.metadata.get("content_sha256")
                != preparation.source_content_sha256
                or current.metadata.get("embedding_profile_fingerprint")
                != preparation.target_profile_fingerprint
                or current.metadata.get("migration_operation_id") != operation.operation_id
                or current.metadata.get("migration_target_artifact_digest")
                != preparation.target_artifact_digest
                or len(current.embedding) != len(planned.embedding)
                or any(
                    abs(left - right) > max(1e-7, 1e-6 * max(abs(left), abs(right)))
                    for left, right in zip(current.embedding, planned.embedding, strict=True)
                )
            ):
                return False
        return True

    def _row_digest(self, operation: CutoverOperation) -> str:
        rows = self._target_rows(operation)
        stable = []
        for row in rows:
            stable.append(
                {
                    "row_id": row.row_id,
                    "text_sha256": hashlib.sha256(row.text.encode("utf-8")).hexdigest(),
                    "owner_id": row.metadata.get("owner_id"),
                    "doc_id": row.metadata.get("doc_id"),
                    "content_sha256": row.metadata.get("content_sha256"),
                    "profile_fingerprint": row.metadata.get(
                        "embedding_profile_fingerprint"
                    ),
                    "operation_id": row.metadata.get("migration_operation_id"),
                    "artifact_digest": row.metadata.get(
                        "migration_target_artifact_digest"
                    ),
                    "embedding_sha256": _sha256(list(row.embedding)),
                }
            )
        return _sha256(stable)

    def _route_digest(self, route: Any) -> str:
        try:
            payload = asdict(route)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("vector route cannot be serialized for recovery evidence.") from exc
        return _sha256(payload)

    @contextmanager
    def exclusive_lock(self, operation: CutoverOperation) -> Iterator[None]:
        local_lock = super().exclusive_lock(operation)
        with local_lock:
            identity = self._population_identity(operation)
            existing = self.population_journal.get(operation.operation_id)
            rows_before_intent = self._target_rows(operation)
            route = self.registry.current_route(
                operation.preparation.owner_id,
                operation.preparation.doc_id,
            )
            if existing is None and rows_before_intent:
                if route is not None and route.collection_id == identity.target_collection_id:
                    raise RuntimeError(
                        "authoritative target collection is visible without population receipt."
                    )
                raise RuntimeError(
                    "unexplained target rows exist without durable population intent."
                )
            self.population_journal.ensure_intent(identity, now=self._now())
            claim = self.population_journal.claim(
                operation.operation_id,
                worker_id=self.worker_id,
                now=self._now(),
                lease_seconds=self.lease_seconds,
            )
            self._population_claim = claim
            active_error = False
            try:
                yield
            except BaseException:
                active_error = True
                raise
            finally:
                current = self._population_claim
                self._population_claim = None
                if current is not None:
                    try:
                        self.population_journal.release(current, now=self._now())
                    except RuntimeError:
                        if not active_error:
                            raise

    def write_hidden_target(self, operation: CutoverOperation) -> TargetPublication:
        self._renew_execution()
        identity = self._population_identity(operation)
        record = self.population_journal.get(operation.operation_id)
        if record is None or record.identity != identity:
            raise RuntimeError("durable target population intent is missing or changed.")
        if record.state in {"visible", "aborted", "rolled_back"}:
            raise RuntimeError("durable target population is already terminal.")
        self._target = self._load_target(operation)
        current_route = self.registry.current_route(
            operation.preparation.owner_id,
            operation.preparation.doc_id,
        )
        if (
            current_route is not None
            and current_route.collection_id == identity.target_collection_id
        ):
            raise RuntimeError("target route became authoritative before hidden publication.")
        rows = self._target_rows(operation)
        if rows and self._target_rows_match(operation):
            return TargetPublication.expected(operation.preparation)
        return super().write_hidden_target(operation)

    def validate_hidden_target(
        self,
        operation: CutoverOperation,
        publication: TargetPublication,
    ) -> TargetPublication:
        self._renew_execution()
        validated = super().validate_hidden_target(operation, publication)
        if not self._target_rows_match(operation):
            raise RuntimeError("durable target population readback is not exact.")
        self.population_journal.mark_populated(
            self._population_identity(operation),
            row_digest=self._row_digest(operation),
            now=self._now(),
        )
        return validated

    def commit_visibility(
        self,
        operation: CutoverOperation,
        publication: TargetPublication,
    ) -> None:
        self._renew_execution()
        record = self.population_journal.get(operation.operation_id)
        if record is None or record.state != "populated":
            raise RuntimeError("visibility requires a durable populated receipt.")
        super().commit_visibility(operation, publication)

    def validate_visible_target(
        self,
        operation: CutoverOperation,
        publication: TargetPublication,
    ) -> None:
        self._renew_execution()
        super().validate_visible_target(operation, publication)
        route = self.registry.current_route(
            operation.preparation.owner_id,
            operation.preparation.doc_id,
        )
        generation = self.generations.current(
            owner_id=operation.preparation.owner_id,
            doc_id=operation.preparation.doc_id,
        )
        if route is None or generation is None:
            raise RuntimeError("visible target route/generation disappeared.")
        self.population_journal.mark_visible(
            self._population_identity(operation),
            row_digest=self._row_digest(operation),
            route_digest=self._route_digest(route),
            generation_sequence=generation.sequence,
            now=self._now(),
        )

    def discard_hidden_target(
        self,
        operation: CutoverOperation,
        publication: TargetPublication,
    ) -> None:
        self._renew_execution()
        identity = self._population_identity(operation)
        route = self.registry.current_route(
            operation.preparation.owner_id,
            operation.preparation.doc_id,
        )
        if route is not None and route.collection_id == identity.target_collection_id:
            raise RuntimeError("authoritative target collection may not be discarded.")
        super().discard_hidden_target(operation, publication)
        self.population_journal.mark_aborted(identity, now=self._now())

    def restore_rollback(self, operation: CutoverOperation) -> None:
        self._renew_execution()
        super().restore_rollback(operation)

    def validate_rollback(self, operation: CutoverOperation) -> None:
        self._renew_execution()
        super().validate_rollback(operation)
        route = self.registry.current_route(
            operation.preparation.owner_id,
            operation.preparation.doc_id,
        )
        generation = self.generations.current(
            owner_id=operation.preparation.owner_id,
            doc_id=operation.preparation.doc_id,
        )
        if route is None or generation is None:
            raise RuntimeError("rollback route/generation disappeared.")
        self.population_journal.mark_rolled_back(
            self._population_identity(operation),
            row_digest=self._row_digest(operation),
            route_digest=self._route_digest(route),
            generation_sequence=generation.sequence,
            now=self._now(),
        )

    def reconcile_population(
        self,
        operation: CutoverOperation,
    ) -> TargetPopulationReconciliation:
        """Return a read-only reconciliation classification for one operation."""

        identity = self._population_identity(operation)
        record = self.population_journal.get(operation.operation_id)
        self._target = self._load_target(operation)
        rows = self._target_rows(operation)
        exact = bool(rows) and self._target_rows_match(operation)
        route = self.registry.current_route(
            operation.preparation.owner_id,
            operation.preparation.doc_id,
        )
        return reconcile_target_population(
            identity,
            record,
            observed_rows=len(rows),
            exact_population_match=exact,
            route_collection_id=None if route is None else route.collection_id,
        )

    def finalize_visible_recovery(
        self,
        operation: CutoverOperation,
    ) -> None:
        """Seal a missing visible receipt after a crash following route publication.

        This must be called inside ``exclusive_lock(operation)``. It performs the same
        target/generation/route/sparse validation as normal visible validation, then
        writes only the missing durable receipt; it does not republish any index data.
        """

        self._renew_execution()
        self._target = self._load_target(operation)
        publication = TargetPublication.expected(operation.preparation)
        super().validate_visible_target(operation, publication)
        route = self.registry.current_route(
            operation.preparation.owner_id,
            operation.preparation.doc_id,
        )
        generation = self.generations.current(
            owner_id=operation.preparation.owner_id,
            doc_id=operation.preparation.doc_id,
        )
        if route is None or generation is None:
            raise RuntimeError("visible recovery route/generation disappeared.")
        self.population_journal.mark_visible(
            self._population_identity(operation),
            row_digest=self._row_digest(operation),
            route_digest=self._route_digest(route),
            generation_sequence=generation.sequence,
            now=self._now(),
        )


__all__ = ["DurableBlueGreenCutoverBackendAdapter"]
