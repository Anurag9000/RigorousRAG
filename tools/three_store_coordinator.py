"""Authoritative vector+sparse+manifest transaction and reconciliation layer."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from tools.generation_store import GenerationRecord, GenerationStore
from tools.index_coordinator import (
    CrossStoreSnapshot,
    IndexCoordinationError,
    IndexCoordinator,
    _document_lock,
)
from tools.sparse_index import SparseField
from tools.vector_generation import restore_vector_generation

try:
    from tools.security import normalize_owner_id
except ImportError:  # focused-test fallback
    def normalize_owner_id(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("owner_id is invalid.")
        return value.strip()

_MAX_SCAN_DOCUMENTS = 10_000


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned)
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


class AuthoritativeCoordinationError(RuntimeError):
    """A three-store commit failed; rollback status is explicit and bounded."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        rollback_errors: Sequence[str] = (),
    ) -> None:
        self.phase = phase
        self.rollback_errors = tuple(rollback_errors)[:20]
        suffix = (
            " Rollback errors: " + ", ".join(self.rollback_errors)
            if self.rollback_errors
            else ""
        )
        super().__init__(f"{message} [phase={phase}].{suffix}")


@dataclass(frozen=True)
class ReconciliationReport:
    owner_id: str
    healthy: tuple[str, ...]
    vector_only: tuple[str, ...]
    sparse_only: tuple[str, ...]
    store_pair_without_manifest: tuple[str, ...]
    manifest_without_store_pair: tuple[str, ...]
    deleted_but_present: tuple[str, ...]
    metadata_mismatch: tuple[str, ...]
    inspection_failed: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not any(
            (
                self.vector_only,
                self.sparse_only,
                self.store_pair_without_manifest,
                self.manifest_without_store_pair,
                self.deleted_but_present,
                self.metadata_mismatch,
                self.inspection_failed,
            )
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "owner_id": self.owner_id,
            "clean": self.clean,
            "healthy": self.healthy,
            "vector_only": self.vector_only,
            "sparse_only": self.sparse_only,
            "store_pair_without_manifest": self.store_pair_without_manifest,
            "manifest_without_store_pair": self.manifest_without_store_pair,
            "deleted_but_present": self.deleted_but_present,
            "metadata_mismatch": self.metadata_mismatch,
            "inspection_failed": self.inspection_failed,
        }


class AuthoritativeIndexCoordinator:
    """Commit vector, sparse, and durable current state as one operation."""

    def __init__(self, *, index: IndexCoordinator, generations: GenerationStore) -> None:
        if not isinstance(index, IndexCoordinator):
            raise ValueError("index must be an IndexCoordinator.")
        if not isinstance(generations, GenerationStore):
            raise ValueError("generations must be a GenerationStore.")
        self.index = index
        self.generations = generations

    def _restore_index(
        self,
        *,
        owner_id: str,
        doc_id: str,
        prior: CrossStoreSnapshot,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        try:
            restore_vector_generation(
                self.index.rag,
                owner_id=owner_id,
                doc_id=doc_id,
                snapshot=prior.vector,
            )
        except Exception as exc:
            errors.append(f"vector:{type(exc).__name__}")
        try:
            self.index.sparse.restore_document(
                owner_id=owner_id,
                doc_id=doc_id,
                snapshot=prior.sparse,
            )
        except Exception as exc:
            errors.append(f"sparse:{type(exc).__name__}")
        return tuple(errors)

    def _restore_generation(
        self,
        *,
        owner_id: str,
        doc_id: str,
        prior: GenerationRecord | None,
    ) -> tuple[str, ...]:
        try:
            current = self.generations.current(
                owner_id=owner_id,
                doc_id=doc_id,
            )
            current_sequence = 0 if current is None else current.sequence
            prior_sequence = 0 if prior is None else prior.sequence
            if current_sequence == prior_sequence:
                return ()
            self.generations.restore_current(
                prior,
                owner_id=owner_id,
                doc_id=doc_id,
                expected_sequence=current_sequence,
                reason="three_store_compensation",
            )
            return ()
        except Exception as exc:
            return (f"generation:{type(exc).__name__}",)

    def replace_document(
        self,
        *,
        owner_id: str,
        doc_id: str,
        text: str,
        sections: Iterable[Any] | None,
        metadata: Mapping[str, Any],
        sparse_fields: Iterable[SparseField],
        content_sha256: str,
        profile_fingerprint: str,
        chunk_size: int = 1_000,
        overlap: int = 120,
        expected_manifest_sequence: int | None = None,
        expected_sparse_generation: int | None = None,
        audit_metadata: Mapping[str, Any] | None = None,
    ) -> GenerationRecord:
        owner = normalize_owner_id(owner_id)
        document_id = _identifier(doc_id, "doc_id")
        if expected_manifest_sequence is not None:
            if (
                isinstance(expected_manifest_sequence, bool)
                or not isinstance(expected_manifest_sequence, int)
                or expected_manifest_sequence < 0
            ):
                raise ValueError(
                    "expected_manifest_sequence must be a non-negative integer."
                )
        lock = _document_lock(owner, document_id)
        with lock:
            prior_index = self.index.snapshot(
                owner_id=owner,
                doc_id=document_id,
            )
            prior_generation = self.generations.current(
                owner_id=owner,
                doc_id=document_id,
            )
            prior_sequence = (
                0 if prior_generation is None else prior_generation.sequence
            )
            if (
                expected_manifest_sequence is not None
                and expected_manifest_sequence != prior_sequence
            ):
                raise RuntimeError("manifest sequence changed concurrently.")
            try:
                manifest = self.index.replace_document(
                    owner_id=owner,
                    doc_id=document_id,
                    text=text,
                    sections=sections,
                    metadata=metadata,
                    sparse_fields=sparse_fields,
                    content_sha256=content_sha256,
                    profile_fingerprint=profile_fingerprint,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    expected_sparse_generation=expected_sparse_generation,
                )
            except IndexCoordinationError:
                raise
            except Exception as exc:
                raise AuthoritativeCoordinationError(
                    "Cross-store index replacement failed",
                    phase="index",
                ) from exc
            try:
                return self.generations.record_active(
                    manifest,
                    expected_sequence=prior_sequence,
                    metadata=audit_metadata,
                )
            except Exception as exc:
                rollback_errors = list(
                    self._restore_index(
                        owner_id=owner,
                        doc_id=document_id,
                        prior=prior_index,
                    )
                )
                rollback_errors.extend(
                    self._restore_generation(
                        owner_id=owner,
                        doc_id=document_id,
                        prior=prior_generation,
                    )
                )
                raise AuthoritativeCoordinationError(
                    f"Generation manifest commit failed ({type(exc).__name__})",
                    phase="manifest",
                    rollback_errors=rollback_errors,
                ) from exc

    def delete_document(
        self,
        *,
        owner_id: str,
        doc_id: str,
        audit_metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        owner = normalize_owner_id(owner_id)
        document_id = _identifier(doc_id, "doc_id")
        lock = _document_lock(owner, document_id)
        with lock:
            prior_index = self.index.snapshot(
                owner_id=owner,
                doc_id=document_id,
            )
            prior_generation = self.generations.current(
                owner_id=owner,
                doc_id=document_id,
            )
            stores_exist = (
                prior_index.vector.row_count > 0
                or prior_index.sparse is not None
            )
            manifest_exists = (
                prior_generation is not None
                and prior_generation.state != "deleted"
            )
            if not stores_exist and not manifest_exists:
                return False
            try:
                if stores_exist:
                    self.index.delete_document(
                        owner_id=owner,
                        doc_id=document_id,
                    )
            except Exception as exc:
                raise AuthoritativeCoordinationError(
                    f"Cross-store deletion failed ({type(exc).__name__})",
                    phase="index_delete",
                ) from exc
            try:
                prior_sequence = (
                    0 if prior_generation is None else prior_generation.sequence
                )
                if prior_generation is None:
                    return True
                self.generations.record_deleted(
                    owner_id=owner,
                    doc_id=document_id,
                    expected_sequence=prior_sequence,
                    prior=prior_generation,
                    metadata=audit_metadata,
                )
                return True
            except Exception as exc:
                rollback_errors = list(
                    self._restore_index(
                        owner_id=owner,
                        doc_id=document_id,
                        prior=prior_index,
                    )
                )
                rollback_errors.extend(
                    self._restore_generation(
                        owner_id=owner,
                        doc_id=document_id,
                        prior=prior_generation,
                    )
                )
                raise AuthoritativeCoordinationError(
                    f"Generation deletion record failed ({type(exc).__name__})",
                    phase="manifest_delete",
                    rollback_errors=rollback_errors,
                ) from exc

    @staticmethod
    def _vector_metadata(snapshot: CrossStoreSnapshot) -> Mapping[str, Any]:
        metadatas = getattr(snapshot.vector, "metadatas", ())
        if not metadatas:
            return {}
        first = metadatas[0]
        return first if isinstance(first, Mapping) else {}

    def _manifest_matches(
        self,
        record: GenerationRecord,
        snapshot: CrossStoreSnapshot,
    ) -> bool:
        if (
            snapshot.vector.row_count != record.vector_rows
            or snapshot.sparse is None
        ):
            return False
        sparse_generation = getattr(snapshot.sparse, "generation", None)
        sparse_profile = getattr(snapshot.sparse, "profile_fingerprint", None)
        if (
            sparse_generation != record.sparse_generation
            or sparse_profile != record.profile_fingerprint
        ):
            return False
        vector_metadata = self._vector_metadata(snapshot)
        return (
            vector_metadata.get("content_sha256") == record.content_sha256
            and vector_metadata.get("embedding_profile_fingerprint")
            == record.profile_fingerprint
        )

    def reconcile_owner(self, *, owner_id: str) -> ReconciliationReport:
        owner = normalize_owner_id(owner_id)
        store_scan = self.index.scan_owner(owner_id=owner)
        vector_only = set(store_scan.get("vector_only", ()))
        sparse_only = set(store_scan.get("sparse_only", ()))
        aligned = set(store_scan.get("aligned", ()))
        records = self.generations.list_current(
            owner_id=owner,
            limit=_MAX_SCAN_DOCUMENTS,
        )
        manifest_map = {record.doc_id: record for record in records}
        active = {
            doc_id
            for doc_id, record in manifest_map.items()
            if record.state in {"active", "restored"}
            and record.vector_rows > 0
        }
        deleted = {
            doc_id
            for doc_id, record in manifest_map.items()
            if record.state == "deleted"
        }
        healthy: set[str] = set()
        mismatch: set[str] = set()
        inspection_failed: set[str] = set()
        for doc_id in itertools.islice(
            sorted(aligned & active),
            _MAX_SCAN_DOCUMENTS,
        ):
            try:
                snapshot = self.index.snapshot(
                    owner_id=owner,
                    doc_id=doc_id,
                )
                if self._manifest_matches(manifest_map[doc_id], snapshot):
                    healthy.add(doc_id)
                else:
                    mismatch.add(doc_id)
            except Exception:
                inspection_failed.add(doc_id)
        return ReconciliationReport(
            owner_id=owner,
            healthy=tuple(sorted(healthy)),
            vector_only=tuple(sorted(vector_only)),
            sparse_only=tuple(sorted(sparse_only)),
            store_pair_without_manifest=tuple(
                sorted(aligned - active - deleted)
            ),
            manifest_without_store_pair=tuple(sorted(active - aligned)),
            deleted_but_present=tuple(
                sorted(deleted & (aligned | vector_only | sparse_only))
            ),
            metadata_mismatch=tuple(sorted(mismatch)),
            inspection_failed=tuple(sorted(inspection_failed)),
        )


__all__ = [
    "AuthoritativeCoordinationError",
    "AuthoritativeIndexCoordinator",
    "ReconciliationReport",
]
