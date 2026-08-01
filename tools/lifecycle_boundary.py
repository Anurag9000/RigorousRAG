"""Lazy authoritative-index and retrieval boundary for fourth-store lifecycle replay."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from types import ModuleType
from typing import Any

from tools.lifecycle_outbox import operation_id_for
from tools.lifecycle_reconciliation import (
    get_cleanup_journal,
    reconcile_lifecycle_operation,
)
from tools.lifecycle_runtime import (
    get_lifecycle_outbox,
    reconcile_lifecycle_before_retrieval,
    remove_source_idempotently,
)
from tools.security import normalize_owner_id


def _retention_enabled() -> bool:
    return os.getenv(
        "RETAIN_SOURCE_FILES",
        os.getenv("RETAIN_UPLOADS", "true"),
    ).lower() in {"1", "true", "yes"}


def _job_id(metadata: Mapping[str, Any], audit_metadata: Any) -> str | None:
    values = []
    if isinstance(audit_metadata, Mapping):
        values.append(audit_metadata.get("job_id"))
    values.append(metadata.get("job_id"))
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()[:200]
    return None


def _idempotency_key(
    *,
    job_id: str | None,
    source_path: str | None,
    metadata: Mapping[str, Any],
) -> str:
    if job_id is not None:
        return f"job:{job_id}"
    if source_path is not None:
        return "source:" + hashlib.sha256(source_path.encode("utf-8")).hexdigest()
    fallback = (
        str(metadata.get("filename") or "")
        + "\0"
        + str(metadata.get("mime_type") or "")
    )
    return "metadata:" + hashlib.sha256(fallback.encode("utf-8")).hexdigest()


def _document_store() -> Any:
    from tools.document_store import get_document_store

    return get_document_store()


def _source_context(
    *,
    owner_id: str,
    document: Any,
    metadata: Mapping[str, Any],
    audit_metadata: Any,
) -> tuple[Any, str | None, bool, str | None]:
    from tools.job_store import JobStore
    from tools.upload_storage import validated_owner_file_path

    registry = _document_store()
    retain = _retention_enabled()
    identifier = _job_id(metadata, audit_metadata)
    source_path: str | None = None
    if identifier is not None:
        try:
            record = JobStore().get_internal(identifier, owner_id)
        except Exception as exc:
            if retain:
                raise RuntimeError(
                    "lifecycle source intent could not read the durable job."
                ) from exc
            record = None
        raw = str((record or {}).get("source_path") or "")
        if raw:
            candidate = validated_owner_file_path(
                registry.upload_root,
                raw,
            )
            if candidate is None and retain:
                raise RuntimeError(
                    "lifecycle source intent could not validate the durable job source."
                )
            source_path = str(candidate) if candidate is not None else None
        elif retain:
            raise RuntimeError(
                "lifecycle source intent is missing the durable job source."
            )
    elif retain:
        raw = getattr(document, "file_path", None)
        candidate = validated_owner_file_path(
            registry.upload_root,
            raw,
        )
        source_path = str(candidate) if candidate is not None else None
        retain = source_path is not None
    return registry, source_path, retain, identifier


def _synthesized_result(module: ModuleType, document: Any, generation: Any) -> Any:
    fields = module.build_sparse_fields(document, doc_id=getattr(document, "id"))
    return module.AuthoritativeIndexResult(generation, len(fields))


def install_authoritative_lifecycle_boundary(module: ModuleType) -> None:
    if not hasattr(module, "_lifecycle_original_commit_finalized_document"):
        module._lifecycle_original_commit_finalized_document = (
            module.commit_finalized_document
        )
    if not hasattr(module, "_lifecycle_original_delete_authoritative_document"):
        module._lifecycle_original_delete_authoritative_document = (
            module.delete_authoritative_document
        )
    original_commit = module._lifecycle_original_commit_finalized_document
    original_delete = module._lifecycle_original_delete_authoritative_document

    def commit_finalized_document(
        document: Any,
        *,
        owner_id: str,
        rag: Any,
        metadata: Mapping[str, Any],
        coordinator: Any = None,
        profile_name: str | None = None,
        chunk_size: int = 1_000,
        overlap: int = 120,
        audit_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        owner = normalize_owner_id(owner_id)
        doc_id = module._identifier(getattr(document, "id", None), "document.id")
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping.")
        content_hash = metadata.get("content_sha256")
        if content_hash is None:
            text = module._text(getattr(document, "text", None), "document.text")
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        registry, source_path, retain, job_id = _source_context(
            owner_id=owner,
            document=document,
            metadata=metadata,
            audit_metadata=audit_metadata,
        )
        key = _idempotency_key(
            job_id=job_id,
            source_path=source_path,
            metadata=metadata,
        )
        operation_id = operation_id_for(
            kind="replace",
            owner_id=owner,
            doc_id=doc_id,
            content_sha256=str(content_hash),
            idempotency_key=key,
        )
        outbox = get_lifecycle_outbox()
        cleanup = get_cleanup_journal()
        operation = outbox.plan_replace(
            operation_id=operation_id,
            owner_id=owner,
            doc_id=doc_id,
            content_sha256=str(content_hash),
            filename=str(
                metadata.get("filename")
                or getattr(document, "filename", "document")
            ),
            mime_type=str(
                metadata.get("mime_type")
                or getattr(document, "mime_type", "application/octet-stream")
            ),
            source_path=source_path,
            retain_source=retain,
        )
        generations = (
            coordinator.generations
            if coordinator is not None
            else module.get_authoritative_index_coordinator(rag=rag).generations
        )
        lock = module._document_lock(owner, doc_id)
        with lock:
            current = generations.current(owner_id=owner, doc_id=doc_id)
            if operation.state != "planned":
                result = reconcile_lifecycle_operation(
                    operation_id,
                    outbox=outbox,
                    generations=generations,
                    registry=registry,
                    cleanup=cleanup,
                    remove_source=lambda value: remove_source_idempotently(
                        registry, value
                    ),
                )
                current = generations.current(owner_id=owner, doc_id=doc_id)
                if result.state == "completed" and bool(
                    current is not None
                    and current.state in {"active", "restored"}
                    and current.content_sha256 == content_hash
                ):
                    return _synthesized_result(module, document, current)
                raise RuntimeError(
                    "lifecycle replacement is pending or was superseded."
                )
            if bool(
                current is not None
                and current.state in {"active", "restored"}
                and current.content_sha256 == content_hash
            ):
                outbox.mark_index_committed(
                    operation_id,
                    generation_sequence=current.sequence,
                )
                reconciled = reconcile_lifecycle_operation(
                    operation_id,
                    outbox=outbox,
                    generations=generations,
                    registry=registry,
                    cleanup=cleanup,
                    remove_source=lambda value: remove_source_idempotently(
                        registry, value
                    ),
                )
                if reconciled.state != "completed":
                    raise RuntimeError(
                        "lifecycle registry finalization is incomplete."
                    )
                return _synthesized_result(module, document, current)
            committed = original_commit(
                document,
                owner_id=owner,
                rag=rag,
                metadata=metadata,
                coordinator=coordinator,
                profile_name=profile_name,
                chunk_size=chunk_size,
                overlap=overlap,
                audit_metadata=audit_metadata,
            )
            outbox.mark_index_committed(
                operation_id,
                generation_sequence=committed.generation.sequence,
            )
            reconciled = reconcile_lifecycle_operation(
                operation_id,
                outbox=outbox,
                generations=generations,
                registry=registry,
                cleanup=cleanup,
                remove_source=lambda value: remove_source_idempotently(
                    registry, value
                ),
            )
            if reconciled.state != "completed":
                raise RuntimeError("lifecycle registry finalization is incomplete.")
            return committed

    def delete_authoritative_document(
        *,
        owner_id: str,
        doc_id: str,
        rag: Any,
        coordinator: Any = None,
        audit_metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        owner = normalize_owner_id(owner_id)
        document = module._identifier(doc_id, "doc_id")
        selected = coordinator or module.get_authoritative_index_coordinator(rag=rag)
        generations = selected.generations
        registry = _document_store()
        lock = module._document_lock(owner, document)
        with lock:
            current = generations.current(owner_id=owner, doc_id=document)
            registry_record = registry.get(owner_id=owner, doc_id=document)
            outbox = get_lifecycle_outbox()
            matching = tuple(
                item
                for item in outbox.list_pending(owner_id=owner, limit=10_000)
                if item.kind == "delete" and item.doc_id == document
            )
            if len(matching) > 1:
                raise RuntimeError(
                    "multiple pending lifecycle deletions exist for one document."
                )
            if matching:
                operation_id = matching[0].operation_id
            else:
                sequence = getattr(current, "sequence", 0)
                updated = (registry_record or {}).get("updated_at", 0)
                operation_id = operation_id_for(
                    kind="delete",
                    owner_id=owner,
                    doc_id=document,
                    idempotency_key=f"{sequence}:{updated}",
                )
            cleanup = get_cleanup_journal()
            operation = outbox.plan_delete(
                operation_id=operation_id,
                owner_id=owner,
                doc_id=document,
            )
            if (
                operation.state != "planned"
                or current is None
                or current.state == "deleted"
            ):
                reconciled = reconcile_lifecycle_operation(
                    operation_id,
                    outbox=outbox,
                    generations=generations,
                    registry=registry,
                    cleanup=cleanup,
                    remove_source=lambda value: remove_source_idempotently(
                        registry, value
                    ),
                )
                if reconciled.state == "completed":
                    return bool(registry_record is not None or current is not None)
                if operation.state != "planned":
                    raise RuntimeError(
                        "lifecycle deletion is pending or was superseded."
                    )
            deleted = original_delete(
                owner_id=owner,
                doc_id=document,
                rag=rag,
                coordinator=selected,
                audit_metadata=audit_metadata,
            )
            current = generations.current(owner_id=owner, doc_id=document)
            sequence = (
                int(getattr(current, "sequence", 0))
                if current is not None
                else 0
            )
            outbox.mark_index_committed(
                operation_id,
                generation_sequence=sequence,
            )
            reconciled = reconcile_lifecycle_operation(
                operation_id,
                outbox=outbox,
                generations=generations,
                registry=registry,
                cleanup=cleanup,
                remove_source=lambda value: remove_source_idempotently(
                    registry, value
                ),
            )
            if reconciled.state != "completed":
                raise RuntimeError("lifecycle registry deletion is incomplete.")
            return bool(deleted or registry_record is not None)

    commit_finalized_document._rigorousrag_lifecycle_boundary = True
    delete_authoritative_document._rigorousrag_lifecycle_boundary = True
    module.commit_finalized_document = commit_finalized_document
    module.delete_authoritative_document = delete_authoritative_document


def install_rag_lifecycle_boundary(module: ModuleType) -> None:
    if not hasattr(module, "_lifecycle_original_get_rag_layer"):
        module._lifecycle_original_get_rag_layer = module.get_rag_layer
    original = module._lifecycle_original_get_rag_layer

    def get_rag_layer(*args: Any, **kwargs: Any) -> Any:
        reconcile_lifecycle_before_retrieval()
        return original(*args, **kwargs)

    get_rag_layer._rigorousrag_lifecycle_boundary = True
    module.get_rag_layer = get_rag_layer


__all__ = [
    "install_authoritative_lifecycle_boundary",
    "install_rag_lifecycle_boundary",
]
