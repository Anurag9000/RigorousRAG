"""Compatibility entrypoint over the FastAPI service implementation.

`server_app` preserves the complete route/application implementation. This shim
replaces only parser-facing ingestion so queued uploads are consumed from immutable,
descriptor-anchored byte snapshots rather than reopened owner pathnames.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import uvicorn

import server_app as _implementation
from tools.ingestion_snapshot import materialize_ingestion_snapshot
from tools.upload_storage import UploadStorageError, validated_owner_file_path


def _validated_upload_file(path: str | Path | None) -> Optional[Path]:
    """Return one current regular owner file through anchored validation."""

    return validated_owner_file_path(_implementation.UPLOAD_DIR, path)


def _retry_or_fail_snapshot(
    *,
    job_id: str,
    owner_id: str,
    display_name: str,
    path: Path,
    exc: BaseException,
) -> None:
    """Return a claimed job to durable retry state or persist terminal failure."""

    internal = _implementation._JOB_STORE.get_internal(job_id, owner_id) or {}
    attempts = int(internal.get("attempts") or 0)
    if attempts < _implementation.INGEST_MAX_ATTEMPTS:
        try:
            _implementation._JOB_STORE.update(
                job_id,
                owner_id,
                status="queued",
                filename=display_name,
                source_path=str(path),
                message=_implementation._internal_failure_message(
                    "Transient ingestion snapshot failure; retry queued",
                    exc,
                ),
            )
            _implementation._submit_ingestion(
                str(path),
                display_name,
                job_id,
                owner_id,
            )
            return
        except Exception:
            return
    _implementation._persist_failed_job(
        job_id,
        owner_id,
        display_name,
        path,
        _implementation._internal_failure_message(
            "Ingestion snapshot failed",
            exc,
        ),
    )


def process_ingestion(
    file_path: str,
    display_name: str,
    job_id: str,
    owner_id: str,
) -> None:
    """Claim and process one job from an immutable anchored upload snapshot."""

    path = _implementation._validated_upload_file(file_path)
    if path is None:
        _implementation._persist_failed_job(
            job_id,
            owner_id,
            display_name,
            file_path,
            "The ingestion source was missing, invalid, symlinked, or outside UPLOAD_DIR.",
        )
        return
    if not _implementation._JOB_STORE.claim(
        job_id,
        owner_id,
        _implementation.INGEST_MAX_ATTEMPTS,
    ):
        return

    try:
        snapshot_context = _implementation.materialize_ingestion_snapshot(
            upload_root=_implementation.UPLOAD_DIR,
            source_path=path,
            max_bytes=_implementation.DEFAULT_MAX_UPLOAD_BYTES,
        )
        with snapshot_context as (snapshot_path, _snapshot_bytes):
            result = _implementation.ingest_file(
                str(snapshot_path),
                owner_id=owner_id,
            )
            if not result.success or result.document is None:
                _implementation._persist_failed_job(
                    job_id,
                    owner_id,
                    display_name,
                    path,
                    result.error or "Document ingestion failed.",
                )
                return

            document = result.document
            document.filename = display_name
            keep_source = True
            try:
                agent = _implementation._new_agent(owner_id)
                indexed = _implementation.index_document(
                    document,
                    owner_id=owner_id,
                    rag=_implementation.get_rag_layer(),
                    client=agent.client,
                    job_id=job_id,
                )
                _implementation._JOB_STORE.update(
                    job_id,
                    owner_id,
                    status="finalizing",
                    filename=display_name,
                    source_path=str(path),
                    message="Vector indexing completed; finalizing source lifecycle.",
                    doc_id=document.id,
                )
                retained_path = str(path) if _implementation.RETAIN_SOURCE_FILES else None
                previous_path = _implementation._DOCUMENT_STORE.register(
                    owner_id=owner_id,
                    doc_id=document.id,
                    filename=document.filename,
                    mime_type=document.mime_type,
                    source_path=retained_path,
                )
                registry_record = _implementation._DOCUMENT_STORE.get(
                    owner_id=owner_id,
                    doc_id=document.id,
                ) or {}
                keep_source = bool(registry_record.get("source_retained"))
                if previous_path:
                    _implementation._safe_unlink_upload(previous_path)
                _implementation._JOB_STORE.update(
                    job_id,
                    owner_id,
                    status="success",
                    filename=display_name,
                    source_path=str(registry_record.get("source_path") or ""),
                    message=f"Indexed {indexed.chunk_count} semantic chunks.",
                    doc_id=document.id,
                )
            except Exception as exc:
                registry_record = _implementation._DOCUMENT_STORE.get(
                    owner_id=owner_id,
                    doc_id=document.id,
                )
                if registry_record is not None:
                    retained_path = str(registry_record.get("source_path") or "")
                    try:
                        _implementation._JOB_STORE.update(
                            job_id,
                            owner_id,
                            status="success",
                            filename=display_name,
                            source_path=retained_path,
                            message="Recovered completed document finalization.",
                            doc_id=document.id,
                        )
                        keep_source = bool(registry_record.get("source_retained"))
                    except Exception:
                        keep_source = True
                else:
                    internal = _implementation._JOB_STORE.get_internal(
                        job_id,
                        owner_id,
                    ) or {}
                    attempts = int(internal.get("attempts") or 0)
                    if attempts < _implementation.INGEST_MAX_ATTEMPTS:
                        keep_source = True
                        try:
                            _implementation._JOB_STORE.update(
                                job_id,
                                owner_id,
                                status="queued",
                                filename=display_name,
                                source_path=str(path),
                                message=_implementation._internal_failure_message(
                                    "Transient ingestion failure; retry queued",
                                    exc,
                                ),
                                doc_id=document.id,
                            )
                            _implementation._submit_ingestion(
                                str(path),
                                display_name,
                                job_id,
                                owner_id,
                            )
                        except Exception:
                            pass
                    else:
                        if not _implementation._persist_failed_job(
                            job_id,
                            owner_id,
                            display_name,
                            path,
                            _implementation._internal_failure_message(
                                "Ingestion failed",
                                exc,
                            ),
                        ):
                            keep_source = True
            finally:
                if not keep_source:
                    _implementation._safe_unlink_upload(path)
    except UploadStorageError as exc:
        _implementation._persist_failed_job(
            job_id,
            owner_id,
            display_name,
            path,
            _implementation._internal_failure_message(
                "The ingestion source could not be snapshotted",
                exc,
            ),
        )
    except Exception as exc:
        _retry_or_fail_snapshot(
            job_id=job_id,
            owner_id=owner_id,
            display_name=display_name,
            path=path,
            exc=exc,
        )


_implementation._validated_upload_file = _validated_upload_file
_implementation.materialize_ingestion_snapshot = materialize_ingestion_snapshot
_implementation._retry_or_fail_snapshot = _retry_or_fail_snapshot
_implementation.process_ingestion = process_ingestion
_implementation.__doc__ = __doc__

if __name__ == "__main__":
    uvicorn.run(
        _implementation.app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
else:
    sys.modules[__name__] = _implementation
