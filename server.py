"""Compatibility entrypoint over the FastAPI service implementation.

`server_app` preserves the complete route/application implementation. This shim
normalizes service-critical configuration before importing that implementation,
then replaces parser-facing ingestion and recovery so every unfinished operation is
replayed idempotently from immutable descriptor-anchored bytes.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path
from typing import Optional

import uvicorn


def _normalize_integer_env(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        value = default
    value = max(minimum, min(value, maximum))
    os.environ[name] = str(value)
    return value


def _normalize_float_env(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        value = default
    if not math.isfinite(value):
        value = default
    value = max(minimum, min(value, maximum))
    os.environ[name] = str(value)
    return value


def _normalize_service_environment() -> None:
    integer_specs = {
        "MAX_UPLOAD_BYTES": (50_000_000, 1, 1_000_000_000),
        "MAX_REMOTE_DOWNLOAD_BYTES": (5_000_000, 1, 1_000_000_000),
        "MAX_REMOTE_REDIRECTS": (4, 0, 20),
        "JOB_TTL_SECONDS": (86_400, 60, 31_536_000),
        "REQUESTS_PER_MINUTE": (60, 1, 1_000_000),
        "QUERY_WORKERS": (8, 1, 64),
        "QUERY_MAX_PENDING": (32, 1, 1000),
        "INGEST_WORKERS": (2, 1, 16),
        "INGEST_MAX_ATTEMPTS": (3, 1, 20),
        "INGEST_MAX_PENDING": (64, 1, 10_000),
        "MAX_TOOL_ARGUMENT_CHARS": (50_000, 1000, 500_000),
        "MAX_TOOL_RESULT_CHARS": (30_000, 1000, 200_000),
        "MAX_EVIDENCE_SOURCES": (100, 1, 500),
        "MAX_CONCURRENT_TOOL_WORKERS": (32, 1, 256),
        "MAX_PENDING_TOOL_TASKS": (64, 1, 4096),
        "MAX_RESPONSE_TOKENS": (4000, 256, 32_000),
        "MAX_CHUNKS_PER_DOCUMENT": (10_000, 100, 100_000),
        "DOCUMENT_LIST_SCAN_BATCH": (500, 50, 5000),
        "MAX_DOCUMENT_LIST_SCAN_CHUNKS": (100_000, 50, 1_000_000),
        "ORPHAN_GRACE_SECONDS": (3600, 60, 31_536_000),
        "VISUAL_MAX_PDF_PAGES": (500, 1, 5000),
        "VISUAL_MAX_RENDER_PIXELS": (2_000_000, 1_000_000, 100_000_000),
        "VISUAL_MAX_ENCODED_BYTES": (10_000_000, 100_000, 100_000_000),
        "OCR_MAX_PAGES": (50, 1, 500),
        "OCR_DPI": (200, 100, 400),
        "OCR_TIMEOUT_SECONDS": (30, 1, 300),
        "OCR_MIN_TEXT_CHARS": (40, 0, 2000),
        "MAX_PDF_PAGES": (2000, 1, 10_000),
        "MAX_PDF_RENDER_PIXELS": (40_000_000, 1_000_000, 250_000_000),
        "MAX_EXTRACTED_CHARS": (5_000_000, 100_000, 50_000_000),
        "MAX_DOCX_MEMBERS": (10_000, 10, 100_000),
        "MAX_DOCX_UNCOMPRESSED_BYTES": (200_000_000, 1, 2_000_000_000),
        "SERPER_MAX_RESPONSE_BYTES": (2_000_000, 10_000, 20_000_000),
        "WEB_SEARCH_MAX_RESULT_CANDIDATES": (30, 10, 100),
        "PORT": (8000, 1, 65_535),
    }
    for name, (default, minimum, maximum) in integer_specs.items():
        _normalize_integer_env(
            name,
            default,
            minimum=minimum,
            maximum=maximum,
        )
    float_specs = {
        "REMOTE_REQUEST_TIMEOUT_SECONDS": (15.0, 0.1, 300.0),
        "QUERY_TIMEOUT_SECONDS": (120.0, 1.0, 900.0),
        "INGEST_ADMISSION_RETRY_SECONDS": (1.0, 0.1, 60.0),
        "MAX_DOCX_COMPRESSION_RATIO": (1000.0, 10.0, 100_000.0),
    }
    for name, (default, minimum, maximum) in float_specs.items():
        _normalize_float_env(
            name,
            default,
            minimum=minimum,
            maximum=maximum,
        )
    upload_limit = int(os.environ["MAX_UPLOAD_BYTES"])
    request_default = min(upload_limit + 1_048_576, 1_000_000_000)
    _normalize_integer_env(
        "MAX_REQUEST_BODY_BYTES",
        request_default,
        minimum=upload_limit,
        maximum=1_000_000_000,
    )
    raw_upload_root = Path(os.getenv("UPLOAD_DIR", "uploads"))
    if raw_upload_root.is_symlink():
        raise RuntimeError("UPLOAD_DIR may not be a symbolic link.")


_normalize_service_environment()

if "server_app" in sys.modules:
    _implementation = importlib.reload(sys.modules["server_app"])
else:
    _implementation = importlib.import_module("server_app")

from tools.ingestion_snapshot import materialize_ingestion_snapshot
from tools.upload_storage import (
    UploadStorageError,
    read_owner_file,
    validated_owner_file_path,
)


def _validated_upload_file(path: str | Path | None) -> Optional[Path]:
    return validated_owner_file_path(_implementation.UPLOAD_DIR, path)


def _retry_or_fail_job(
    *,
    job_id: str,
    owner_id: str,
    display_name: str,
    path: Path,
    exc: BaseException,
    retry_prefix: str,
    failure_prefix: str,
) -> None:
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
                message=_implementation._internal_failure_message(retry_prefix, exc),
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
        _implementation._internal_failure_message(failure_prefix, exc),
    )


def _retry_or_fail_snapshot(
    *,
    job_id: str,
    owner_id: str,
    display_name: str,
    path: Path,
    exc: BaseException,
) -> None:
    _retry_or_fail_job(
        job_id=job_id,
        owner_id=owner_id,
        display_name=display_name,
        path=path,
        exc=exc,
        retry_prefix="Transient ingestion snapshot failure; retry queued",
        failure_prefix="Ingestion snapshot failed",
    )


def _recover_interrupted_jobs() -> None:
    """Replay every unfinished job from its durable source after restart."""

    for record in _implementation._JOB_STORE.recoverable():
        job_id = str(record["job_id"])
        owner_id = str(record["owner_id"])
        display_name = str(record.get("filename") or "upload")
        source_path = str(record.get("source_path") or "")
        attempts = int(record.get("attempts") or 0)
        if attempts >= _implementation.INGEST_MAX_ATTEMPTS:
            _implementation._persist_failed_job(
                job_id,
                owner_id,
                display_name,
                source_path,
                "Interrupted ingestion exhausted its retry limit.",
            )
            continue
        candidate = _implementation._validated_upload_file(source_path)
        if candidate is None:
            _implementation._persist_failed_job(
                job_id,
                owner_id,
                display_name,
                source_path,
                (
                    "Interrupted ingestion could not resume because its source file "
                    "was missing, invalid, symlinked, or outside UPLOAD_DIR."
                ),
            )
            continue
        try:
            _implementation._JOB_STORE.update(
                job_id,
                owner_id,
                status="queued",
                filename=display_name,
                source_path=str(candidate),
                message="Recovered after service restart.",
            )
        except Exception:
            continue
        _implementation._submit_ingestion(
            str(candidate),
            display_name,
            job_id,
            owner_id,
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
        with snapshot_context as (snapshot_path, snapshot_bytes):
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

            current_source_bytes = _implementation.read_owner_file(
                _implementation.UPLOAD_DIR,
                path,
                max_bytes=_implementation.DEFAULT_MAX_UPLOAD_BYTES,
            )
            if current_source_bytes is None or current_source_bytes != snapshot_bytes:
                raise UploadStorageError(
                    "The queued upload changed after its immutable snapshot was created."
                )

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
                )
                if registry_record is None:
                    raise RuntimeError("Document registry did not return the committed row.")
                registry_keeps_source = bool(registry_record.get("source_retained"))
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
                keep_source = registry_keeps_source
            except Exception as exc:
                keep_source = True
                _retry_or_fail_job(
                    job_id=job_id,
                    owner_id=owner_id,
                    display_name=display_name,
                    path=path,
                    exc=exc,
                    retry_prefix="Transient ingestion failure; retry queued",
                    failure_prefix="Ingestion failed",
                )
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
_implementation.read_owner_file = read_owner_file
_implementation._retry_or_fail_job = _retry_or_fail_job
_implementation._retry_or_fail_snapshot = _retry_or_fail_snapshot
_implementation._recover_interrupted_jobs = _recover_interrupted_jobs
_implementation.process_ingestion = process_ingestion
_implementation.__doc__ = __doc__

if __name__ == "__main__":
    uvicorn.run(
        _implementation.app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.environ["PORT"]),
        reload=False,
    )
else:
    sys.modules[__name__] = _implementation
