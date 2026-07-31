"""Validated compatibility entrypoint over the FastAPI service implementation.

``server_app`` owns route declarations and response schemas. This module validates
configuration before importing it, then installs strict runtime helpers for model
selection, identifiers, upload scheduling, replay-only durable recovery, immutable
ingestion, document deletion, and public error translation.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import stat
import sys
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Mapping, Optional

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse


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


def _safe_state_path(name: str, default: str) -> Path:
    raw = os.getenv(name, default)
    if (
        not isinstance(raw, str)
        or not raw
        or len(raw) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise RuntimeError(f"{name} is invalid or too long.")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"{name} could not be inspected safely.") from exc
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or bool(
            attributes & _WINDOWS_REPARSE_POINT
        ):
            raise RuntimeError(
                f"{name} may not contain symbolic-link components or reparse points."
            )
    os.environ[name] = str(absolute)
    return absolute


def _normalize_service_environment() -> None:
    integer_specs = {
        "MAX_UPLOAD_BYTES": (50_000_000, 1, 1_000_000_000),
        "MAX_REMOTE_DOWNLOAD_BYTES": (5_000_000, 1, 1_000_000_000),
        "MAX_REMOTE_REDIRECTS": (4, 0, 20),
        "MAX_REMOTE_REQUEST_BODY_BYTES": (1_000_000, 1, 20_000_000),
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
        "MAX_RESPONSE_TOKENS": (2000, 128, 16_000),
        "MAX_CHUNKS_PER_DOCUMENT": (10_000, 100, 100_000),
        "DOCUMENT_LIST_SCAN_BATCH": (500, 50, 5000),
        "MAX_DOCUMENT_LIST_SCAN_CHUNKS": (100_000, 50, 1_000_000),
        "MAX_VECTOR_METADATA_ITEMS": (200, 10, 2000),
        "MAX_SECTIONS_PER_DOCUMENT": (10_000, 1, 100_000),
        "MAX_RAG_QUERY_CHARS": (20_000, 1000, 100_000),
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
    normalized = {
        name: _normalize_integer_env(
            name,
            default,
            minimum=minimum,
            maximum=maximum,
        )
        for name, (default, minimum, maximum) in integer_specs.items()
    }
    for name, (default, minimum, maximum) in {
        "REMOTE_REQUEST_TIMEOUT_SECONDS": (15.0, 0.1, 300.0),
        "QUERY_TIMEOUT_SECONDS": (120.0, 1.0, 900.0),
        "INGEST_ADMISSION_RETRY_SECONDS": (1.0, 0.1, 60.0),
        "MAX_DOCX_COMPRESSION_RATIO": (1000.0, 10.0, 100_000.0),
    }.items():
        _normalize_float_env(
            name,
            default,
            minimum=minimum,
            maximum=maximum,
        )

    os.environ["QUERY_MAX_PENDING"] = str(
        max(normalized["QUERY_MAX_PENDING"], normalized["QUERY_WORKERS"])
    )
    os.environ["INGEST_MAX_PENDING"] = str(
        max(normalized["INGEST_MAX_PENDING"], normalized["INGEST_WORKERS"])
    )
    os.environ["MAX_PENDING_TOOL_TASKS"] = str(
        max(
            normalized["MAX_PENDING_TOOL_TASKS"],
            normalized["MAX_CONCURRENT_TOOL_WORKERS"],
        )
    )
    upload_limit = normalized["MAX_UPLOAD_BYTES"]
    _normalize_integer_env(
        "MAX_REQUEST_BODY_BYTES",
        min(upload_limit + 1_048_576, 1_000_000_000),
        minimum=upload_limit,
        maximum=1_000_000_000,
    )
    for name, default in (
        ("UPLOAD_DIR", "uploads"),
        ("JOB_DB_PATH", "data/jobs.sqlite3"),
        ("DOCUMENT_DB_PATH", "data/documents.sqlite3"),
        ("CHROMA_PATH", "rag_storage"),
        ("CLASSIC_STORAGE_DIR", "data"),
    ):
        _safe_state_path(name, default)


_normalize_service_environment()

if "server_app" in sys.modules:
    _implementation = importlib.reload(sys.modules["server_app"])
else:
    _implementation = importlib.import_module("server_app")

from tools.ingestion_snapshot import materialize_ingestion_snapshot
from tools.security import SecurityError, normalize_owner_id
from tools.upload_storage import (
    UploadStorageError,
    read_owner_file,
    validated_owner_file_path,
)


def _required_identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    bounded = value.strip()
    if (
        not bounded
        or len(bounded) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in bounded)
    ):
        raise ValueError(
            f"{label} must contain between 1 and {maximum} valid characters."
        )
    return bounded


def _bounded_identifier(value: Any, label: str, max_length: int = 200) -> str:
    try:
        return _required_identifier(value, label, max_length)
    except ValueError as exc:
        raise _implementation.HTTPException(
            status_code=400,
            detail=(
                f"{label} must contain between 1 and {max_length} valid characters."
            ),
        ) from exc


def _safe_request_id(raw_value: Any) -> str:
    if not isinstance(raw_value, str):
        return _implementation.uuid.uuid4().hex
    candidate = raw_value.strip()
    return (
        candidate
        if _implementation._REQUEST_ID_RE.fullmatch(candidate)
        else _implementation.uuid.uuid4().hex
    )


def _model_name(value: Any) -> str:
    return _required_identifier(value, "model", 200)


def _new_agent(owner_id: str, model: Optional[str] = None):
    owner = normalize_owner_id(owner_id)
    selected = _model_name(
        _implementation._DEFAULT_MODEL if model is None else model
    )
    if selected not in _implementation._ALLOWED_MODELS:
        raise _implementation.HTTPException(
            status_code=400,
            detail="The requested model is not enabled by the server.",
        )
    return _implementation.SearchAgent(
        model=selected,
        owner_id=owner,
        api_key=_implementation._PROVIDER_KEY,
        base_url=_implementation._BASE_URL,
        request_timeout=min(_implementation.QUERY_TIMEOUT_SECONDS, 300.0),
    )


def _validated_upload_file(path: str | Path | None) -> Optional[Path]:
    return validated_owner_file_path(_implementation.UPLOAD_DIR, path)


def _safe_attempt_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, min(int(value), 1_000_000))
    except (TypeError, ValueError, OverflowError):
        return 0


def _safe_due_at(value: Any, now: float) -> float:
    try:
        deadline = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(deadline) or deadline < 0:
        return 0.0
    maximum = now + max(
        float(_implementation._JOB_STORE.retry_max_seconds),
        604_800.0,
    )
    return min(deadline, maximum)


def _safe_wall_time() -> float:
    try:
        current = float(time.time())
    except (TypeError, ValueError, OverflowError):
        current = 0.0
    return current if math.isfinite(current) and current >= 0 else 0.0


def _forget_future(future: Future[Any]) -> None:
    release = False
    with _implementation._INGEST_FUTURES_LOCK:
        if future in _implementation._INGEST_FUTURES:
            _implementation._INGEST_FUTURES.remove(future)
            release = True
    if release:
        try:
            _implementation._INGEST_ADMISSION.release()
        except ValueError:
            pass


def _release_scheduled_ingestion(
    file_path: str,
    display_name: str,
    job_id: str,
    owner_id: str,
) -> None:
    if not _implementation._INGEST_SHUTDOWN.is_set():
        _submit_ingestion(file_path, display_name, job_id, owner_id)


def _schedule_ingestion_attempt(
    file_path: str,
    display_name: str,
    job_id: str,
    owner_id: str,
    due_at: float,
) -> None:
    if _implementation._INGEST_SHUTDOWN.is_set():
        return
    identifier = _required_identifier(job_id, "job_id")
    owner = normalize_owner_id(owner_id)
    deadline = _safe_due_at(due_at, _safe_wall_time())
    _implementation._INGEST_SCHEDULER.schedule(
        identifier,
        deadline,
        _release_scheduled_ingestion,
        str(file_path)[:4096],
        str(display_name)[:500],
        identifier,
        owner,
    )


def _submit_ingestion(
    file_path: str,
    display_name: str,
    job_id: str,
    owner_id: str,
) -> None:
    """Schedule a delayed job or admit one due job without leaking semaphore slots."""

    if _implementation._INGEST_SHUTDOWN.is_set():
        return
    try:
        identifier = _required_identifier(job_id, "job_id")
        owner = normalize_owner_id(owner_id)
        internal = _implementation._JOB_STORE.get_internal(identifier, owner)
    except Exception:
        return
    if not internal or internal.get("status") != "queued":
        try:
            _implementation._INGEST_SCHEDULER.cancel(identifier)
        except Exception:
            pass
        return

    now = _safe_wall_time()
    due_at = _safe_due_at(internal.get("next_attempt_at"), now)
    if due_at > now:
        try:
            _schedule_ingestion_attempt(
                file_path,
                display_name,
                identifier,
                owner,
                due_at,
            )
        except Exception:
            pass
        return

    try:
        _implementation._INGEST_SCHEDULER.cancel(identifier)
    except Exception:
        pass
    if not _implementation._INGEST_ADMISSION.acquire(blocking=False):
        try:
            _schedule_ingestion_attempt(
                file_path,
                display_name,
                identifier,
                owner,
                now + _implementation.INGEST_ADMISSION_RETRY_SECONDS,
            )
        except Exception:
            pass
        return

    try:
        future = _implementation._INGEST_EXECUTOR.submit(
            process_ingestion,
            str(file_path)[:4096],
            str(display_name)[:500],
            identifier,
            owner,
        )
    except Exception:
        _implementation._INGEST_ADMISSION.release()
        try:
            _schedule_ingestion_attempt(
                file_path,
                display_name,
                identifier,
                owner,
                _safe_wall_time()
                + _implementation.INGEST_ADMISSION_RETRY_SECONDS,
            )
        except Exception:
            pass
        return
    with _implementation._INGEST_FUTURES_LOCK:
        _implementation._INGEST_FUTURES.add(future)
    try:
        future.add_done_callback(_forget_future)
    except Exception:
        future.cancel()
        _forget_future(future)
        try:
            _schedule_ingestion_attempt(
                file_path,
                display_name,
                identifier,
                owner,
                _safe_wall_time()
                + _implementation.INGEST_ADMISSION_RETRY_SECONDS,
            )
        except Exception:
            pass


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
    attempts = _safe_attempt_count(internal.get("attempts"))
    if attempts < _implementation.INGEST_MAX_ATTEMPTS:
        try:
            _implementation._JOB_STORE.update(
                job_id,
                owner_id,
                status="queued",
                filename=display_name,
                source_path=str(path),
                message=_implementation._internal_failure_message(
                    retry_prefix,
                    exc,
                ),
            )
            _submit_ingestion(str(path), display_name, job_id, owner_id)
            return
        except Exception:
            try:
                _schedule_ingestion_attempt(
                    str(path),
                    display_name,
                    job_id,
                    owner_id,
                    _safe_wall_time()
                    + _implementation.INGEST_ADMISSION_RETRY_SECONDS,
                )
            except Exception:
                pass
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
    """Replay every unfinished job; registry existence alone is not a commit token."""

    try:
        records = _implementation._JOB_STORE.recoverable()
    except Exception:
        return
    if not isinstance(records, list):
        return
    for record in records[:100_000]:
        if not isinstance(record, Mapping):
            continue
        try:
            job_id = _required_identifier(record.get("job_id"), "job_id")
            owner_id = normalize_owner_id(record.get("owner_id"))
        except Exception:
            continue
        display_name = str(record.get("filename") or "upload")[:500]
        source_path = str(record.get("source_path") or "")[:4096]
        attempts = _safe_attempt_count(record.get("attempts"))
        if attempts >= _implementation.INGEST_MAX_ATTEMPTS:
            _implementation._persist_failed_job(
                job_id,
                owner_id,
                display_name,
                source_path,
                "Interrupted ingestion exhausted its retry limit.",
            )
            continue
        candidate = _validated_upload_file(source_path)
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
        _submit_ingestion(str(candidate), display_name, job_id, owner_id)


def _same_retained_source(previous_path: Any, current_path: Path) -> bool:
    if previous_path in (None, ""):
        return False
    previous = _validated_upload_file(previous_path)
    return previous is not None and previous == current_path


def process_ingestion(
    file_path: str,
    display_name: str,
    job_id: str,
    owner_id: str,
) -> None:
    """Claim and process one job from an immutable anchored upload snapshot."""

    path = _validated_upload_file(file_path)
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
        with materialize_ingestion_snapshot(
            upload_root=_implementation.UPLOAD_DIR,
            source_path=path,
            max_bytes=_implementation.DEFAULT_MAX_UPLOAD_BYTES,
        ) as (snapshot_path, snapshot_bytes):
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

            current_source_bytes = read_owner_file(
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
                agent = _new_agent(owner_id)
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
                retained_path = (
                    str(path) if _implementation.RETAIN_SOURCE_FILES else None
                )
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
                if not isinstance(registry_record, Mapping):
                    raise RuntimeError(
                        "Document registry did not return the committed row."
                    )
                registry_keeps_source = (
                    registry_record.get("source_retained") is True
                )
                if previous_path and not (
                    retained_path is not None
                    and _same_retained_source(previous_path, path)
                ):
                    _implementation._safe_unlink_upload(previous_path)
                _implementation._JOB_STORE.update(
                    job_id,
                    owner_id,
                    status="success",
                    filename=display_name,
                    source_path=str(registry_record.get("source_path") or "")[:4096],
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


def _delete_document_for_owner(owner_id: str, document_id: str) -> bool:
    """Delete owner-scoped vectors, registry state, and retained source bytes."""

    owner = normalize_owner_id(owner_id)
    doc_id = _required_identifier(document_id, "doc_id")
    rag = _implementation.get_rag_layer()
    try:
        results = rag.collection.get(
            where={
                "$and": [
                    {"owner_id": {"$eq": owner}},
                    {"doc_id": {"$eq": doc_id}},
                ]
            },
            include=["metadatas"],
            limit=1,
        )
    except Exception as exc:
        raise RuntimeError("Vector document lookup is unavailable.") from exc
    if not isinstance(results, Mapping):
        raise RuntimeError("Vector document lookup returned an invalid response.")
    metadatas = results.get("metadatas") or []
    if not isinstance(metadatas, list):
        raise RuntimeError("Vector document lookup returned invalid metadata.")
    vector_exists = any(
        isinstance(metadata, Mapping)
        and metadata.get("owner_id") == owner
        and metadata.get("doc_id") == doc_id
        for metadata in metadatas[:1]
    )
    registry_record = _implementation._DOCUMENT_STORE.get(
        owner_id=owner,
        doc_id=doc_id,
    )
    if not vector_exists and registry_record is None:
        return False
    if vector_exists:
        rag.delete_document(owner_id=owner, doc_id=doc_id)
    record = _implementation._DOCUMENT_STORE.delete(
        owner_id=owner,
        doc_id=doc_id,
    )
    source_path = str(
        (record or registry_record or {}).get("source_path") or ""
    )[:4096]
    _implementation._safe_unlink_upload(source_path)
    return True


async def _security_error_handler(
    _request: Request,
    _exc: SecurityError,
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request."},
        headers={"Cache-Control": "no-store"},
    )


async def _json_decode_error_handler(
    _request: Request,
    _exc: json.JSONDecodeError,
) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"detail": "A scientific tool returned an invalid response."},
        headers={"Cache-Control": "no-store"},
    )


_implementation._bounded_identifier = _bounded_identifier
_implementation._safe_request_id = _safe_request_id
_implementation._new_agent = _new_agent
_implementation._validated_upload_file = _validated_upload_file
_implementation._forget_future = _forget_future
_implementation._release_scheduled_ingestion = _release_scheduled_ingestion
_implementation._schedule_ingestion_attempt = _schedule_ingestion_attempt
_implementation._submit_ingestion = _submit_ingestion
_implementation.materialize_ingestion_snapshot = materialize_ingestion_snapshot
_implementation.read_owner_file = read_owner_file
_implementation._retry_or_fail_job = _retry_or_fail_job
_implementation._retry_or_fail_snapshot = _retry_or_fail_snapshot
_implementation._recover_interrupted_jobs = _recover_interrupted_jobs
_implementation.process_ingestion = process_ingestion
_implementation._delete_document_for_owner = _delete_document_for_owner
_implementation.app.add_exception_handler(SecurityError, _security_error_handler)
_implementation.app.add_exception_handler(
    json.JSONDecodeError,
    _json_decode_error_handler,
)
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
