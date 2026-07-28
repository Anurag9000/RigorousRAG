"""FastAPI service with request-scoped identity and durable ingestion."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from search_agent import SearchAgent
from tools.bib import export_to_bibtex
from tools.document_service import index_document
from tools.document_store import DocumentStore, get_document_store
from tools.ingestion import ingest_file
from tools.integrity import check_visual_entailment, extract_protocol
from tools.job_store import JobStore
from tools.models import AgentAnswer
from tools.rag import get_rag_layer
from tools.rate_limit import SlidingWindowRateLimiter
from tools.security import (
    DEFAULT_MAX_UPLOAD_BYTES,
    Principal,
    generated_upload_name,
    normalize_owner_id,
    parse_api_key_owners,
)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RETAIN_SOURCE_FILES = os.getenv(
    "RETAIN_SOURCE_FILES", os.getenv("RETAIN_UPLOADS", "true")
).lower() in {"1", "true", "yes"}
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", str(24 * 60 * 60)))
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "60"))
INGEST_WORKERS = max(1, min(int(os.getenv("INGEST_WORKERS", "2")), 16))
INGEST_MAX_ATTEMPTS = max(1, min(int(os.getenv("INGEST_MAX_ATTEMPTS", "3")), 20))
_JOB_STORE = JobStore(ttl_seconds=JOB_TTL_SECONDS)
_DOCUMENT_STORE: DocumentStore = get_document_store(upload_root=UPLOAD_DIR)
_RATE_LIMITER = SlidingWindowRateLimiter(REQUESTS_PER_MINUTE)
_INGEST_EXECUTOR = ThreadPoolExecutor(
    max_workers=INGEST_WORKERS,
    thread_name_prefix="rigorousrag-ingest",
)
_INGEST_FUTURES: set[Future[Any]] = set()
_INGEST_FUTURES_LOCK = threading.Lock()
_INGEST_TIMERS: Dict[str, threading.Timer] = {}
_INGEST_TIMERS_LOCK = threading.Lock()
_INGEST_SHUTDOWN = threading.Event()
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--local", action="store_true")
_parser.add_argument("--demo", action="store_true")
_args, _unknown = _parser.parse_known_args()

if _args.demo:
    _DEFAULT_MODEL = "qwen2.5:0.5b"
    _BASE_URL = "http://localhost:11434/v1"
    _PROVIDER_KEY = "ollama"
elif _args.local:
    _DEFAULT_MODEL = "llama3.1"
    _BASE_URL = "http://localhost:11434/v1"
    _PROVIDER_KEY = "ollama"
else:
    _DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
    _BASE_URL = os.getenv("OPENAI_BASE_URL")
    _PROVIDER_KEY = os.getenv("OPENAI_API_KEY")

_ALLOWED_MODELS = {
    value.strip()
    for value in os.getenv("ALLOWED_MODELS", _DEFAULT_MODEL).split(",")
    if value.strip()
}
_ALLOWED_MODELS.add(_DEFAULT_MODEL)
_API_KEY_OWNERS = parse_api_key_owners()
_SINGLE_USER_OWNER = normalize_owner_id(os.getenv("SINGLE_USER_OWNER_ID", "default_user"))


def _new_agent(owner_id: str, model: Optional[str] = None) -> SearchAgent:
    selected = model or _DEFAULT_MODEL
    if selected not in _ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{selected[:200]}' is not enabled by the server.",
        )
    return SearchAgent(
        model=selected,
        owner_id=owner_id,
        api_key=_PROVIDER_KEY,
        base_url=_BASE_URL,
    )


def _bounded_identifier(value: str, label: str, max_length: int = 200) -> str:
    identifier = (value or "").strip()
    if not identifier or len(identifier) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"{label} must contain between 1 and {max_length} characters.",
        )
    return identifier


def _safe_request_id(raw_value: Optional[str]) -> str:
    candidate = (raw_value or "").strip()
    return candidate if _REQUEST_ID_RE.fullmatch(candidate) else uuid.uuid4().hex


def _internal_failure_message(prefix: str, exc: BaseException) -> str:
    return f"{prefix} ({type(exc).__name__})."


def _validated_upload_file(path: str | Path | None) -> Optional[Path]:
    """Return a regular, non-symlink file contained by UPLOAD_DIR."""

    if path in (None, ""):
        return None
    raw_path = Path(path)
    if raw_path.is_symlink():
        return None
    candidate = raw_path.resolve()
    try:
        candidate.relative_to(UPLOAD_DIR)
    except ValueError:
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


def _safe_unlink_upload(path: str | Path | None) -> bool:
    """Delete one regular non-symlink file only when contained by UPLOAD_DIR."""

    candidate = _validated_upload_file(path)
    if candidate is None:
        return False
    candidate.unlink(missing_ok=True)
    return True


def _persist_failed_job(
    job_id: str,
    owner_id: str,
    display_name: str,
    path: str | Path | None,
    message: str,
) -> bool:
    """Persist failure before removing its source; preserve the file if persistence fails."""

    try:
        _JOB_STORE.update(
            job_id,
            owner_id,
            status="failed",
            filename=display_name,
            source_path="",
            message=message[:2000],
        )
    except Exception:
        return False
    _safe_unlink_upload(path)
    return True


def _forget_future(future: Future[Any]) -> None:
    with _INGEST_FUTURES_LOCK:
        _INGEST_FUTURES.discard(future)


def _release_scheduled_ingestion(
    file_path: str,
    display_name: str,
    job_id: str,
    owner_id: str,
) -> None:
    with _INGEST_TIMERS_LOCK:
        _INGEST_TIMERS.pop(job_id, None)
    if not _INGEST_SHUTDOWN.is_set():
        _submit_ingestion(file_path, display_name, job_id, owner_id)


def _submit_ingestion(
    file_path: str,
    display_name: str,
    job_id: str,
    owner_id: str,
) -> None:
    """Submit a due job or schedule one timer without occupying an executor worker."""

    if _INGEST_SHUTDOWN.is_set():
        return
    try:
        internal = _JOB_STORE.get_internal(job_id, owner_id)
    except Exception:
        return
    if not internal or str(internal.get("status") or "") != "queued":
        return
    due_at = float(internal.get("next_attempt_at") or 0.0)
    delay = max(0.0, due_at - time.time())
    if delay > 0:
        with _INGEST_TIMERS_LOCK:
            existing = _INGEST_TIMERS.get(job_id)
            if existing is not None and existing.is_alive():
                return
            timer = threading.Timer(
                delay,
                _release_scheduled_ingestion,
                args=(file_path, display_name, job_id, owner_id),
            )
            timer.daemon = True
            _INGEST_TIMERS[job_id] = timer
            timer.start()
        return

    with _INGEST_TIMERS_LOCK:
        existing = _INGEST_TIMERS.pop(job_id, None)
        if existing is not None:
            existing.cancel()
    try:
        future = _INGEST_EXECUTOR.submit(
            process_ingestion,
            file_path,
            display_name,
            job_id,
            owner_id,
        )
    except RuntimeError:
        return
    with _INGEST_FUTURES_LOCK:
        _INGEST_FUTURES.add(future)
    future.add_done_callback(_forget_future)


def _cancel_scheduled_ingestions() -> None:
    _INGEST_SHUTDOWN.set()
    with _INGEST_TIMERS_LOCK:
        timers = list(_INGEST_TIMERS.values())
        _INGEST_TIMERS.clear()
    for timer in timers:
        timer.cancel()


def _recover_interrupted_jobs() -> None:
    for record in _JOB_STORE.recoverable():
        job_id = str(record["job_id"])
        owner_id = str(record["owner_id"])
        display_name = str(record.get("filename") or "upload")
        doc_id = str(record.get("doc_id") or "")
        source_path = str(record.get("source_path") or "")
        attempts = int(record.get("attempts") or 0)
        registry_record = (
            _DOCUMENT_STORE.get(owner_id=owner_id, doc_id=doc_id)
            if doc_id
            else None
        )
        if registry_record is not None:
            retained_path = str(registry_record.get("source_path") or "")
            try:
                _JOB_STORE.update(
                    job_id,
                    owner_id,
                    status="success",
                    filename=display_name,
                    source_path=retained_path,
                    message="Recovered completed document finalization after restart.",
                    doc_id=doc_id,
                )
            except Exception:
                continue
            if not bool(registry_record.get("source_retained")):
                _safe_unlink_upload(source_path)
            continue
        candidate = _validated_upload_file(source_path)
        if attempts >= INGEST_MAX_ATTEMPTS:
            _persist_failed_job(
                job_id,
                owner_id,
                display_name,
                source_path,
                "Interrupted ingestion exhausted its retry limit.",
            )
            continue
        if candidate is None:
            _persist_failed_job(
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
            _JOB_STORE.update(
                job_id,
                owner_id,
                status="queued",
                filename=display_name,
                source_path=str(candidate),
                message="Recovered after service restart.",
                doc_id=doc_id or None,
            )
        except Exception:
            continue
        _submit_ingestion(str(candidate), display_name, job_id, owner_id)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    _INGEST_SHUTDOWN.clear()
    await run_in_threadpool(_recover_interrupted_jobs)
    yield
    _cancel_scheduled_ingestions()
    _INGEST_EXECUTOR.shutdown(wait=False, cancel_futures=False)


app = FastAPI(title="RigorousRAG API", version="4.3.0", lifespan=lifespan)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    request_id = _safe_request_id(request.headers.get("X-Request-ID"))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; "
        "base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
    )
    if request.url.path.startswith(("/query", "/ingest", "/status", "/docs", "/tool")):
        response.headers["Cache-Control"] = "no-store"
    return response


async def get_principal(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Principal:
    if not _API_KEY_OWNERS:
        return Principal(owner_id=_SINGLE_USER_OWNER, authenticated=False)
    owner_id = _API_KEY_OWNERS.get(x_api_key or "")
    if not owner_id:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return Principal(owner_id=owner_id, authenticated=True)


async def get_rate_limited_principal(
    principal: Principal = Depends(get_principal),
) -> Principal:
    retry_after = _RATE_LIMITER.retry_after(principal.owner_id)
    if retry_after > 0:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )
    return principal


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=20_000)
    model: Optional[str] = Field(default=None, max_length=200)


class JobStatus(BaseModel):
    job_id: str
    status: str
    filename: str
    message: Optional[str] = None
    doc_id: Optional[str] = None


class VisualEntailmentRequest(BaseModel):
    claim_text: str = Field(..., min_length=1, max_length=10_000)
    figure_id: str = Field(..., min_length=1, max_length=200)
    doc_id: str = Field(..., min_length=1, max_length=200)


class ProtocolRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=30_000)
    doc_id: Optional[str] = Field(default="", max_length=200)


class BibTeXRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=1000)
    authors: Optional[str] = Field(default="", max_length=3000)
    year: Optional[int] = Field(default=None, ge=1000, le=9999)
    doi: Optional[str] = Field(default="", max_length=500)
    journal: Optional[str] = Field(default="", max_length=1000)
    entry_type: str = Field(default="article", max_length=50)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "version": app.version}


@app.get("/config")
async def public_config() -> Dict[str, Any]:
    return {
        "auth_required": bool(_API_KEY_OWNERS),
        "allowed_models": sorted(_ALLOWED_MODELS),
        "default_model": _DEFAULT_MODEL,
        "max_upload_bytes": DEFAULT_MAX_UPLOAD_BYTES,
        "retain_source_files": RETAIN_SOURCE_FILES,
        "retain_uploads": RETAIN_SOURCE_FILES,
        "requests_per_minute": REQUESTS_PER_MINUTE,
        "ingest_workers": INGEST_WORKERS,
        "ingest_max_attempts": INGEST_MAX_ATTEMPTS,
        "ingest_retry_base_seconds": _JOB_STORE.retry_base_seconds,
        "ingest_retry_max_seconds": _JOB_STORE.retry_max_seconds,
    }


@app.post("/query", response_model=AgentAnswer)
async def run_query(
    request: QueryRequest,
    principal: Principal = Depends(get_rate_limited_principal),
) -> AgentAnswer:
    agent = _new_agent(principal.owner_id, request.model)
    return await run_in_threadpool(agent.run, request.query)


async def _save_upload(file: UploadFile, destination: Path) -> int:
    total = 0
    try:
        with destination.open("xb") as handle:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > DEFAULT_MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {DEFAULT_MAX_UPLOAD_BYTES}-byte limit.",
                    )
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        return total
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.post("/ingest", response_model=JobStatus)
async def ingest_document(
    file: UploadFile = File(...),
    principal: Principal = Depends(get_rate_limited_principal),
) -> JobStatus:
    suffix = generated_upload_name(file.filename)
    owner_dir = UPLOAD_DIR / principal.owner_id
    owner_dir.mkdir(parents=True, exist_ok=True)
    destination = owner_dir / f"{uuid.uuid4().hex}{suffix}"
    await _save_upload(file, destination)
    job_id = f"job_{uuid.uuid4().hex}"
    display_name = Path(file.filename or f"upload{suffix}").name
    _JOB_STORE.update(
        job_id,
        principal.owner_id,
        status="queued",
        filename=display_name,
        source_path=str(destination),
        message="Waiting for an ingestion worker.",
    )
    _submit_ingestion(
        str(destination),
        display_name,
        job_id,
        principal.owner_id,
    )
    return JobStatus(job_id=job_id, status="queued", filename=display_name)


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    principal: Principal = Depends(get_principal),
) -> JobStatus:
    identifier = _bounded_identifier(job_id, "job_id")
    entry = await run_in_threadpool(_JOB_STORE.get, identifier, principal.owner_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatus(**entry)


def process_ingestion(file_path: str, display_name: str, job_id: str, owner_id: str) -> None:
    """Claim and process one job; only the winning worker owns source cleanup."""

    path = _validated_upload_file(file_path)
    if path is None:
        _persist_failed_job(
            job_id,
            owner_id,
            display_name,
            file_path,
            "The ingestion source was missing, invalid, symlinked, or outside UPLOAD_DIR.",
        )
        return
    if not _JOB_STORE.claim(job_id, owner_id, INGEST_MAX_ATTEMPTS):
        return

    result = ingest_file(str(path), owner_id=owner_id)
    if not result.success or result.document is None:
        _persist_failed_job(
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
        agent = _new_agent(owner_id)
        indexed = index_document(
            document,
            owner_id=owner_id,
            rag=get_rag_layer(),
            client=agent.client,
            job_id=job_id,
        )
        _JOB_STORE.update(
            job_id,
            owner_id,
            status="finalizing",
            filename=display_name,
            source_path=str(path),
            message="Vector indexing completed; finalizing source lifecycle.",
            doc_id=document.id,
        )
        retained_path = str(path) if RETAIN_SOURCE_FILES else None
        previous_path = _DOCUMENT_STORE.register(
            owner_id=owner_id,
            doc_id=document.id,
            filename=document.filename,
            mime_type=document.mime_type,
            source_path=retained_path,
        )
        registry_record = _DOCUMENT_STORE.get(owner_id=owner_id, doc_id=document.id) or {}
        keep_source = bool(registry_record.get("source_retained"))
        if previous_path:
            _safe_unlink_upload(previous_path)
        _JOB_STORE.update(
            job_id,
            owner_id,
            status="success",
            filename=display_name,
            source_path=str(registry_record.get("source_path") or ""),
            message=f"Indexed {indexed.chunk_count} semantic chunks.",
            doc_id=document.id,
        )
    except Exception as exc:
        registry_record = _DOCUMENT_STORE.get(owner_id=owner_id, doc_id=document.id)
        if registry_record is not None:
            retained_path = str(registry_record.get("source_path") or "")
            try:
                _JOB_STORE.update(
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
            internal = _JOB_STORE.get_internal(job_id, owner_id) or {}
            attempts = int(internal.get("attempts") or 0)
            if attempts < INGEST_MAX_ATTEMPTS:
                keep_source = True
                try:
                    _JOB_STORE.update(
                        job_id,
                        owner_id,
                        status="queued",
                        filename=display_name,
                        source_path=str(path),
                        message=_internal_failure_message(
                            "Transient ingestion failure; retry queued", exc
                        ),
                        doc_id=document.id,
                    )
                    _submit_ingestion(str(path), display_name, job_id, owner_id)
                except Exception:
                    pass
            else:
                if not _persist_failed_job(
                    job_id,
                    owner_id,
                    display_name,
                    path,
                    _internal_failure_message("Ingestion failed", exc),
                ):
                    keep_source = True
    finally:
        if not keep_source:
            _safe_unlink_upload(path)


@app.get("/docs/list")
async def list_documents(
    principal: Principal = Depends(get_principal),
) -> list[Dict[str, Any]]:
    documents = await run_in_threadpool(
        get_rag_layer().list_documents,
        owner_id=principal.owner_id,
        limit=1000,
    )
    for document in documents:
        record = _DOCUMENT_STORE.get(
            owner_id=principal.owner_id,
            doc_id=str(document.get("doc_id") or ""),
        )
        document["source_retained"] = bool((record or {}).get("source_retained"))
    return documents


@app.delete("/docs/{doc_id}")
async def delete_document(
    doc_id: str,
    principal: Principal = Depends(get_rate_limited_principal),
) -> Dict[str, str]:
    document_id = _bounded_identifier(doc_id, "doc_id")
    rag = get_rag_layer()
    results = rag.collection.get(
        where={
            "$and": [
                {"owner_id": {"$eq": principal.owner_id}},
                {"doc_id": {"$eq": document_id}},
            ]
        },
        include=["metadatas"],
        limit=1,
    )
    vector_exists = bool(results.get("metadatas") or [])
    registry_record = _DOCUMENT_STORE.get(
        owner_id=principal.owner_id,
        doc_id=document_id,
    )
    if not vector_exists and registry_record is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if vector_exists:
        await run_in_threadpool(
            rag.delete_document,
            owner_id=principal.owner_id,
            doc_id=document_id,
        )
    record = await run_in_threadpool(
        _DOCUMENT_STORE.delete,
        owner_id=principal.owner_id,
        doc_id=document_id,
    )
    _safe_unlink_upload(str((record or registry_record or {}).get("source_path") or ""))
    return {"status": "deleted", "doc_id": document_id}


@app.post("/tool/visual-entailment")
async def direct_visual_entailment(
    request: VisualEntailmentRequest,
    principal: Principal = Depends(get_rate_limited_principal),
) -> Dict[str, Any]:
    agent = _new_agent(principal.owner_id)
    raw = await run_in_threadpool(
        check_visual_entailment,
        request.claim_text,
        request.figure_id,
        request.doc_id,
        owner_id=principal.owner_id,
        client=agent.client,
        model=agent.model,
    )
    return json.loads(raw)


@app.post("/tool/protocol")
async def direct_extract_protocol(
    request: ProtocolRequest,
    principal: Principal = Depends(get_rate_limited_principal),
) -> Dict[str, Any]:
    agent = _new_agent(principal.owner_id)
    raw = await run_in_threadpool(
        extract_protocol,
        request.text,
        request.doc_id or "",
        client=agent.client,
        model=agent.model,
    )
    return json.loads(raw)


@app.post("/tool/bibtex")
async def direct_bibtex(
    request: BibTeXRequest,
    _principal: Principal = Depends(get_rate_limited_principal),
) -> Dict[str, str]:
    return {
        "bibtex": export_to_bibtex(citations=[{
            "entry_type": request.entry_type,
            "title": request.title,
            "authors": request.authors or "Unknown",
            "year": str(request.year) if request.year else "n.d.",
            "doi": request.doi or "",
            "url": f"https://doi.org/{request.doi}" if request.doi else "",
            "journal": request.journal or "",
        }])
    }


app.mount("/", StaticFiles(directory="frontend", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
