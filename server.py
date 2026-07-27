"""FastAPI service with request-scoped identity, agents, and bounded uploads."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from search_agent import SearchAgent
from tools.bib import export_to_bibtex
from tools.document_service import index_document
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

app = FastAPI(title="RigorousRAG API", version="4.0.0")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RETAIN_UPLOADS = os.getenv("RETAIN_UPLOADS", "false").lower() in {"1", "true", "yes"}
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", str(24 * 60 * 60)))
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "60"))
_JOB_STORE = JobStore(ttl_seconds=JOB_TTL_SECONDS)
_RATE_LIMITER = SlidingWindowRateLimiter(REQUESTS_PER_MINUTE)

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


@app.middleware("http")
async def security_headers(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id[:128]
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


def _new_agent(owner_id: str, model: Optional[str] = None) -> SearchAgent:
    selected = model or _DEFAULT_MODEL
    if selected not in _ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{selected}' is not enabled by the server.")
    return SearchAgent(
        model=selected,
        owner_id=owner_id,
        api_key=_PROVIDER_KEY,
        base_url=_BASE_URL,
    )


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=20_000)
    model: Optional[str] = None


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
        "retain_uploads": RETAIN_UPLOADS,
        "requests_per_minute": REQUESTS_PER_MINUTE,
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
        return total
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.post("/ingest", response_model=JobStatus)
async def ingest_document(
    background_tasks: BackgroundTasks,
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
        status="processing",
        filename=display_name,
    )
    background_tasks.add_task(
        process_ingestion,
        str(destination),
        display_name,
        job_id,
        principal.owner_id,
    )
    return JobStatus(job_id=job_id, status="processing", filename=display_name)


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    principal: Principal = Depends(get_principal),
) -> JobStatus:
    entry = await run_in_threadpool(_JOB_STORE.get, job_id, principal.owner_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatus(**entry)


def process_ingestion(file_path: str, display_name: str, job_id: str, owner_id: str) -> None:
    path = Path(file_path)
    try:
        result = ingest_file(file_path, owner_id=owner_id)
        if not result.success or result.document is None:
            raise ValueError(result.error or "Document ingestion failed.")
        document = result.document
        document.filename = display_name
        agent = _new_agent(owner_id)
        indexed = index_document(
            document,
            owner_id=owner_id,
            rag=get_rag_layer(),
            client=agent.client,
            job_id=job_id,
            storage_path=file_path if RETAIN_UPLOADS else None,
        )
        _JOB_STORE.update(
            job_id,
            owner_id,
            status="success",
            filename=display_name,
            message=f"Indexed {indexed.chunk_count} semantic chunks.",
            doc_id=document.id,
        )
    except Exception as exc:
        _JOB_STORE.update(
            job_id,
            owner_id,
            status="failed",
            filename=display_name,
            message=str(exc)[:2000],
        )
    finally:
        if not RETAIN_UPLOADS:
            path.unlink(missing_ok=True)


@app.get("/docs/list")
async def list_documents(
    principal: Principal = Depends(get_principal),
) -> list[Dict[str, Any]]:
    return await run_in_threadpool(
        get_rag_layer().list_documents,
        owner_id=principal.owner_id,
        limit=1000,
    )


@app.delete("/docs/{doc_id}")
async def delete_document(
    doc_id: str,
    principal: Principal = Depends(get_rate_limited_principal),
) -> Dict[str, str]:
    rag = get_rag_layer()
    results = rag.collection.get(
        where={
            "$and": [
                {"owner_id": {"$eq": principal.owner_id}},
                {"doc_id": {"$eq": doc_id}},
            ]
        },
        include=["metadatas"],
        limit=1,
    )
    metadatas = results.get("metadatas") or []
    if not metadatas:
        raise HTTPException(status_code=404, detail="Document not found.")
    storage_path = str((metadatas[0] or {}).get("storage_path") or "")
    await run_in_threadpool(rag.delete_document, owner_id=principal.owner_id, doc_id=doc_id)
    if storage_path:
        candidate = Path(storage_path).resolve()
        try:
            candidate.relative_to(UPLOAD_DIR)
            candidate.unlink(missing_ok=True)
        except ValueError:
            pass
    return {"status": "deleted", "doc_id": doc_id}


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
