"""
FastAPI backend for RigorousRAG.

New in this revision:
- Job registry with GET /status/{job_id} endpoint.
- LLM-powered 2-sentence summary generated during web-upload ingestion.
- X-Owner-ID header threading for multi-tenant document isolation.
- API key authentication via ALLOWED_API_KEYS env var (Gap A).
"""

import argparse
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from search_agent import SearchAgent
from tools.ingestion import ingest_file
from tools.models import AgentAnswer
from tools.rag import get_rag_layer

# ---------------------------------------------------------------------------
# API Key Authentication (Gap A)
# ---------------------------------------------------------------------------
# Set ALLOWED_API_KEYS=key1,key2,key3 in your environment to enable auth.
# When the variable is empty or unset, auth is DISABLED (dev/local mode).

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_ALLOWED_KEYS: set[str] = {
    k.strip() for k in os.getenv("ALLOWED_API_KEYS", "").split(",") if k.strip()
}


async def require_api_key(x_api_key: Optional[str] = Depends(_API_KEY_HEADER)) -> None:
    """
    Validates the X-API-Key header against ALLOWED_API_KEYS.

    Behaviour:
    - If ALLOWED_API_KEYS is unset/empty → auth is disabled (open access, dev mode).
    - If ALLOWED_API_KEYS is set → key must be present and valid, else 401.
    """
    if not _ALLOWED_KEYS:          # dev / local mode — skip auth
        return
    if not x_api_key or x_api_key not in _ALLOWED_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide a valid X-API-Key header.",
        )


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(title="RigorousRAG API", version="3.0.0")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# CLI mode selection
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="RigorousRAG Server")
_parser.add_argument("--local", action="store_true", help="Use Ollama locally")
_parser.add_argument("--demo", action="store_true", help="Ultra-fast demo via qwen2.5")
try:
    _args, _ = _parser.parse_known_args()
except Exception:
    class _DummyArgs:
        local = False
        demo = False
    _args = _DummyArgs()  # type: ignore[assignment]

if _args.demo:
    print("[INFO] DEMO mode — qwen2.5:0.5b via Ollama")
    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:11434/v1")
    agent = SearchAgent(
        model="qwen2.5:0.5b",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )
elif _args.local:
    print("[INFO] LOCAL mode — llama3.1 via Ollama")
    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:11434/v1")
    agent = SearchAgent(
        model="llama3.1",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )
else:
    agent = SearchAgent()

# ---------------------------------------------------------------------------
# Job registry (in-memory, protected by a threading.Lock)
# ---------------------------------------------------------------------------

_job_registry: Dict[str, dict] = {}
_job_lock = threading.Lock()


def _update_job(job_id: str, **fields) -> None:
    """Thread-safe upsert into the job registry, always storing job_id."""
    with _job_lock:
        entry = _job_registry.setdefault(job_id, {})
        entry["job_id"] = job_id   # always present so JobStatus(**entry) works
        entry.update(fields)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: str          # "processing" | "success" | "failed"
    filename: str
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/query", response_model=AgentAnswer, dependencies=[Depends(require_api_key)])
async def run_query(
    request: QueryRequest,
    x_owner_id: str = Header(default="default_user"),
) -> AgentAnswer:
    """
    Execute a research query through the agentic reasoning loop.

    The `X-Owner-ID` header scopes uploaded-document retrieval to the
    requesting user's own documents (multi-tenant isolation).
    """
    try:
        if request.model:
            agent.model = request.model
        # Thread owner_id into the agent for this request
        agent.owner_id = x_owner_id
        return agent.run(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest", response_model=JobStatus, dependencies=[Depends(require_api_key)])
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_owner_id: str = Header(default="default_user"),
) -> JobStatus:
    """
    Upload and asynchronously ingest a document into the vector store.

    Returns a `job_id` immediately.  Poll `GET /status/{job_id}` for status.
    """
    filename = file.filename or "upload"
    dest = UPLOAD_DIR / filename

    with dest.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    _update_job(job_id, status="processing", filename=filename)

    background_tasks.add_task(process_ingestion, str(dest), job_id, x_owner_id)

    return JobStatus(job_id=job_id, status="processing", filename=filename)


@app.get("/status/{job_id}", response_model=JobStatus, dependencies=[Depends(require_api_key)])
async def get_job_status(job_id: str) -> JobStatus:
    """
    Retrieve the current status of an ingestion job.

    Status values:
    - `"processing"` — background task is running.
    - `"success"`    — document ingested and indexed successfully.
    - `"failed"`     — error during parsing or indexing; see `message`.
    """
    with _job_lock:
        entry = _job_registry.get(job_id)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.  It may have expired or never existed.",
        )
    return JobStatus(**entry)


# ---------------------------------------------------------------------------
# Background ingestion worker
# ---------------------------------------------------------------------------


def process_ingestion(file_path: str, job_id: str, owner_id: str = "default_user") -> None:
    """
    Parse, redact, summarise (LLM), and index an uploaded document.

    Steps:
    1.  Ingest the file (parse + PII redaction + section detection).
    2.  Optionally generate a 2-sentence LLM summary (Goal 19), identical
        to the behaviour of `ingest_docs.py` CLI.
    3.  Add the document to the ChromaDB vector store, propagating all
        metadata including `owner_id` for per-user isolation.
    4.  Write final job status to the registry.
    """
    filename = Path(file_path).name
    print(f"[{job_id}] Ingesting {filename} for owner '{owner_id}' …")

    try:
        result = ingest_file(file_path, owner_id=owner_id)
    except Exception as exc:
        _update_job(job_id, status="failed", message=str(exc))
        print(f"[{job_id}] Parse error: {exc}")
        return

    if not result.success or not result.document:
        _update_job(
            job_id,
            status="failed",
            message=result.error or "Unknown ingestion error.",
        )
        print(f"[{job_id}] Failed: {result.error}")
        return

    doc = result.document

    # --- LLM 2-sentence summary (Gap 5 / Goal 19) ---
    llm_summary: Optional[str] = None
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key not in ("ollama", "local-no-key"):
        try:
            from openai import OpenAI  # local import avoids circular import at module level

            lm = OpenAI(api_key=api_key)
            resp = lm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a technical document summariser.  "
                            "Write exactly 2 concise sentences summarising the key contribution "
                            "and methodology of the provided document excerpt."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Document title: {doc.title}\n\n{doc.text[:4000]}",
                    },
                ],
                max_tokens=150,
                temperature=0.3,
            )
            llm_summary = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"[{job_id}] LLM summary skipped: {exc}")

    if llm_summary:
        doc.metadata["llm_summary"] = llm_summary
    elif doc.text:
        doc.metadata["llm_summary"] = doc.text[:500]

    # --- Index into ChromaDB ---
    try:
        rag = get_rag_layer()
        rag.add_document(
            doc_id=doc.id,
            text=doc.text,
            metadata={
                "filename": doc.filename,
                "mime_type": doc.mime_type,
                "owner_id": owner_id,
                "file_path": str(Path(file_path).absolute()),
                "job_id": job_id,
                "llm_summary": doc.metadata.get("llm_summary", ""),
                **{
                    k: v
                    for k, v in doc.metadata.items()
                    if k not in {"owner_id", "file_path", "job_id", "llm_summary"}
                    and isinstance(v, (str, int, float, bool))
                },
            },
        )
    except Exception as exc:
        _update_job(job_id, status="failed", message=f"Indexing error: {exc}")
        print(f"[{job_id}] Indexing error: {exc}")
        return

    _update_job(
        job_id,
        status="success",
        message=f"Indexed '{doc.filename}' ({len(doc.text)} chars, {len(doc.sections)} sections).",
    )
    print(f"[{job_id}] ✅ Indexed '{doc.filename}'.")



# ---------------------------------------------------------------------------
# Direct Tool Endpoints (bypass agent loop for UI panels)
# ---------------------------------------------------------------------------

class DocListItem(BaseModel):
    doc_id: str
    filename: str
    owner_id: str
    llm_summary: Optional[str] = None
    mime_type: Optional[str] = None


@app.get("/docs/list", dependencies=[Depends(require_api_key)])
async def list_documents(x_owner_id: str = Header(default="default_user")) -> list:
    """Return all documents indexed in ChromaDB for a given owner (for the Docs tab)."""
    try:
        rag = get_rag_layer()
        results = rag.collection.get(
            where={"owner_id": {"$eq": x_owner_id}},
            include=["metadatas"],
            limit=1000,
        )
        seen: dict = {}
        for meta in (results.get("metadatas") or []):
            doc_id = meta.get("doc_id", "")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", doc_id),
                    "owner_id": meta.get("owner_id", x_owner_id),
                    "llm_summary": meta.get("llm_summary"),
                    "mime_type": meta.get("mime_type"),
                }
        return list(seen.values())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class VisualEntailmentRequest(BaseModel):
    claim_text: str
    figure_id: str
    doc_id: str


@app.post("/tool/visual-entailment", dependencies=[Depends(require_api_key)])
async def direct_visual_entailment(req: VisualEntailmentRequest) -> dict:
    """Direct visual entailment — bypasses agent loop; calls GPT-4o Vision."""
    import json
    from tools.integrity import check_visual_entailment
    raw = check_visual_entailment(req.claim_text, req.figure_id, req.doc_id)
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}


class ProtocolRequest(BaseModel):
    text: str
    doc_id: Optional[str] = ""


@app.post("/tool/protocol", dependencies=[Depends(require_api_key)])
async def direct_extract_protocol(req: ProtocolRequest) -> dict:
    """Direct protocol extraction — GPT-4o JSON schema parse + regex fallback."""
    import json
    from tools.integrity import extract_protocol
    raw = extract_protocol(req.text, req.doc_id or "")
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}


class BibTeXRequest(BaseModel):
    title: str
    authors: Optional[str] = ""
    year: Optional[int] = None
    doi: Optional[str] = ""
    journal: Optional[str] = ""


@app.post("/tool/bibtex", dependencies=[Depends(require_api_key)])
async def direct_bibtex(req: BibTeXRequest) -> dict:
    """Generate a BibTeX entry directly for a given paper."""
    from tools.bib import export_to_bibtex
    bib = export_to_bibtex(citations=[{
        "title":   req.title,
        "authors": req.authors or "Unknown",
        "year":    str(req.year) if req.year else "n.d.",
        "doi":     req.doi or "",
        "url":     f"https://doi.org/{req.doi}" if req.doi else "",
        "journal": req.journal or "",
    }])
    return {"bibtex": bib}



# ---------------------------------------------------------------------------
# Static frontend (must be mounted AFTER all API routes)
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
