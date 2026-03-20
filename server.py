import os
import shutil
import argparse
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from pathlib import Path
import uvicorn

from search_agent import SearchAgent
from tools.models import AgentAnswer
from tools.ingestion import ingest_file
from tools.rag import get_rag_layer

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Academic Search Engine API", version="2.0.0")

# --- Configuration ---
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Parse command line args to check for --local
parser = argparse.ArgumentParser(description="RigorousRAG API Server")
parser.add_argument("--local", action="store_true", help="Run 100% locally via Ollama")
try:
    args, unknown = parser.parse_known_args()
except Exception:
    # Handle cases where uvicorn or tests pass unexpected args
    class DummyArgs:
        local = False
    args = DummyArgs()

# Global Agent Instance
if args.local:
    print("[INFO] Server starting in LOCAL mode (Ollama API)")
    os.environ["OPENAI_API_KEY"] = "ollama"
    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
    agent = SearchAgent(model="llama3.1", api_key="ollama", base_url="http://localhost:11434/v1")
else:
    agent = SearchAgent()

# --- Schemas ---

class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = "gpt-4o"

class JobStatus(BaseModel):
    job_id: str
    status: str
    filename: str

# --- Endpoints ---

# Web Frontend replaces API root
# @app.get("/")
# async def root():
#     return {"message": "Academic Search Engine API is running."}

@app.post("/query", response_model=AgentAnswer)
async def run_query(request: QueryRequest):
    """
    Run a specific query through the search agent.
    """
    try:
        if request.model:
            agent.model = request.model
        result = agent.run(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=JobStatus)
async def ingest_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload and ingest a document asynchronously.
    """
    filename = file.filename or "unknown_file"
    file_path = UPLOAD_DIR / filename
    
    # Save file
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    job_id = f"job_{os.urandom(4).hex()}"
    
    # Background processing
    background_tasks.add_task(process_ingestion, str(file_path), job_id)
    
    return JobStatus(job_id=job_id, status="processing", filename=filename)

def process_ingestion(file_path: str, job_id: str):
    """
    Background worker for ingestion and indexing.
    """
    print(f"[{job_id}] Processing {file_path}...")
    result = ingest_file(file_path)
    if result.success and result.document:
        rag = get_rag_layer()
        rag.add_document(
            doc_id=result.document.id,
            text=result.document.text,
            metadata={
                "filename": result.document.filename,
                "mime_type": result.document.mime_type,
                "job_id": job_id
            }
        )
        print(f"[{job_id}] Success and Indexed.")
    else:
        print(f"[{job_id}] Failed: {result.error}")

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
