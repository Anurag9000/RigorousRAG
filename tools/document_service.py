"""One ingestion/indexing service shared by CLI and HTTP entrypoints."""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from tools.ingestion import ingest_file
from tools.ingestion_models import IngestedDocument
from tools.privacy import mask_metadata_text, sanitize_metadata_dict
from tools.rag import RAGLayer, get_rag_layer
from tools.security import normalize_owner_id


@dataclass(frozen=True)
class IndexedDocument:
    document: IngestedDocument
    chunk_count: int


def _summary_sample(document: IngestedDocument, max_chars: int = 9000) -> str:
    """Sample across the document rather than summarising only its first page."""

    if len(document.text) <= max_chars:
        return document.text
    third = max_chars // 3
    middle_start = max(0, len(document.text) // 2 - third // 2)
    return (
        "\n\n[BEGINNING]\n" + document.text[:third]
        + "\n\n[MIDDLE]\n" + document.text[middle_start:middle_start + third]
        + "\n\n[END]\n" + document.text[-third:]
    )


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_identity(document: IngestedDocument, owner_id: str) -> None:
    """Reject a source changed after parsing, even if metadata was preserved."""

    if document.metadata.get("document_identity") != "owner_and_source_sha256":
        return
    owner = normalize_owner_id(owner_id)
    unresolved = Path(document.file_path)
    if unresolved.is_symlink():
        raise ValueError("The source became a symbolic link before indexing.")
    source = unresolved.resolve()
    if not source.exists() or not source.is_file():
        raise ValueError("The source disappeared before indexing.")
    current_hash = _source_sha256(source)
    expected_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"rigorousrag:{owner}:{current_hash}")
    )
    if expected_id != document.id:
        raise ValueError("The source changed after parsing and before indexing.")


def summarize_document(
    document: IngestedDocument,
    *,
    client: Optional[Any] = None,
    model: Optional[str] = None,
) -> str:
    if client is None:
        return document.text[:800].strip()
    summary_model = model or os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
    try:
        response = client.chat.completions.create(
            model=summary_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize the document in exactly two cautious sentences: "
                        "first its main contribution or purpose, then its method/evidence. "
                        "Do not invent facts not present in the sample."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Title: {document.title or document.filename}\n\n"
                        f"{_summary_sample(document)}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=220,
        )
        value = (response.choices[0].message.content or "").strip()
        return mask_metadata_text(value) or document.text[:800].strip()
    except Exception:
        return document.text[:800].strip()


def index_document(
    document: IngestedDocument,
    *,
    owner_id: str,
    rag: Optional[RAGLayer] = None,
    client: Optional[Any] = None,
    summary_model: Optional[str] = None,
    job_id: Optional[str] = None,
) -> IndexedDocument:
    """Index evidence metadata only; filesystem paths belong in DocumentStore."""

    _verify_source_identity(document, owner_id)
    rag = rag or get_rag_layer()
    summary = summarize_document(document, client=client, model=summary_model)
    document.metadata = sanitize_metadata_dict({
        **document.metadata,
        "llm_summary": summary,
    })
    metadata = {
        "filename": document.filename,
        "mime_type": document.mime_type,
        "owner_id": owner_id,
        "created_at": document.created_at.isoformat(),
        "llm_summary": summary,
        **{
            key: value
            for key, value in document.metadata.items()
            if isinstance(value, (str, int, float, bool))
        },
    }
    if job_id:
        metadata["job_id"] = job_id
    chunk_count = rag.add_document(
        doc_id=document.id,
        text=document.text,
        sections=document.sections,
        metadata=metadata,
        replace=True,
    )
    return IndexedDocument(document=document, chunk_count=chunk_count)


def ingest_and_index(
    file_path: str,
    *,
    owner_id: str,
    rag: Optional[RAGLayer] = None,
    client: Optional[Any] = None,
    summary_model: Optional[str] = None,
    job_id: Optional[str] = None,
) -> IndexedDocument:
    result = ingest_file(file_path, owner_id=owner_id)
    if not result.success or result.document is None:
        raise ValueError(result.error or "Document ingestion failed.")
    return index_document(
        result.document,
        owner_id=owner_id,
        rag=rag,
        client=client,
        summary_model=summary_model,
        job_id=job_id,
    )
