"""Small owner-independent policy handbook retrieval."""

from __future__ import annotations

import math
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

HANDBOOK_PATH = Path(__file__).resolve().parent.parent / "handbook.md"
_CACHE: Dict[str, Any] = {"mtime_ns": None, "index": None, "chunks": None}
_CACHE_LOCK = threading.Lock()


def _paragraph_chunks(content: str) -> List[Tuple[str, str]]:
    paragraphs = [paragraph.strip() for paragraph in content.split("\n\n") if paragraph.strip()]
    chunks: List[Tuple[str, str]] = []
    buffer: List[str] = []
    length = 0
    for paragraph in paragraphs:
        if buffer and length + len(paragraph) > 1200:
            chunks.append((f"handbook-{len(chunks) + 1}", "\n\n".join(buffer)))
            buffer, length = [], 0
        buffer.append(paragraph)
        length += len(paragraph)
    if buffer:
        chunks.append((f"handbook-{len(chunks) + 1}", "\n\n".join(buffer)))
    return chunks


def _build_index(content: str):
    from Crawler import Page
    from Indexer import InvertedIndex

    chunks = _paragraph_chunks(content)
    pages = {
        chunk_id: Page(
            url=chunk_id,
            title=chunk_id,
            text=text,
            links=[],
            content_type="text/markdown",
            content_length=len(text),
        )
        for chunk_id, text in chunks
    }
    index = InvertedIndex()
    index.build(pages)
    return index, chunks


def _search(query: str, index, chunks: List[Tuple[str, str]], top_k: int = 3) -> List[Tuple[str, str]]:
    from Indexer import tokenize

    tokens = tokenize(query)
    if not tokens:
        return []
    scores: Dict[str, float] = {}
    for term, frequency in Counter(tokens).items():
        idf = index.idf.get(term)
        if idf is None:
            continue
        query_weight = (1.0 + math.log(frequency)) * idf
        for chunk_id, document_weight in index.index.get(term, {}).items():
            scores[chunk_id] = scores.get(chunk_id, 0.0) + query_weight * document_weight
    chunk_map = dict(chunks)
    return [
        (chunk_id, chunk_map[chunk_id])
        for chunk_id in sorted(scores, key=scores.get, reverse=True)[:top_k]
        if chunk_id in chunk_map
    ]


def search_handbook(query: str) -> str:
    query = (query or "").strip()
    if not query:
        raise ValueError("A handbook query is required.")
    if not HANDBOOK_PATH.exists():
        raise FileNotFoundError(f"Handbook not found at {HANDBOOK_PATH}.")
    stat = HANDBOOK_PATH.stat()
    with _CACHE_LOCK:
        if _CACHE["mtime_ns"] != stat.st_mtime_ns:
            content = HANDBOOK_PATH.read_text(encoding="utf-8")
            index, chunks = _build_index(content)
            _CACHE.update({"mtime_ns": stat.st_mtime_ns, "index": index, "chunks": chunks})
        results = _search(query, _CACHE["index"], _CACHE["chunks"], top_k=3)
    if not results:
        return "No handbook passage matched the query."
    return "\n\n---\n\n".join(
        f"**{chunk_id}**\n\n{text}" for chunk_id, text in results
    )


HANDBOOK_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_handbook",
        "description": "Retrieve relevant internal operating or privacy policy passages.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 2000}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
