"""Handbook search tool with adaptive full-text / TF-IDF retrieval."""

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Handbook file location (sibling of tools/ directory)
# ---------------------------------------------------------------------------
HANDBOOK_PATH = Path(__file__).parent.parent / "handbook.md"

# In-memory cache: invalidated whenever the handbook file's mtime changes
_HANDBOOK_CACHE: Dict = {"mtime": None, "index": None, "chunks": None}

# Maximum handbook size (chars) before switching to TF-IDF retrieval
_FULL_TEXT_THRESHOLD = 10_000


# ---------------------------------------------------------------------------
# In-memory TF-IDF index helpers (reuses repo's own Indexer + Crawler)
# ---------------------------------------------------------------------------


def _paragraph_chunks(content: str) -> List[Tuple[str, str]]:
    """
    Split handbook content into (chunk_id, text) pairs by double-newline.
    Merges very short paragraphs (< 120 chars) with the next one.
    """
    raw = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks: List[Tuple[str, str]] = []
    buffer = ""
    for i, para in enumerate(raw):
        buffer = (buffer + " " + para).strip() if buffer else para
        if len(buffer) >= 120 or i == len(raw) - 1:
            chunks.append((f"hb_{i}", buffer))
            buffer = ""
    return chunks


def _build_handbook_index(content: str):
    """
    Build an in-memory TF-IDF index over handbook chunks using the repo's own
    InvertedIndex implementation.  Returns (index, chunks_list).
    """
    from Crawler import Page  # local import to avoid circular import at module level
    from Indexer import InvertedIndex

    chunks = _paragraph_chunks(content)
    pages: Dict = {}
    for cid, text in chunks:
        pages[cid] = Page(
            url=cid,
            title=f"Handbook — {cid}",
            text=text,
            links=[],
            content_type="text/plain",
            content_length=len(text),
        )

    idx = InvertedIndex()
    idx.build(pages)
    return idx, chunks


def _tfidf_search(query: str, idx, chunks: List[Tuple[str, str]], top_k: int = 3) -> str:
    """
    Score handbook chunks against a query using the inverted index's IDF weights
    and return the top-k results formatted as Markdown passages.
    """
    from Indexer import tokenize

    tokens = tokenize(query)
    if not tokens:
        return ""

    q_counter = Counter(tokens)
    raw_scores: Dict[str, float] = {}

    for term, freq in q_counter.items():
        idf = idx.idf.get(term)
        if idf is None:
            continue
        q_weight = (1.0 + math.log(freq)) * idf
        for chunk_id, d_weight in idx.index.get(term, {}).items():
            raw_scores[chunk_id] = raw_scores.get(chunk_id, 0.0) + q_weight * d_weight

    if not raw_scores:
        # No index hits — return first 5 000 chars as a safe fallback
        return ""

    chunk_map = {cid: text for cid, text in chunks}
    top_ids = sorted(raw_scores, key=raw_scores.__getitem__, reverse=True)[:top_k]

    parts = [
        f"**Handbook Passage {i + 1}:**\n\n{chunk_map[cid]}"
        for i, cid in enumerate(top_ids)
        if cid in chunk_map
    ]
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_handbook(query: str) -> str:
    """
    Retrieve relevant content from the internal handbook for the agent to use.

    **Small handbooks (≤ 10 000 chars):** returns the full text so that the
    LLM can process it without any retrieval loss.

    **Large handbooks (> 10 000 chars):** builds an in-memory TF-IDF index
    (using the same InvertedIndex as the main search engine) and returns the
    top-3 most relevant passages.  The index is rebuilt only when the handbook
    file's modification time changes (mtime-keyed cache invalidation).

    Falls back to returning the first 5 000 characters if the index produces
    no hits for the given query.
    """
    if not HANDBOOK_PATH.exists():
        return (
            "⚠️  Handbook file not found.  Expected location: "
            f"{HANDBOOK_PATH}"
        )

    try:
        stat = HANDBOOK_PATH.stat()
        content = HANDBOOK_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return f"⚠️  Error reading handbook: {exc}"

    # Small handbook — return full text
    if len(content) <= _FULL_TEXT_THRESHOLD:
        return content

    # Large handbook — use TF-IDF retrieval with mtime-based cache invalidation
    if _HANDBOOK_CACHE["mtime"] != stat.st_mtime:
        idx, chunks = _build_handbook_index(content)
        _HANDBOOK_CACHE["mtime"] = stat.st_mtime
        _HANDBOOK_CACHE["index"] = idx
        _HANDBOOK_CACHE["chunks"] = chunks

    result = _tfidf_search(
        query,
        _HANDBOOK_CACHE["index"],
        _HANDBOOK_CACHE["chunks"],
        top_k=3,
    )

    # Graceful fallback: return start of handbook when index has no hits
    return result if result else content[:5_000]


# ---------------------------------------------------------------------------
# Tool schema definition (imported by search_agent.py)
# ---------------------------------------------------------------------------

HANDBOOK_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_handbook",
        "description": (
            "Search the internal company/lab handbook for policies, guidelines, "
            "operating procedures, and data-privacy rules.  Use this for questions "
            "about internal best practices, compliance, or operational instructions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The specific policy or topic to look up in the handbook.",
                }
            },
            "required": ["query"],
        },
    },
}
