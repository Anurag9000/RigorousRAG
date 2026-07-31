"""Small owner-independent policy handbook retrieval."""

from __future__ import annotations

import math
import operator
import os
import stat
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tools.config import bounded_int_env

HANDBOOK_PATH = Path(__file__).resolve().parent.parent / "handbook.md"
HANDBOOK_MAX_BYTES = bounded_int_env(
    "HANDBOOK_MAX_BYTES",
    1_000_000,
    minimum=4096,
    maximum=20_000_000,
)
HANDBOOK_MAX_CHUNKS = bounded_int_env(
    "HANDBOOK_MAX_CHUNKS",
    1000,
    minimum=10,
    maximum=10_000,
)
_CHUNK_CHARS = 1200
_MAX_QUERY_CHARS = 2000
_MAX_PATH_CHARS = 4096
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_CACHE: Dict[str, Any] = {"signature": None, "index": None, "chunks": None}
_CACHE_LOCK = threading.Lock()


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_link_or_reparse(info: os.stat_result) -> bool:
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _identity(info: os.stat_result) -> tuple[int, int]:
    return int(info.st_dev), int(info.st_ino)


def _paragraph_chunks(content: str) -> List[Tuple[str, str]]:
    """Build hard-bounded chunks even when one paragraph is extremely long."""

    if not isinstance(content, str):
        raise ValueError("Handbook content must be text.")
    paragraphs = [
        paragraph.strip()
        for paragraph in content.split("\n\n")
        if paragraph.strip()
    ]
    units: List[str] = []
    for paragraph in paragraphs:
        for start in range(0, len(paragraph), _CHUNK_CHARS):
            unit = paragraph[start:start + _CHUNK_CHARS].strip()
            if unit:
                units.append(unit)
                if len(units) > HANDBOOK_MAX_CHUNKS:
                    raise ValueError("The handbook exceeds the configured chunk limit.")

    chunks: List[Tuple[str, str]] = []
    buffer: List[str] = []
    length = 0
    for unit in units:
        separator = 2 if buffer else 0
        if buffer and length + separator + len(unit) > _CHUNK_CHARS:
            chunks.append((f"handbook-{len(chunks) + 1}", "\n\n".join(buffer)))
            buffer, length = [], 0
        buffer.append(unit)
        length += (2 if length else 0) + len(unit)
    if buffer:
        chunks.append((f"handbook-{len(chunks) + 1}", "\n\n".join(buffer)))
    if len(chunks) > HANDBOOK_MAX_CHUNKS:
        raise ValueError("The handbook exceeds the configured chunk limit.")
    return chunks


def _lexical_handbook_path(path: Any) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise FileNotFoundError("The handbook is unavailable.")
    try:
        rendered = os.fspath(path)
    except TypeError as exc:
        raise FileNotFoundError("The handbook is unavailable.") from exc
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH_CHARS
        or _contains_ascii_control(rendered)
    ):
        raise FileNotFoundError("The handbook is unavailable.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    try:
        for component in (absolute, *absolute.parents):
            try:
                info = os.lstat(component)
            except FileNotFoundError:
                continue
            if _is_link_or_reparse(info):
                raise FileNotFoundError("The handbook is unavailable.")
    except OSError as exc:
        raise FileNotFoundError("The handbook is unavailable.") from exc
    return absolute


def _read_handbook(path: Path) -> Tuple[str, Tuple[str, int, int, int, int, int]]:
    """Read one bounded regular handbook through a stable no-follow identity."""

    source = _lexical_handbook_path(path)
    try:
        before = os.lstat(source)
    except OSError as exc:
        raise FileNotFoundError("The handbook is unavailable.") from exc
    if _is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError("The handbook must be a regular file.")
    if before.st_size > HANDBOOK_MAX_BYTES:
        raise ValueError("The handbook exceeds the configured byte limit.")
    expected_identity = _identity(before)
    expected_metadata = (
        int(before.st_ctime_ns),
        int(before.st_mtime_ns),
        int(before.st_size),
    )

    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise FileNotFoundError("The handbook is unavailable.") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            _is_link_or_reparse(opened)
            or not stat.S_ISREG(opened.st_mode)
            or _identity(opened) != expected_identity
        ):
            raise ValueError("The handbook changed while it was being opened.")
        data = bytearray()
        while True:
            remaining = HANDBOOK_MAX_BYTES + 1 - len(data)
            if remaining <= 0:
                raise ValueError("The handbook exceeds the configured byte limit.")
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > HANDBOOK_MAX_BYTES:
                raise ValueError("The handbook exceeds the configured byte limit.")
        opened_after = os.fstat(descriptor)
        if (
            _identity(opened_after) != expected_identity
            or (
                int(opened_after.st_ctime_ns),
                int(opened_after.st_mtime_ns),
                int(opened_after.st_size),
            ) != expected_metadata
        ):
            raise ValueError("The handbook changed while it was being read.")
    finally:
        os.close(descriptor)

    source = _lexical_handbook_path(source)
    try:
        after = os.lstat(source)
    except OSError as exc:
        raise FileNotFoundError("The handbook changed while it was being read.") from exc
    if (
        _is_link_or_reparse(after)
        or not stat.S_ISREG(after.st_mode)
        or _identity(after) != expected_identity
        or (
            int(after.st_ctime_ns),
            int(after.st_mtime_ns),
            int(after.st_size),
        ) != expected_metadata
    ):
        raise ValueError("The handbook changed while it was being read.")
    try:
        content = bytes(data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("The handbook must contain valid UTF-8 text.") from exc
    signature = (
        str(source),
        expected_identity[0],
        expected_identity[1],
        expected_metadata[0],
        expected_metadata[1],
        expected_metadata[2],
    )
    return content, signature


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


def _bounded_top_k(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("top_k must be an integer.")
    try:
        parsed = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("top_k must be an integer.") from exc
    result = int(parsed)
    if not 1 <= result <= 10:
        raise ValueError("top_k must be between 1 and 10.")
    return result


def _bounded_query(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Handbook queries must be strings.")
    bounded = value.strip()
    if (
        not bounded
        or len(bounded) > _MAX_QUERY_CHARS
        or _contains_ascii_control(bounded)
    ):
        raise ValueError("Handbook queries may contain at most 2,000 valid characters.")
    return bounded


def _search(
    query: str,
    index: Any,
    chunks: List[Tuple[str, str]],
    top_k: int = 3,
) -> List[Tuple[str, str]]:
    from Indexer import tokenize

    bounded_query = _bounded_query(query)
    limit = _bounded_top_k(top_k)
    tokens = tokenize(bounded_query)
    if not tokens:
        return []
    scores: Dict[str, float] = {}
    for term, frequency in Counter(tokens).items():
        idf = index.idf.get(term)
        if not isinstance(idf, (int, float)) or not math.isfinite(float(idf)) or idf <= 0:
            continue
        query_weight = (1.0 + math.log(frequency)) * float(idf)
        for chunk_id, document_weight in index.index.get(term, {}).items():
            if (
                not isinstance(chunk_id, str)
                or not isinstance(document_weight, (int, float))
                or not math.isfinite(float(document_weight))
                or document_weight <= 0
            ):
                continue
            score = scores.get(chunk_id, 0.0) + query_weight * float(document_weight)
            if math.isfinite(score):
                scores[chunk_id] = score
    chunk_map = dict(chunks)
    return [
        (chunk_id, chunk_map[chunk_id])
        for chunk_id in sorted(
            scores,
            key=lambda chunk_id: (-scores[chunk_id], chunk_id),
        )[:limit]
        if chunk_id in chunk_map
    ]


def search_handbook(query: str) -> str:
    bounded_query = _bounded_query(query)
    with _CACHE_LOCK:
        content, signature = _read_handbook(HANDBOOK_PATH)
        if _CACHE["signature"] != signature:
            index, chunks = _build_index(content)
            _CACHE.update({"signature": signature, "index": index, "chunks": chunks})
        results = _search(
            bounded_query,
            _CACHE["index"],
            _CACHE["chunks"],
            top_k=3,
        )
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
