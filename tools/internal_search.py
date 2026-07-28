"""Adapter from the classic academic index to canonical citations."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from Searching import AcademicSearchEngine, SearchHit
from tools.models import Citation

_FileIdentity = Tuple[str, int, int, int, int, int]
_StorageSignature = Tuple[str, Tuple[_FileIdentity, ...]]
_ENGINE_INSTANCE: Optional[AcademicSearchEngine] = None
_ENGINE_SIGNATURE: Optional[_StorageSignature] = None
_ENGINE_LOCK = threading.Lock()
_MAX_MANIFEST_BYTES = 1_000_000


def _absolute_without_resolving(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _file_identity(path: Path) -> _FileIdentity:
    """Return identity that changes on atomic replacement, metadata change, or removal."""

    try:
        info = path.lstat()
        return (
            path.name,
            int(info.st_dev),
            int(info.st_ino),
            int(info.st_ctime_ns),
            int(info.st_mtime_ns),
            int(info.st_size),
        )
    except OSError:
        return (path.name, -1, -1, -1, -1, -1)


def _manifest_member_paths(root: Path, manifest: Path) -> List[Path]:
    """Return only exact single-component generation members named by a bounded manifest."""

    if manifest.is_symlink():
        return []
    try:
        info = manifest.stat()
        if not 0 < info.st_size <= _MAX_MANIFEST_BYTES:
            return []
        parsed = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return []
    files = parsed.get("files") if isinstance(parsed, dict) else None
    if not isinstance(files, dict):
        return []
    expected_prefixes = {
        "crawl": "crawl_state.",
        "index": "index.",
        "pagerank": "pagerank.",
    }
    members: List[Path] = []
    for key, prefix in expected_prefixes.items():
        entry = files.get(key)
        name = str(entry.get("name") or "") if isinstance(entry, dict) else ""
        if (
            not name
            or Path(name).name != name
            or not name.startswith(prefix)
            or not name.endswith(".json")
        ):
            return []
        members.append(root / name)
    return members


def _storage_signature(storage_dir: str) -> _StorageSignature:
    raw_root = _absolute_without_resolving(storage_dir)
    if raw_root.is_symlink():
        raise ValueError("CLASSIC_STORAGE_DIR may not be a symbolic link.")
    manifest = raw_root / "snapshot_manifest.json"
    paths = [
        manifest,
        raw_root / "crawl_state.json",
        raw_root / "index.json",
        raw_root / "pagerank.json",
    ]
    paths.extend(_manifest_member_paths(raw_root, manifest))
    identities = tuple(_file_identity(path) for path in paths)
    return str(raw_root), identities


def get_engine() -> AcademicSearchEngine:
    """Return an engine reloaded whenever committed or legacy storage identity changes."""

    global _ENGINE_INSTANCE, _ENGINE_SIGNATURE
    storage_dir = os.getenv("CLASSIC_STORAGE_DIR", "data")
    signature = _storage_signature(storage_dir)
    with _ENGINE_LOCK:
        if _ENGINE_INSTANCE is None or _ENGINE_SIGNATURE != signature:
            _ENGINE_INSTANCE = AcademicSearchEngine(storage_dir=storage_dir)
            _ENGINE_SIGNATURE = _storage_signature(storage_dir)
        return _ENGINE_INSTANCE


class InternalSearchInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


INTERNAL_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_internal",
        "description": "Search the locally crawled academic web index.",
        "parameters": InternalSearchInput.model_json_schema(),
    },
}


def search_internal(query: str, limit: int = 5) -> List[Citation]:
    query = (query or "").strip()
    if not query:
        return []
    if len(query) > 2000:
        raise ValueError("Internal-search queries may contain at most 2,000 characters.")
    limit = max(1, min(int(limit), 20))
    hits: List[SearchHit] = get_engine().search(query, limit=limit)
    return [
        Citation(
            label=f"[{index}]",
            title=hit.title,
            url=hit.url,
            source_type="academic_index",
            snippet=hit.snippet,
            source_id=hit.url,
            metadata={
                "combined_score": hit.score,
                "cosine": hit.cosine,
                "pagerank": hit.pagerank,
            },
        )
        for index, hit in enumerate(hits, start=1)
    ]
