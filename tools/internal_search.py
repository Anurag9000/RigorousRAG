"""Adapter from the classic academic index to canonical citations."""

from __future__ import annotations

import itertools
import json
import operator
import os
import stat
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from Searching import AcademicSearchEngine, SearchHit
from tools.models import Citation

_FileIdentity = Tuple[str, int, int, int, int, int]
_StorageSignature = Tuple[str, Tuple[_FileIdentity, ...]]
_ENGINE_INSTANCE: Optional[AcademicSearchEngine] = None
_ENGINE_SIGNATURE: Optional[_StorageSignature] = None
_ENGINE_LOCK = threading.Lock()
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_STORAGE_PATH_CHARS = 4096
_MAX_RESULTS = 20
_MAX_ENGINE_RELOAD_ATTEMPTS = 3
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_link_or_reparse(info: os.stat_result) -> bool:
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _identity(info: os.stat_result) -> tuple[int, int]:
    return int(info.st_dev), int(info.st_ino)


def _absolute_without_resolving(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError("CLASSIC_STORAGE_DIR must be a filesystem path.")
    try:
        rendered = os.fspath(path)
    except TypeError as exc:
        raise ValueError("CLASSIC_STORAGE_DIR must be a filesystem path.") from exc
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_STORAGE_PATH_CHARS
        or _contains_ascii_control(rendered)
    ):
        raise ValueError("CLASSIC_STORAGE_DIR is invalid or too long.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for component in (absolute, *absolute.parents):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(
                "CLASSIC_STORAGE_DIR could not be inspected safely."
            ) from exc
        if _is_link_or_reparse(info):
            raise ValueError(
                "CLASSIC_STORAGE_DIR may not contain symbolic-link or reparse-point components."
            )
    return absolute


def _file_identity(path: Path) -> _FileIdentity:
    """Return identity that changes on replacement, metadata change, or removal."""

    try:
        info = os.lstat(path)
        if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
            return (path.name, -2, -2, -2, -2, -2)
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


def _safe_manifest_path(manifest: Path) -> Optional[Path]:
    try:
        absolute = Path(os.path.abspath(manifest))
        for component in (absolute, *absolute.parents):
            try:
                info = os.lstat(component)
            except FileNotFoundError:
                if component == absolute:
                    return None
                continue
            if _is_link_or_reparse(info):
                return None
        return absolute
    except (OSError, TypeError, ValueError):
        return None


def _read_manifest(manifest: Path) -> Optional[dict[str, Any]]:
    absolute = _safe_manifest_path(manifest)
    if absolute is None:
        return None
    try:
        before = os.lstat(absolute)
    except OSError:
        return None
    if (
        _is_link_or_reparse(before)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > _MAX_MANIFEST_BYTES
    ):
        return None
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
        descriptor = os.open(absolute, flags)
    except OSError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if (
            _is_link_or_reparse(metadata)
            or not stat.S_ISREG(metadata.st_mode)
            or _identity(metadata) != expected_identity
            or metadata.st_size <= 0
            or metadata.st_size > _MAX_MANIFEST_BYTES
        ):
            return None
        payload = bytearray()
        while True:
            remaining = _MAX_MANIFEST_BYTES + 1 - len(payload)
            if remaining <= 0:
                return None
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > _MAX_MANIFEST_BYTES:
                return None
        opened_after = os.fstat(descriptor)
        if (
            _identity(opened_after) != expected_identity
            or (
                int(opened_after.st_ctime_ns),
                int(opened_after.st_mtime_ns),
                int(opened_after.st_size),
            ) != expected_metadata
        ):
            return None
    finally:
        os.close(descriptor)
    safe_after = _safe_manifest_path(absolute)
    if safe_after is None:
        return None
    try:
        after = os.lstat(safe_after)
    except OSError:
        return None
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
        return None
    try:
        parsed = json.loads(
            bytes(payload).decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-standard JSON constant {value}")
            ),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        RecursionError,
    ):
        return None
    return parsed if isinstance(parsed, dict) else None


def _manifest_member_paths(root: Path, manifest: Path) -> List[Path]:
    """Return exact generation members named by one bounded regular manifest."""

    parsed = _read_manifest(manifest)
    if parsed is None:
        return []
    generation = parsed.get("generation")
    if not isinstance(generation, str) or not (
        len(generation) == 32
        and all(character in "0123456789abcdef" for character in generation)
    ):
        return []
    files = parsed.get("files")
    if not isinstance(files, dict):
        return []
    expected_names = {
        "crawl": f"crawl_state.{generation}.json",
        "index": f"index.{generation}.json",
        "pagerank": f"pagerank.{generation}.json",
    }
    members: List[Path] = []
    for key, expected in expected_names.items():
        entry = files.get(key)
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or name != expected or Path(name).name != name:
            return []
        members.append(root / name)
    return members


def _storage_signature(storage_dir: str | os.PathLike[str]) -> _StorageSignature:
    root = _absolute_without_resolving(storage_dir)
    manifest = root / "snapshot_manifest.json"
    paths = [
        manifest,
        root / "crawl_state.json",
        root / "index.json",
        root / "pagerank.json",
    ]
    paths.extend(_manifest_member_paths(root, manifest))
    identities = tuple(_file_identity(path) for path in paths)
    return str(root), identities


def _close_engine(engine: Any) -> None:
    if engine is None:
        return
    try:
        engine.close()
    except Exception:
        pass


def get_engine() -> AcademicSearchEngine:
    """Reload only from storage that remains stable throughout initialization."""

    global _ENGINE_INSTANCE, _ENGINE_SIGNATURE
    storage_dir = os.getenv("CLASSIC_STORAGE_DIR", "data")
    with _ENGINE_LOCK:
        signature = _storage_signature(storage_dir)
        if _ENGINE_INSTANCE is not None and _ENGINE_SIGNATURE == signature:
            return _ENGINE_INSTANCE

        for _attempt in range(_MAX_ENGINE_RELOAD_ATTEMPTS):
            if _ENGINE_INSTANCE is not None and _ENGINE_SIGNATURE == signature:
                return _ENGINE_INSTANCE
            replacement = AcademicSearchEngine(storage_dir=str(signature[0]))
            replacement_signature = _storage_signature(signature[0])
            if replacement_signature != signature:
                _close_engine(replacement)
                signature = replacement_signature
                continue

            previous = _ENGINE_INSTANCE
            _ENGINE_INSTANCE = replacement
            _ENGINE_SIGNATURE = replacement_signature
            if previous is not replacement:
                _close_engine(previous)
            return replacement

        raise RuntimeError(
            "Classic search state changed repeatedly during engine initialization."
        )


class InternalSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

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


def _bounded_limit(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("limit must be an integer.")
    try:
        parsed = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("limit must be an integer.") from exc
    numeric = int(parsed)
    if not 1 <= numeric <= _MAX_RESULTS:
        raise ValueError(f"limit must be between 1 and {_MAX_RESULTS}.")
    return numeric


def search_internal(query: str, limit: int = 5) -> List[Citation]:
    if not isinstance(query, str):
        raise ValueError("Internal-search queries must be strings.")
    bounded_query = query.strip()
    if not bounded_query:
        return []
    if len(bounded_query) > 2000 or _contains_ascii_control(bounded_query):
        raise ValueError(
            "Internal-search queries may contain at most 2,000 valid characters."
        )
    requested = _bounded_limit(limit)
    hits = get_engine().search(bounded_query, limit=requested)
    if isinstance(hits, (str, bytes, bytearray)):
        return []
    try:
        candidates = itertools.islice(iter(hits), requested)
    except Exception:
        return []
    citations: List[Citation] = []
    try:
        for hit in candidates:
            if not isinstance(hit, SearchHit):
                continue
            try:
                citation = Citation(
                    label=f"[{len(citations) + 1}]",
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
            except Exception:
                continue
            citations.append(citation)
    except Exception:
        return citations
    return citations