"""Adapter from the classic academic index to canonical citations."""

from __future__ import annotations

import hashlib
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

_FileIdentity = Tuple[str, int, int, int, int, int, int, str]
_StorageSignature = Tuple[str, Tuple[_FileIdentity, ...]]
_DigestCacheKey = Tuple[str, int, int, int, int, int, int]
_ENGINE_INSTANCE: Optional[AcademicSearchEngine] = None
_ENGINE_SIGNATURE: Optional[_StorageSignature] = None
_ENGINE_LOCK = threading.Lock()
_DIGEST_CACHE: dict[_DigestCacheKey, str] = {}
_DIGEST_CACHE_LOCK = threading.Lock()
_MAX_DIGEST_CACHE_ENTRIES = 32
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_SIGNATURE_FILE_BYTES = 2_000_000_000
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


def _birthtime_ns(info: os.stat_result) -> int:
    value = getattr(info, "st_birthtime_ns", None)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return 0
    seconds = getattr(info, "st_birthtime", None)
    if seconds is None:
        return 0
    try:
        return int(float(seconds) * 1_000_000_000)
    except (TypeError, ValueError, OverflowError):
        return 0


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


def _metadata_key(path: Path, info: os.stat_result) -> _DigestCacheKey:
    return (
        str(path),
        int(info.st_dev),
        int(info.st_ino),
        _birthtime_ns(info),
        int(info.st_ctime_ns),
        int(info.st_mtime_ns),
        int(info.st_size),
    )


def _same_file_metadata(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _identity(left) == _identity(right)
        and _birthtime_ns(left) == _birthtime_ns(right)
        and int(left.st_ctime_ns) == int(right.st_ctime_ns)
        and int(left.st_mtime_ns) == int(right.st_mtime_ns)
        and int(left.st_size) == int(right.st_size)
    )


def _cache_digest(key: _DigestCacheKey, value: str) -> str:
    with _DIGEST_CACHE_LOCK:
        if key in _DIGEST_CACHE:
            return _DIGEST_CACHE[key]
        if len(_DIGEST_CACHE) >= _MAX_DIGEST_CACHE_ENTRIES:
            oldest = next(iter(_DIGEST_CACHE))
            _DIGEST_CACHE.pop(oldest, None)
        _DIGEST_CACHE[key] = value
    return value


def _file_content_digest(path: Path, before: os.stat_result) -> str:
    """Hash one stable regular file, caching only where metadata identity is reliable.

    Windows deliberately rehashes on every signature check. Hosted NT filesystems can
    report path and handle timestamps/file identifiers differently for the same regular
    file, so Windows validates descriptor stability and size rather than requiring an
    exact lstat/fstat metadata tuple. The digest itself remains content-sensitive, and
    the caller compares complete storage signatures before and after engine construction.
    """

    key = _metadata_key(path, before)
    if os.name != "nt":
        with _DIGEST_CACHE_LOCK:
            cached = _DIGEST_CACHE.get(key)
        if cached is not None:
            return cached
    if before.st_size < 0 or before.st_size > _MAX_SIGNATURE_FILE_BYTES:
        return "oversize"

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOINHERIT", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return "unreadable"
    try:
        opened = os.fstat(descriptor)
        if _is_link_or_reparse(opened) or not stat.S_ISREG(opened.st_mode):
            return "unstable"
        if os.name == "nt":
            if int(before.st_size) != int(opened.st_size):
                return "unstable"
        elif not _same_file_metadata(before, opened):
            return "unstable"

        hasher = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_SIGNATURE_FILE_BYTES:
                return "oversize"
            hasher.update(chunk)
        if total != int(opened.st_size):
            return "unstable"
        opened_after = os.fstat(descriptor)
        if os.name == "nt":
            if not _same_file_metadata(opened, opened_after):
                return "unstable"
        elif not _same_file_metadata(before, opened_after):
            return "unstable"
    except OSError:
        return "unreadable"
    finally:
        os.close(descriptor)

    try:
        after = os.lstat(path)
    except OSError:
        return "unreadable"
    if _is_link_or_reparse(after) or not stat.S_ISREG(after.st_mode):
        return "unstable"
    if os.name == "nt":
        if int(after.st_size) != total:
            return "unstable"
    elif not _same_file_metadata(before, after):
        return "unstable"

    computed = hasher.hexdigest()
    if os.name == "nt":
        return computed
    return _cache_digest(key, computed)


def _file_identity(path: Path) -> _FileIdentity:
    """Return identity that changes on replacement, content/metadata change, or removal."""

    try:
        info = os.lstat(path)
        if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
            return (path.name, -2, -2, -2, -2, -2, -2, "invalid")
        return (
            path.name,
            int(info.st_dev),
            int(info.st_ino),
            _birthtime_ns(info),
            int(info.st_ctime_ns),
            int(info.st_mtime_ns),
            int(info.st_size),
            _file_content_digest(path, info),
        )
    except OSError:
        return (path.name, -1, -1, -1, -1, -1, -1, "missing")


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
    """Read one bounded regular manifest without trusting NT path/handle identity parity."""

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
        _birthtime_ns(before),
        int(before.st_ctime_ns),
        int(before.st_mtime_ns),
        int(before.st_size),
    )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOINHERIT", 0)
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
            or metadata.st_size <= 0
            or metadata.st_size > _MAX_MANIFEST_BYTES
        ):
            return None
        if os.name == "nt":
            if int(metadata.st_size) != int(before.st_size):
                return None
        elif _identity(metadata) != expected_identity:
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
        if len(payload) != int(metadata.st_size):
            return None
        opened_after = os.fstat(descriptor)
        if os.name == "nt":
            if not _same_file_metadata(metadata, opened_after):
                return None
        elif (
            _identity(opened_after) != expected_identity
            or (
                _birthtime_ns(opened_after),
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
    if _is_link_or_reparse(after) or not stat.S_ISREG(after.st_mode):
        return None
    if os.name == "nt":
        if int(after.st_size) != len(payload):
            return None
    elif (
        _identity(after) != expected_identity
        or (
            _birthtime_ns(after),
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