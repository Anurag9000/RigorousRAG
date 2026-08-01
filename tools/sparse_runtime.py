"""Process-local runtime factories for sparse and coordinated index stores."""

from __future__ import annotations

import os
import stat
import threading
from pathlib import Path
from typing import Any

from tools.generation_store import GenerationStore
from tools.index_coordinator import IndexCoordinator
from tools.sparse_index import SparseIndex
from tools.three_store_coordinator import AuthoritativeIndexCoordinator

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH_CHARS = 4096
_LOCK = threading.RLock()
_SPARSE: dict[str, SparseIndex] = {}
_GENERATIONS: dict[str, GenerationStore] = {}
_AUTHORITATIVE: dict[tuple[int, str, str], AuthoritativeIndexCoordinator] = {}


def _absolute_path(value: str | os.PathLike[str], label: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{label} must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH_CHARS
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in rendered)
    ):
        raise ValueError(f"{label} is invalid.")
    raw = Path(rendered)
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    absolute = Path(os.path.abspath(raw))
    for candidate in (absolute, *absolute.parents):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"{label} could not be validated.") from exc
        if stat.S_ISLNK(metadata.st_mode) or bool(
            int(getattr(metadata, "st_file_attributes", 0))
            & _WINDOWS_REPARSE_POINT
        ):
            raise ValueError(f"{label} may not contain redirects.")
    return str(absolute)


def get_sparse_index(path: str | os.PathLike[str] | None = None) -> SparseIndex:
    selected = (
        path
        if path is not None
        else os.getenv("SPARSE_INDEX_PATH", "data/sparse_index.sqlite3")
    )
    absolute = _absolute_path(selected, "SPARSE_INDEX_PATH")
    with _LOCK:
        instance = _SPARSE.get(absolute)
        if instance is None:
            instance = SparseIndex(absolute)
            _SPARSE[absolute] = instance
        return instance


def get_generation_store(
    path: str | os.PathLike[str] | None = None,
) -> GenerationStore:
    selected = path if path is not None else os.getenv(
        "INDEX_GENERATION_DB_PATH",
        "data/index_generations.sqlite3",
    )
    absolute = _absolute_path(selected, "INDEX_GENERATION_DB_PATH")
    with _LOCK:
        instance = _GENERATIONS.get(absolute)
        if instance is None:
            instance = GenerationStore(absolute)
            _GENERATIONS[absolute] = instance
        return instance


def get_authoritative_index_coordinator(
    *,
    rag: Any,
    sparse_path: str | os.PathLike[str] | None = None,
    generation_path: str | os.PathLike[str] | None = None,
) -> AuthoritativeIndexCoordinator:
    sparse = get_sparse_index(sparse_path)
    generations = get_generation_store(generation_path)
    sparse_key = str(sparse.path) if hasattr(sparse, "path") else repr(sparse)
    generation_key = str(generations.path)
    key = (id(rag), sparse_key, generation_key)
    with _LOCK:
        instance = _AUTHORITATIVE.get(key)
        if instance is None:
            instance = AuthoritativeIndexCoordinator(
                index=IndexCoordinator(rag=rag, sparse=sparse),
                generations=generations,
            )
            _AUTHORITATIVE[key] = instance
        return instance


def clear_index_runtime_caches() -> None:
    """Clear only process-local factories; durable data is not modified."""

    with _LOCK:
        _AUTHORITATIVE.clear()
        _SPARSE.clear()
        _GENERATIONS.clear()


__all__ = [
    "clear_index_runtime_caches",
    "get_authoritative_index_coordinator",
    "get_generation_store",
    "get_sparse_index",
]
