"""Path-safe boundary over classic crawl/index persistence.

The complete generation, digest, migration, and locking implementation remains in
``storage_legacy``. This module prevents CLASSIC_STORAGE_DIR from being silently
redirected through a final-path symbolic link.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import storage_legacy as _implementation

_original_storage_manager = _implementation.StorageManager


def _lexical_absolute(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


class StorageManager(_original_storage_manager):
    """Classic storage manager with a non-symlinked persistent root."""

    def __init__(self, base_dir: Path | str = "data") -> None:
        raw_root = Path(base_dir)
        if raw_root.is_symlink():
            raise ValueError("CLASSIC_STORAGE_DIR may not be a symbolic link.")
        super().__init__(_lexical_absolute(raw_root))
        self._ensure_storage_root()

    def _ensure_storage_root(self) -> None:
        if self.base_dir.is_symlink():
            raise OSError("CLASSIC_STORAGE_DIR became a symbolic link.")
        if not self.base_dir.exists() or not self.base_dir.is_dir():
            raise OSError("CLASSIC_STORAGE_DIR must remain an existing directory.")

    @contextmanager
    def _snapshot_guard(self) -> Iterator[None]:
        self._ensure_storage_root()
        with super()._snapshot_guard():
            self._ensure_storage_root()
            yield
            self._ensure_storage_root()

    def _read_json(self, path: Path):
        self._ensure_storage_root()
        return super()._read_json(path)

    def _write_bytes(self, path: Path, encoded: bytes) -> None:
        self._ensure_storage_root()
        super()._write_bytes(path, encoded)
        self._ensure_storage_root()

    def _write_json(self, path: Path, payload: Any) -> None:
        self._ensure_storage_root()
        super()._write_json(path, payload)
        self._ensure_storage_root()


_implementation.StorageManager = StorageManager
_implementation.__doc__ = __doc__
sys.modules[__name__] = _implementation
