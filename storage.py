"""Path-safe boundary over classic crawl/index persistence.

The complete generation, digest, migration, and locking implementation remains in
``storage_legacy``. This module normalizes standalone configuration and prevents
classic persistence from following symbolic links or accepting non-standard JSON.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from tools.config import bounded_int_env

# ``storage_legacy`` evaluates this value in its constructor. Normalize it before
# exposing that implementation through the public compatibility boundary.
os.environ["CLASSIC_MAX_SNAPSHOT_FILE_BYTES"] = str(
    bounded_int_env(
        "CLASSIC_MAX_SNAPSHOT_FILE_BYTES",
        250_000_000,
        minimum=1_000_000,
        maximum=2_000_000_000,
        write_back=True,
    )
)

import storage_legacy as _implementation

_original_storage_manager = _implementation.StorageManager


def _lexical_absolute(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant {value!r} is not allowed.")


class StorageManager(_original_storage_manager):
    """Classic storage manager with bounded no-follow persistent reads."""

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
        """Read one bounded regular file without following its final component."""

        self._ensure_storage_root()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            return None
        except OSError:
            return None

        valid_identity = False
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                return None
            if info.st_size < 0 or info.st_size > self.max_snapshot_file_bytes:
                raise ValueError("Persisted JSON exceeds the configured byte limit.")
            data = bytearray()
            while True:
                remaining = self.max_snapshot_file_bytes + 1 - len(data)
                if remaining <= 0:
                    raise ValueError("Persisted JSON exceeds the configured byte limit.")
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                data.extend(chunk)
            if len(data) > self.max_snapshot_file_bytes:
                raise ValueError("Persisted JSON exceeds the configured byte limit.")
            current = os.stat(path, follow_symlinks=False)
            valid_identity = (
                stat.S_ISREG(current.st_mode)
                and current.st_dev == info.st_dev
                and current.st_ino == info.st_ino
            )
            if not valid_identity:
                return None
            return json.loads(
                bytes(data).decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            if valid_identity and path.exists() and not path.is_symlink():
                self._quarantine(path)
            return None
        finally:
            os.close(descriptor)

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
