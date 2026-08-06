"""Path-safe boundary over classic crawl/index persistence.

The complete generation, digest, migration, and validation implementation remains in
``storage_legacy``. This module binds the public manager to one lexical root identity,
uses descriptor-relative persistence on POSIX, and retains identity-checked pathname
fallbacks on Windows where ``dir_fd`` support is incomplete.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from tools.config import bounded_int_env

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

if not hasattr(_implementation, "_boundary_original_StorageManager"):
    _implementation._boundary_original_StorageManager = _implementation.StorageManager
_original_storage_manager = _implementation._boundary_original_StorageManager
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


def _lexical_absolute(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError("CLASSIC_STORAGE_DIR must be a filesystem path.")
    try:
        rendered = os.fspath(path)
    except TypeError as exc:
        raise ValueError("CLASSIC_STORAGE_DIR must be a filesystem path.") from exc
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("CLASSIC_STORAGE_DIR is invalid or too long.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _is_link_or_reparse(info: os.stat_result) -> bool:
    """Return whether a stat result represents a symlink or Windows reparse point."""

    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _has_symlink_component(path: Path) -> bool:
    absolute = _lexical_absolute(path)
    for component in (absolute, *absolute.parents):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError:
            return True
        if _is_link_or_reparse(info):
            return True
    return False


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant {value!r} is not allowed.")


class _StorageManagerBoundary(_original_storage_manager):
    """Classic storage manager with identity-bound, bounded persistent I/O."""

    def __init__(self, base_dir: Path | str = "data") -> None:
        lexical_root = _lexical_absolute(base_dir)
        if _has_symlink_component(lexical_root):
            raise ValueError(
                "CLASSIC_STORAGE_DIR may not contain symbolic-link components."
            )
        lexical_root.mkdir(parents=True, exist_ok=True)
        if _has_symlink_component(lexical_root):
            raise ValueError(
                "CLASSIC_STORAGE_DIR may not contain symbolic-link components."
            )
        try:
            initial = os.stat(lexical_root, follow_symlinks=False)
        except OSError as exc:
            raise OSError("CLASSIC_STORAGE_DIR could not be opened safely.") from exc
        if not stat.S_ISDIR(initial.st_mode):
            raise ValueError("CLASSIC_STORAGE_DIR must be a directory.")

        self._lexical_root = lexical_root
        self._root_identity = (int(initial.st_dev), int(initial.st_ino))
        super().__init__(lexical_root)
        self._ensure_storage_root()

    def _ensure_storage_root(self) -> None:
        if _has_symlink_component(self._lexical_root):
            raise OSError("CLASSIC_STORAGE_DIR contains a symbolic-link component.")
        try:
            current = os.stat(self._lexical_root, follow_symlinks=False)
        except OSError as exc:
            raise OSError("CLASSIC_STORAGE_DIR is unavailable.") from exc
        if not stat.S_ISDIR(current.st_mode):
            raise OSError("CLASSIC_STORAGE_DIR must remain a directory.")
        identity = (int(current.st_dev), int(current.st_ino))
        if identity != self._root_identity:
            raise OSError("CLASSIC_STORAGE_DIR identity changed after initialization.")
        if _lexical_absolute(self.base_dir) != self._lexical_root:
            raise OSError("CLASSIC_STORAGE_DIR resolved to an unexpected location.")

    def _open_root_descriptor(self) -> int:
        self._ensure_storage_root()
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self._lexical_root, flags)
        try:
            info = os.fstat(descriptor)
            identity = (int(info.st_dev), int(info.st_ino))
            if not stat.S_ISDIR(info.st_mode) or identity != self._root_identity:
                raise OSError("CLASSIC_STORAGE_DIR descriptor identity is invalid.")
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    def _member_name(self, path: Path) -> str:
        candidate = _lexical_absolute(path)
        if candidate.parent != self._lexical_root:
            raise ValueError("Classic storage members must be direct children of the root.")
        name = candidate.name
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or any(ord(character) < 32 or ord(character) == 127 for character in name)
        ):
            raise ValueError("Classic storage member name is invalid.")
        return name

    def _member_path(self, path: Path) -> Path:
        return self._lexical_root / self._member_name(path)

    def _fsync_directory(self) -> None:
        if os.name == "nt":  # pragma: no cover - Windows-specific fallback
            self._ensure_storage_root()
            super()._fsync_directory()
            self._ensure_storage_root()
            return
        try:
            descriptor = self._open_root_descriptor()
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @contextmanager
    def _snapshot_guard(self) -> Iterator[None]:
        self._ensure_storage_root()
        if os.name == "nt":  # pragma: no cover - Windows-specific fallback
            with super()._snapshot_guard():
                self._ensure_storage_root()
                yield
                self._ensure_storage_root()
            return

        import fcntl

        with self._lock:
            root_descriptor = self._open_root_descriptor()
            lock_descriptor = -1
            try:
                flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
                lock_descriptor = os.open(
                    ".snapshot.lock",
                    flags,
                    0o600,
                    dir_fd=root_descriptor,
                )
                info = os.fstat(lock_descriptor)
                if not stat.S_ISREG(info.st_mode):
                    raise OSError("Snapshot lock must be a regular file.")
                try:
                    os.fchmod(lock_descriptor, 0o600)
                except OSError:
                    pass
                fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
                self._ensure_storage_root()
                yield
                self._ensure_storage_root()
            finally:
                if lock_descriptor >= 0:
                    try:
                        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                    except OSError:
                        pass
                    os.close(lock_descriptor)
                os.close(root_descriptor)

    def _quarantine_member(
        self,
        root_descriptor: int,
        name: str,
        expected_identity: tuple[int, int],
    ) -> None:
        try:
            current = os.stat(
                name,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
        except OSError:
            return
        identity = (int(current.st_dev), int(current.st_ino))
        if not stat.S_ISREG(current.st_mode) or identity != expected_identity:
            return
        destination = f"{name}.corrupt-{uuid.uuid4().hex[:8]}"
        try:
            os.replace(
                name,
                destination,
                src_dir_fd=root_descriptor,
                dst_dir_fd=root_descriptor,
            )
            os.fsync(root_descriptor)
        except OSError:
            pass

    def _quarantine_path_member(
        self,
        member_path: Path,
        expected_identity: tuple[int, int],
    ) -> None:
        """Conservatively quarantine one identity-checked pathname fallback member."""

        try:
            self._ensure_storage_root()
            current = os.lstat(member_path)
        except OSError:
            return
        identity = (int(current.st_dev), int(current.st_ino))
        if (
            _is_link_or_reparse(current)
            or not stat.S_ISREG(current.st_mode)
            or identity != expected_identity
        ):
            return
        super()._quarantine(member_path)
        self._ensure_storage_root()

    def _quarantine(self, path: Path) -> None:
        member_path = self._member_path(path)
        if os.name == "nt":  # pragma: no cover - Windows-specific fallback
            try:
                current = os.lstat(member_path)
            except OSError:
                return
            self._quarantine_path_member(
                member_path,
                (int(current.st_dev), int(current.st_ino)),
            )
            return
        try:
            root_descriptor = self._open_root_descriptor()
        except OSError:
            return
        try:
            try:
                current = os.stat(
                    member_path.name,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
            except OSError:
                return
            self._quarantine_member(
                root_descriptor,
                member_path.name,
                (int(current.st_dev), int(current.st_ino)),
            )
        finally:
            os.close(root_descriptor)

    def _read_json_path_fallback(self, path: Path):
        """Strict bounded pathname read used where descriptor-relative I/O is unavailable."""

        member_path = self._member_path(path)
        descriptor = -1
        identity: tuple[int, int] | None = None
        should_quarantine = False
        with self._lock:
            try:
                self._ensure_storage_root()
                before = os.lstat(member_path)
                if _is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
                    return None
                identity = (int(before.st_dev), int(before.st_ino))
                if before.st_size < 0 or before.st_size > self.max_snapshot_file_bytes:
                    raise ValueError("Persisted JSON exceeds the configured byte limit.")

                flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
                flags |= getattr(os, "O_NOINHERIT", 0)
                descriptor = os.open(member_path, flags)
                opened = os.fstat(descriptor)
                opened_identity = (int(opened.st_dev), int(opened.st_ino))
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or _is_link_or_reparse(opened)
                    or opened_identity != identity
                ):
                    return None

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

                after = os.lstat(member_path)
                after_identity = (int(after.st_dev), int(after.st_ino))
                if (
                    _is_link_or_reparse(after)
                    or not stat.S_ISREG(after.st_mode)
                    or after_identity != identity
                ):
                    return None
                self._ensure_storage_root()
                return json.loads(
                    bytes(data).decode("utf-8"),
                    parse_constant=_reject_json_constant,
                )
            except FileNotFoundError:
                return None
            except OSError:
                # Root-integrity failures are authority failures, not corrupt members.
                # Revalidation re-raises a swapped/missing root while preserving the
                # quarantine path for ordinary member-level I/O errors.
                self._ensure_storage_root()
                should_quarantine = identity is not None
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                RecursionError,
                TypeError,
                ValueError,
            ):
                should_quarantine = identity is not None
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

            if should_quarantine and identity is not None:
                self._quarantine_path_member(member_path, identity)
            return None

    def _read_json(self, path: Path):
        """Read one bounded regular root member without following path components."""

        member_path = self._member_path(path)
        if os.name == "nt":  # pragma: no cover - Windows-specific fallback
            return self._read_json_path_fallback(member_path)

        name = member_path.name
        with self._lock:
            root_descriptor = self._open_root_descriptor()
            descriptor = -1
            identity: tuple[int, int] | None = None
            try:
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                flags |= getattr(os, "O_NONBLOCK", 0)
                try:
                    descriptor = os.open(name, flags, dir_fd=root_descriptor)
                except FileNotFoundError:
                    return None
                except OSError:
                    return None
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode):
                    return None
                identity = (int(info.st_dev), int(info.st_ino))
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
                return json.loads(
                    bytes(data).decode("utf-8"),
                    parse_constant=_reject_json_constant,
                )
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RecursionError,
                TypeError,
                ValueError,
            ):
                if identity is not None:
                    self._quarantine_member(root_descriptor, name, identity)
                return None
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                os.close(root_descriptor)

    def _write_bytes(self, path: Path, encoded: bytes) -> None:
        if not isinstance(encoded, bytes):
            raise TypeError("Persisted content must be bytes.")
        if len(encoded) > self.max_snapshot_file_bytes:
            raise ValueError(
                f"Persisted JSON exceeds the {self.max_snapshot_file_bytes}-byte limit."
            )
        member_path = self._member_path(path)
        if os.name == "nt":  # pragma: no cover - Windows-specific fallback
            self._ensure_storage_root()
            super()._write_bytes(member_path, encoded)
            self._ensure_storage_root()
            return

        name = member_path.name
        temporary = f".{name}.{uuid.uuid4().hex}.tmp"
        with self._lock:
            root_descriptor = self._open_root_descriptor()
            descriptor = -1
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(
                    temporary,
                    flags,
                    0o600,
                    dir_fd=root_descriptor,
                )
                view = memoryview(encoded)
                offset = 0
                while offset < len(view):
                    written = os.write(descriptor, view[offset:])
                    if written <= 0:
                        raise OSError("Persisted JSON write made no progress.")
                    offset += written
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = -1
                os.replace(
                    temporary,
                    name,
                    src_dir_fd=root_descriptor,
                    dst_dir_fd=root_descriptor,
                )
                os.fsync(root_descriptor)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                try:
                    os.unlink(temporary, dir_fd=root_descriptor)
                except (FileNotFoundError, OSError):
                    pass
                os.close(root_descriptor)
            self._ensure_storage_root()

    def _write_json(self, path: Path, payload: Any) -> None:
        self._write_bytes(path, self._encode_json(payload))


if not hasattr(_implementation, "_boundary_public_StorageManager"):
    _implementation._boundary_public_StorageManager = _StorageManagerBoundary
StorageManager = _implementation._boundary_public_StorageManager

_implementation.StorageManager = StorageManager
_implementation.__doc__ = __doc__
sys.modules[__name__] = _implementation
