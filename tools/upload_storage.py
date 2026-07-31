"""Descriptor-anchored owner upload writes, reads, copies, and deletions.

Final owner-directory and file lookups are relative to already-opened descriptors on
POSIX. Portable fallbacks repeat path and directory-identity checks. All lexical root
components are validated before any path normalization can hide a symbolic link or
Windows reparse point.
"""

from __future__ import annotations

import operator
import os
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Tuple

from tools.security import normalize_owner_id, safe_upload_suffix

_MAX_LOCAL_FILE_BYTES = 1_000_000_000
_MAX_LOCAL_PATH_CHARS = 4096
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


class UploadStorageError(ValueError):
    """Raised when local upload storage violates an ownership or path invariant."""


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_link_or_reparse(info: os.stat_result) -> bool:
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _identity(info: os.stat_result) -> tuple[int, int]:
    return int(info.st_dev), int(info.st_ino)


def _positive_limit(value: object) -> int:
    if isinstance(value, bool):
        raise UploadStorageError("max_bytes must be an integer.")
    try:
        parsed = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UploadStorageError("max_bytes must be an integer.") from exc
    limit = int(parsed)
    if not 1 <= limit <= _MAX_LOCAL_FILE_BYTES:
        raise UploadStorageError(
            f"max_bytes must be between 1 and {_MAX_LOCAL_FILE_BYTES}."
        )
    return limit


def _absolute_lexical_path(value: str | os.PathLike[str], label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise UploadStorageError(f"{label} must be a filesystem path.")
    try:
        rendered = os.fspath(value)
    except TypeError as exc:
        raise UploadStorageError(f"{label} must be a filesystem path.") from exc
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_LOCAL_PATH_CHARS
        or _contains_ascii_control(rendered)
    ):
        raise UploadStorageError(f"{label} is invalid or too long.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _no_redirected_components(path: Path, label: str) -> None:
    for component in (path, *path.parents):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise UploadStorageError(f"{label} could not be inspected safely.") from exc
        if _is_link_or_reparse(info):
            raise UploadStorageError(
                f"{label} may not contain symbolic-link or reparse-point components."
            )


def _root_directory(upload_root: str | Path) -> Path:
    root = _absolute_lexical_path(upload_root, "UPLOAD_DIR")
    _no_redirected_components(root, "UPLOAD_DIR")
    try:
        info = os.lstat(root)
    except OSError as exc:
        raise UploadStorageError("UPLOAD_DIR must be an existing directory.") from exc
    if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise UploadStorageError("UPLOAD_DIR must be an existing safe directory.")
    return root


def _portable_owner_directory(root: Path, owner: str, *, create: bool) -> Path:
    owner_dir = root / owner
    if create:
        owner_dir.mkdir(mode=0o700, exist_ok=True)
    try:
        info = os.lstat(owner_dir)
    except OSError as exc:
        raise UploadStorageError("Owner upload directory is unavailable.") from exc
    if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise UploadStorageError("Owner upload directory is invalid or redirected.")
    return owner_dir


@contextmanager
def _owner_directory(
    upload_root: str | Path,
    owner_id: str,
    *,
    create: bool,
) -> Iterator[Tuple[Path, Optional[int], Path]]:
    """Yield ``(root, owner_fd, owner_path)`` with no-follow owner resolution."""

    root = _root_directory(upload_root)
    owner = normalize_owner_id(owner_id)
    root_before = os.lstat(root)
    expected_root = _identity(root_before)
    if os.name == "nt":  # pragma: no cover
        owner_path = _portable_owner_directory(root, owner, create=create)
        before = os.lstat(owner_path)
        try:
            yield root, None, owner_path
        finally:
            _no_redirected_components(root, "UPLOAD_DIR")
            root_after = os.lstat(root)
            if (
                _is_link_or_reparse(root_after)
                or not stat.S_ISDIR(root_after.st_mode)
                or _identity(root_after) != expected_root
            ):
                raise UploadStorageError("UPLOAD_DIR changed during operation.")
            after = os.lstat(owner_path)
            if (
                _is_link_or_reparse(after)
                or not stat.S_ISDIR(after.st_mode)
                or _identity(before) != _identity(after)
            ):
                raise UploadStorageError(
                    "Owner upload directory changed during operation."
                )
        return

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_fd = os.open(root, directory_flags)
    owner_fd: Optional[int] = None
    try:
        opened_root = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(opened_root.st_mode)
            or _identity(opened_root) != expected_root
        ):
            raise UploadStorageError("UPLOAD_DIR identity changed while it was opened.")
        current_root = os.lstat(root)
        if (
            _is_link_or_reparse(current_root)
            or _identity(current_root) != expected_root
        ):
            raise UploadStorageError("UPLOAD_DIR changed before owner access.")
        if create:
            try:
                os.mkdir(owner, mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
        try:
            owner_fd = os.open(owner, directory_flags, dir_fd=root_fd)
        except FileNotFoundError as exc:
            raise UploadStorageError(
                "Owner upload directory does not exist."
            ) from exc
        if not stat.S_ISDIR(os.fstat(owner_fd).st_mode):
            raise UploadStorageError("Owner upload path is not a directory.")
        yield root, owner_fd, root / owner
    finally:
        if owner_fd is not None:
            os.close(owner_fd)
        os.close(root_fd)


def _relative_owner_file(
    upload_root: str | Path,
    source_path: str | Path | None,
) -> Optional[Tuple[Path, str, str]]:
    if source_path is None or source_path == "":
        return None
    try:
        root = _root_directory(upload_root)
        candidate = _absolute_lexical_path(source_path, "source_path")
    except UploadStorageError:
        return None
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) != 2 or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        return None
    owner, filename = relative.parts
    try:
        owner = normalize_owner_id(owner)
    except ValueError:
        return None
    if Path(filename).name != filename or _contains_ascii_control(filename):
        return None
    return root, owner, filename


def validated_owner_file_path(
    upload_root: str | Path,
    source_path: str | Path | None,
) -> Optional[Path]:
    parsed = _relative_owner_file(upload_root, source_path)
    if parsed is None:
        return None
    root, owner, filename = parsed
    try:
        with _owner_directory(root, owner, create=False) as (
            _root,
            owner_fd,
            owner_path,
        ):
            if owner_fd is None:  # pragma: no cover
                candidate = owner_path / filename
                metadata = os.lstat(candidate)
            else:
                metadata = os.stat(
                    filename,
                    dir_fd=owner_fd,
                    follow_symlinks=False,
                )
            if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
                return None
            return root / owner / filename
    except (FileNotFoundError, NotADirectoryError, OSError, UploadStorageError):
        return None


def _read_bounded(source: BinaryIO, limit: int) -> Optional[bytes]:
    payload = bytearray()
    while True:
        remaining = limit + 1 - len(payload)
        if remaining <= 0:
            return None
        chunk = source.read(min(1024 * 1024, remaining))
        if not chunk:
            return bytes(payload)
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise UploadStorageError("Upload streams must produce bytes.")
        payload.extend(bytes(chunk))
        if len(payload) > limit:
            return None


def read_owner_file(
    upload_root: str | Path,
    source_path: str | Path | None,
    *,
    max_bytes: int,
) -> Optional[bytes]:
    limit = _positive_limit(max_bytes)
    parsed = _relative_owner_file(upload_root, source_path)
    if parsed is None:
        return None
    root, owner, filename = parsed
    try:
        with _owner_directory(root, owner, create=False) as (
            _root,
            owner_fd,
            owner_path,
        ):
            if owner_fd is None:  # pragma: no cover
                candidate = owner_path / filename
                before = os.lstat(candidate)
                if (
                    _is_link_or_reparse(before)
                    or not stat.S_ISREG(before.st_mode)
                    or before.st_size > limit
                ):
                    return None
                with candidate.open("rb") as handle:
                    opened = os.fstat(handle.fileno())
                    if _identity(before) != _identity(opened):
                        return None
                    payload = _read_bounded(handle, limit)
                after = os.lstat(candidate)
                if (
                    _is_link_or_reparse(after)
                    or _identity(after) != _identity(before)
                ):
                    return None
                return payload

            flags = (
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            descriptor = os.open(filename, flags, dir_fd=owner_fd)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
                    return None
                with os.fdopen(descriptor, "rb", closefd=False) as handle:
                    return _read_bounded(handle, limit)
            finally:
                os.close(descriptor)
    except (FileNotFoundError, NotADirectoryError, OSError, UploadStorageError):
        return None


def store_owner_stream(
    source: BinaryIO,
    *,
    upload_root: str | Path,
    owner_id: str,
    suffix: str,
    max_bytes: int,
) -> Path:
    owner = normalize_owner_id(owner_id)
    safe_suffix = safe_upload_suffix(f"upload{suffix}")
    limit = _positive_limit(max_bytes)
    filename = f"{uuid.uuid4().hex}{safe_suffix}"
    try:
        source.seek(0)
    except (AttributeError, OSError):
        pass

    with _owner_directory(upload_root, owner, create=True) as (
        root,
        owner_fd,
        owner_path,
    ):
        if owner_fd is None:  # pragma: no cover
            destination = owner_path / filename
            total = 0
            try:
                with destination.open("xb") as handle:
                    opened = os.fstat(handle.fileno())
                    if not stat.S_ISREG(opened.st_mode):
                        raise UploadStorageError(
                            "Upload destination is not a regular file."
                        )
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        if not isinstance(chunk, (bytes, bytearray, memoryview)):
                            raise UploadStorageError(
                                "Upload streams must produce bytes."
                            )
                        data = bytes(chunk)
                        total += len(data)
                        if total > limit:
                            raise UploadStorageError(
                                f"Upload exceeds the {limit}-byte limit."
                            )
                        handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                final = os.lstat(destination)
                if (
                    _is_link_or_reparse(final)
                    or not stat.S_ISREG(final.st_mode)
                    or _identity(final) != _identity(opened)
                ):
                    raise UploadStorageError(
                        "Upload destination changed during storage."
                    )
            except Exception:
                try:
                    current = os.lstat(destination)
                    if (
                        'opened' in locals()
                        and not _is_link_or_reparse(current)
                        and _identity(current) == _identity(opened)
                    ):
                        destination.unlink()
                except OSError:
                    pass
                raise
            return root / owner / filename

        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(filename, flags, 0o600, dir_fd=owner_fd)
        total = 0
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise UploadStorageError(
                    "Upload destination is not a regular file."
                )
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise UploadStorageError(
                            "Upload streams must produce bytes."
                        )
                    data = bytes(chunk)
                    total += len(data)
                    if total > limit:
                        raise UploadStorageError(
                            f"Upload exceeds the {limit}-byte limit."
                        )
                    handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.unlink(filename, dir_fd=owner_fd)
            except OSError:
                pass
            raise
        finally:
            os.close(descriptor)
        return root / owner / filename


def copy_path_to_owner(
    source_path: str | Path,
    *,
    upload_root: str | Path,
    owner_id: str,
    max_bytes: int,
) -> Path:
    source = _absolute_lexical_path(source_path, "source_path")
    _no_redirected_components(source, "source_path")
    before = os.lstat(source)
    if _is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise UploadStorageError("Source file must be a safe regular file.")
    limit = _positive_limit(max_bytes)
    if before.st_size > limit:
        raise UploadStorageError(
            f"Source file exceeds the {limit}-byte retention limit."
        )
    suffix = safe_upload_suffix(source.name)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(source, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise UploadStorageError("Source file must be a regular file.")
        if _identity(before) != _identity(opened):
            raise UploadStorageError(
                "Source file changed before it could be copied."
            )
        if opened.st_size > limit:
            raise UploadStorageError(
                f"Source file exceeds the {limit}-byte retention limit."
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return store_owner_stream(
                handle,
                upload_root=upload_root,
                owner_id=owner_id,
                suffix=suffix,
                max_bytes=limit,
            )
    finally:
        os.close(descriptor)


def remove_owner_file(
    upload_root: str | Path,
    source_path: str | Path | None,
) -> bool:
    parsed = _relative_owner_file(upload_root, source_path)
    if parsed is None:
        return False
    root, owner, filename = parsed
    try:
        with _owner_directory(root, owner, create=False) as (
            _root,
            owner_fd,
            owner_path,
        ):
            if owner_fd is None:  # pragma: no cover
                candidate = owner_path / filename
                metadata = os.lstat(candidate)
                if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
                    return False
                candidate.unlink()
                return True

            metadata = os.stat(
                filename,
                dir_fd=owner_fd,
                follow_symlinks=False,
            )
            if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
                return False
            os.unlink(filename, dir_fd=owner_fd)
            return True
    except (FileNotFoundError, NotADirectoryError, OSError, UploadStorageError):
        return False
