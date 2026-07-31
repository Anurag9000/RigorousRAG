"""Failure-isolated, privacy-conscious bounded JSONL telemetry."""

from __future__ import annotations

import hashlib
import json
import math
import operator
import os
import stat
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from tools.privacy import mask_metadata_text, sanitize_metadata, sanitize_metadata_dict


def _bounded_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        value = default
    return max(minimum, min(value, maximum))


LOG_FILE = os.getenv("USAGE_LOG_FILE", "usage_metrics.jsonl")
LOG_MAX_BYTES = _bounded_int_env(
    "USAGE_LOG_MAX_BYTES",
    10 * 1024 * 1024,
    minimum=1024,
    maximum=10 * 1024 * 1024 * 1024,
)
LOG_BACKUPS = _bounded_int_env(
    "USAGE_LOG_BACKUPS",
    3,
    minimum=0,
    maximum=20,
)
_MAX_EVENT_BYTES = 64 * 1024
_MAX_PATH_CHARS = 4096
_MAX_PRIVATE_HASH_INPUT_CHARS = 100_000
_MAX_PUBLIC_INTEGER = 1_000_000_000
_LOG_LOCK = threading.Lock()
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _safe_text(value: Any, *, maximum: int, default: str = "") -> str:
    if value is None:
        rendered = default
    elif isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = str(value)
        except Exception:
            rendered = default
    return rendered[:maximum]


def _safe_bool(value: Any, default: bool = False) -> bool:
    try:
        return bool(value)
    except Exception:
        return default


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Retain the historical tighter telemetry shape over shared sanitization."""

    if depth > 6:
        return "[TRUNCATED_DEPTH]"
    sanitized = sanitize_metadata(value)
    if isinstance(sanitized, str):
        return sanitized[:4000]
    if isinstance(sanitized, bool) or sanitized is None:
        return sanitized
    if isinstance(sanitized, int):
        return max(-_MAX_PUBLIC_INTEGER, min(sanitized, _MAX_PUBLIC_INTEGER))
    if isinstance(sanitized, float):
        return sanitized if math.isfinite(sanitized) else None
    if isinstance(sanitized, dict):
        result: Dict[str, Any] = {}
        for index, (key, item) in enumerate(sanitized.items()):
            if index >= 100:
                result["__truncated_items__"] = True
                break
            result[str(key)[:200]] = _json_safe(item, depth=depth + 1)
        return result
    if isinstance(sanitized, list):
        items = [
            _json_safe(item, depth=depth + 1)
            for item in sanitized[:100]
        ]
        if len(sanitized) > 100:
            items.append({"__truncated_items__": True})
        return items
    return mask_metadata_text(sanitized)[:1000]


def _finite_nonnegative(value: Any, *, digits: int = 3) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return round(max(numeric, 0.0), digits)


def _nonnegative_integer(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        numeric = int(operator.index(value))
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(numeric, _MAX_PUBLIC_INTEGER))


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_redirecting(metadata: Any) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        int(getattr(metadata, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT
    )


def _is_regular_nonredirecting(metadata: Any) -> bool:
    return stat.S_ISREG(metadata.st_mode) and not _is_redirecting(metadata)


def _same_identity(left: Any, right: Any) -> bool:
    return _identity(left) == _identity(right)


def _snapshot_identity(metadata: Any) -> tuple[int, int, int, int, int, int]:
    ctime_ns = getattr(metadata, "st_ctime_ns", None)
    if ctime_ns is None:
        ctime_ns = int(float(getattr(metadata, "st_ctime", 0.0)) * 1_000_000_000)
    mtime_ns = getattr(metadata, "st_mtime_ns", None)
    if mtime_ns is None:
        mtime_ns = int(float(getattr(metadata, "st_mtime", 0.0)) * 1_000_000_000)
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(ctime_ns),
        int(mtime_ns),
        int(getattr(metadata, "st_size", -1)),
        int(metadata.st_mode),
    )


def _same_snapshot(left: Any, right: Any) -> bool:
    return _snapshot_identity(left) == _snapshot_identity(right)


def _absolute_without_resolving(path: Any) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError("Telemetry path must be a filesystem path.")
    rendered = os.fspath(path)
    if (
        not rendered
        or len(rendered) > _MAX_PATH_CHARS
        or _contains_ascii_control(rendered)
    ):
        raise ValueError("Telemetry path is invalid or too long.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _has_symlink_component(path: Path) -> bool:
    try:
        candidate = _absolute_without_resolving(path)
        for component in (candidate, *candidate.parents):
            try:
                metadata = component.lstat()
            except FileNotFoundError:
                continue
            if _is_redirecting(metadata):
                return True
        return False
    except (OSError, ValueError):
        return True


def _rotated_name(name: str, index: int) -> str:
    return f"{name}.{index}"


def _rotated_path(path: Path, index: int) -> Path:
    return path.with_name(_rotated_name(path.name, index))


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


@contextmanager
def _log_directory(path: Path) -> Iterator[tuple[Path, Optional[int]]]:
    """Anchor all POSIX member operations to one verified directory descriptor."""

    destination = _absolute_without_resolving(path)
    parent = destination.parent
    if _has_symlink_component(parent):
        raise OSError("Telemetry parent path is unsafe.")
    parent.mkdir(parents=True, exist_ok=True)
    if _has_symlink_component(parent):
        raise OSError("Telemetry parent path is unsafe.")
    before = os.stat(parent, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode) or _is_redirecting(before):
        raise OSError("Telemetry parent must be a non-redirecting directory.")

    if os.name == "nt":  # pragma: no cover - Windows-specific fallback
        try:
            yield destination, None
        finally:
            if _has_symlink_component(parent):
                raise OSError("Telemetry parent changed during publication.")
            after = os.stat(parent, follow_symlinks=False)
            if _identity(after) != _identity(before):
                raise OSError("Telemetry parent identity changed during publication.")
        return

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(parent, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _is_redirecting(opened)
            or not _same_identity(opened, before)
        ):
            raise OSError("Telemetry parent descriptor identity is invalid.")
        current = os.stat(parent, follow_symlinks=False)
        if _is_redirecting(current) or not _same_identity(current, opened):
            raise OSError("Telemetry parent changed before publication.")
        yield destination, descriptor
        current = os.stat(parent, follow_symlinks=False)
        if _is_redirecting(current) or not _same_identity(current, opened):
            raise OSError("Telemetry parent changed during publication.")
    finally:
        os.close(descriptor)


def _member_stat(path: Path, parent_fd: Optional[int]) -> Optional[os.stat_result]:
    try:
        if parent_fd is None:
            return path.lstat()
        return os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _regular_or_missing(path: Path, parent_fd: Optional[int] = None) -> bool:
    try:
        metadata = _member_stat(path, parent_fd)
    except OSError:
        return False
    return metadata is None or _is_regular_nonredirecting(metadata)


def _unlink_member(
    path: Path,
    parent_fd: Optional[int],
    *,
    expected: Optional[os.stat_result] = None,
) -> None:
    current = _member_stat(path, parent_fd)
    if current is None:
        return
    if not _is_regular_nonredirecting(current):
        raise OSError("Telemetry unlink refused a redirected or non-regular path.")
    if expected is not None and not _same_snapshot(current, expected):
        raise OSError("Telemetry unlink refused a replaced path.")
    if parent_fd is None:
        path.unlink()
        return
    os.unlink(path.name, dir_fd=parent_fd)


def _replace_member(
    source: Path,
    destination: Path,
    parent_fd: Optional[int],
    *,
    expected_source: os.stat_result,
    expected_destination: Optional[os.stat_result],
) -> None:
    current_source = _member_stat(source, parent_fd)
    current_destination = _member_stat(destination, parent_fd)
    if (
        current_source is None
        or not _is_regular_nonredirecting(current_source)
        or not _same_snapshot(current_source, expected_source)
    ):
        raise OSError("Telemetry rotation source changed before replacement.")
    if expected_destination is None:
        if current_destination is not None:
            raise OSError("Telemetry rotation destination appeared unexpectedly.")
    elif (
        current_destination is None
        or not _is_regular_nonredirecting(current_destination)
        or not _same_snapshot(current_destination, expected_destination)
    ):
        raise OSError("Telemetry rotation destination changed before replacement.")
    if parent_fd is None:
        source.replace(destination)
    else:
        os.replace(
            source.name,
            destination.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
    published = _member_stat(destination, parent_fd)
    if published is None or not _same_identity(published, expected_source):
        raise OSError("Telemetry rotation publication identity is invalid.")


@contextmanager
def _process_log_lock(path: Path, parent_fd: Optional[int] = None) -> Iterator[None]:
    """Serialize publication and rotation across one identity-stable lock file."""

    lock_path = path.with_name(f".{path.name}.lock")
    before = _member_stat(lock_path, parent_fd)
    if before is not None and not _is_regular_nonredirecting(before):
        raise OSError("Telemetry lock path is unsafe.")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    if parent_fd is None:
        descriptor = os.open(lock_path, flags, 0o600)
    else:
        descriptor = os.open(lock_path.name, flags, 0o600, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        current = _member_stat(lock_path, parent_fd)
        if (
            not _is_regular_nonredirecting(opened)
            or current is None
            or not _is_regular_nonredirecting(current)
            or not _same_identity(current, opened)
            or (before is not None and not _same_identity(before, opened))
        ):
            raise OSError("Telemetry lock path identity is invalid.")
        try:
            os.fchmod(descriptor, 0o600)
        except OSError:
            pass

        def verify_visible_lock() -> None:
            visible = _member_stat(lock_path, parent_fd)
            if (
                visible is None
                or not _is_regular_nonredirecting(visible)
                or not _same_identity(visible, opened)
            ):
                raise OSError("Telemetry lock path changed during publication.")

        if os.name == "nt":  # pragma: no cover
            import msvcrt

            if opened.st_size < 1:
                os.write(descriptor, b"0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            try:
                verify_visible_lock()
                yield
                verify_visible_lock()
            finally:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                verify_visible_lock()
                yield
                verify_visible_lock()
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _rotate(path: Path, parent_fd: Optional[int] = None) -> None:
    metadata = _member_stat(path, parent_fd)
    if metadata is None:
        return
    if not _is_regular_nonredirecting(metadata):
        raise OSError("Telemetry rotation refused a redirected or non-regular path.")
    if LOG_BACKUPS <= 0:
        _unlink_member(path, parent_fd, expected=metadata)
        return

    oldest = _rotated_path(path, LOG_BACKUPS)
    oldest_metadata = _member_stat(oldest, parent_fd)
    if oldest_metadata is not None:
        if not _is_regular_nonredirecting(oldest_metadata):
            raise OSError("Telemetry backup path is not a safe regular file.")
        _unlink_member(oldest, parent_fd, expected=oldest_metadata)

    for index in range(LOG_BACKUPS - 1, 0, -1):
        source = _rotated_path(path, index)
        destination = _rotated_path(path, index + 1)
        source_metadata = _member_stat(source, parent_fd)
        if source_metadata is None:
            continue
        if not _is_regular_nonredirecting(source_metadata):
            raise OSError("Telemetry backup rotation encountered an unsafe path.")
        destination_metadata = _member_stat(destination, parent_fd)
        if (
            destination_metadata is not None
            and not _is_regular_nonredirecting(destination_metadata)
        ):
            raise OSError("Telemetry backup rotation encountered an unsafe path.")
        _replace_member(
            source,
            destination,
            parent_fd,
            expected_source=source_metadata,
            expected_destination=destination_metadata,
        )

    first_backup = _rotated_path(path, 1)
    first_metadata = _member_stat(first_backup, parent_fd)
    if first_metadata is not None and not _is_regular_nonredirecting(first_metadata):
        raise OSError("Telemetry first backup path is unsafe.")
    _replace_member(
        path,
        first_backup,
        parent_fd,
        expected_source=metadata,
        expected_destination=first_metadata,
    )


def _append_line(path: Path, encoded: bytes, parent_fd: Optional[int] = None) -> None:
    if not encoded or len(encoded) > _MAX_EVENT_BYTES:
        raise OSError("Telemetry event exceeds the append limit.")
    before = _member_stat(path, parent_fd)
    if before is not None and not _is_regular_nonredirecting(before):
        raise OSError("Telemetry destination path is unsafe.")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if parent_fd is None:
        descriptor = os.open(path, flags, 0o600)
    else:
        descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        visible = _member_stat(path, parent_fd)
        if (
            not _is_regular_nonredirecting(opened)
            or visible is None
            or not _is_regular_nonredirecting(visible)
            or not _same_identity(visible, opened)
            or (before is not None and not _same_identity(before, opened))
        ):
            raise OSError("Telemetry destination identity is invalid.")
        try:
            os.fchmod(descriptor, 0o600)
        except OSError:
            pass
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("Telemetry append made no progress.")
            offset += written
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        visible = _member_stat(path, parent_fd)
        if (
            visible is None
            or not _is_regular_nonredirecting(visible)
            or not _same_identity(visible, opened)
        ):
            raise OSError("Telemetry destination changed during append.")
    finally:
        os.close(descriptor)


def _encoded_entry(entry: Dict[str, Any]) -> bytes:
    line = json.dumps(
        entry,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    encoded = line.encode("utf-8")
    if len(encoded) <= min(LOG_MAX_BYTES, _MAX_EVENT_BYTES):
        return encoded
    fallback = {
        "timestamp": entry.get("timestamp"),
        "type": entry.get("type"),
        "details": {"telemetry_truncated": True},
    }
    encoded = (
        json.dumps(
            fallback,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return encoded if len(encoded) <= min(LOG_MAX_BYTES, _MAX_EVENT_BYTES) else b""


def log_activity(activity_type: str, details: Dict[str, Any]) -> None:
    """Append one bounded event. Telemetry failure never fails the user request."""

    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": mask_metadata_text(
                _safe_text(activity_type, maximum=100, default="unknown")
            )[:100],
            "details": _json_safe(
                sanitize_metadata_dict(details if isinstance(details, dict) else {})
            ),
        }
        encoded = _encoded_entry(entry)
        if not encoded:
            return
        path = _absolute_without_resolving(LOG_FILE)
        with _LOG_LOCK, _log_directory(path) as (destination, parent_fd):
            with _process_log_lock(destination, parent_fd):
                metadata = _member_stat(destination, parent_fd)
                if metadata is not None and not _is_regular_nonredirecting(metadata):
                    return
                current_size = metadata.st_size if metadata is not None else 0
                if current_size + len(encoded) > LOG_MAX_BYTES:
                    _rotate(destination, parent_fd)
                if not _regular_or_missing(destination, parent_fd):
                    return
                _append_line(destination, encoded, parent_fd)
    except Exception:
        return


def log_tool_call(
    tool_name: str,
    duration: float,
    success: bool,
    tokens: int = 0,
    error_type: str | None = None,
) -> None:
    try:
        log_activity(
            "tool_call",
            {
                "tool": _safe_text(tool_name, maximum=200, default="unknown"),
                "duration_sec": _finite_nonnegative(duration),
                "success": _safe_bool(success),
                "estimated_tokens": _nonnegative_integer(tokens),
                "error_type": (
                    _safe_text(error_type, maximum=200)
                    if error_type is not None
                    else None
                ),
            },
        )
    except Exception:
        return


def log_agent_run(
    query: str,
    total_time: float,
    citation_count: int,
    *,
    success: bool = True,
    owner_id: str | None = None,
) -> None:
    try:
        bounded_query = _safe_text(
            query,
            maximum=_MAX_PRIVATE_HASH_INPUT_CHARS,
        )
        query_bytes = bounded_query.encode("utf-8", errors="replace")
        bounded_owner = (
            _safe_text(owner_id, maximum=500)
            if owner_id is not None
            else ""
        )
        owner_hash = (
            hashlib.sha256(
                bounded_owner.encode("utf-8", errors="replace")
            ).hexdigest()
            if bounded_owner
            else None
        )
        log_activity(
            "agent_run",
            {
                "query_sha256": hashlib.sha256(query_bytes).hexdigest(),
                "query_length": len(bounded_query),
                "duration_sec": _finite_nonnegative(total_time),
                "citations": _nonnegative_integer(citation_count),
                "success": _safe_bool(success, default=True),
                "owner_sha256": owner_hash,
            },
        )
    except Exception:
        return
