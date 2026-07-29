"""Failure-isolated, privacy-conscious bounded JSONL telemetry."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator

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
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(numeric, _MAX_PUBLIC_INTEGER))


def _absolute_without_resolving(path: Any) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError("Telemetry path must be a filesystem path.")
    rendered = os.fspath(path)
    if not rendered or len(rendered) > _MAX_PATH_CHARS or "\x00" in rendered:
        raise ValueError("Telemetry path is invalid or too long.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _has_symlink_component(path: Path) -> bool:
    try:
        candidate = _absolute_without_resolving(path)
        return any(component.is_symlink() for component in (candidate, *candidate.parents))
    except (OSError, ValueError):
        return True


def _rotated_path(path: Path, index: int) -> Path:
    return path.with_name(f"{path.name}.{index}")


def _regular_or_missing(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode)


@contextmanager
def _process_log_lock(path: Path) -> Iterator[None]:
    """Serialize publication and rotation across service processes."""

    lock_path = path.with_name(f".{path.name}.lock")
    if lock_path.is_symlink() or not _regular_or_missing(lock_path):
        raise OSError("Telemetry lock path is unsafe.")
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("Telemetry lock path must be a regular file.")
        try:
            os.fchmod(descriptor, 0o600)
        except OSError:
            pass
        if os.name == "nt":  # pragma: no cover
            import msvcrt

            if os.fstat(descriptor).st_size < 1:
                os.write(descriptor, b"0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _rotate(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not _regular_or_missing(path):
        raise OSError("Telemetry rotation refused a non-regular path.")
    if LOG_BACKUPS <= 0:
        path.unlink(missing_ok=True)
        return
    oldest = _rotated_path(path, LOG_BACKUPS)
    if oldest.is_symlink() or _regular_or_missing(oldest):
        oldest.unlink(missing_ok=True)
    else:
        raise OSError("Telemetry backup path is not a regular file.")
    for index in range(LOG_BACKUPS - 1, 0, -1):
        source = _rotated_path(path, index)
        destination = _rotated_path(path, index + 1)
        if source.is_symlink():
            source.unlink(missing_ok=True)
            continue
        if source.exists():
            if not _regular_or_missing(source) or destination.is_symlink():
                raise OSError("Telemetry backup rotation encountered an unsafe path.")
            source.replace(destination)
    path.replace(_rotated_path(path, 1))


def _append_line(path: Path, encoded: bytes) -> None:
    if not encoded or len(encoded) > _MAX_EVENT_BYTES:
        raise OSError("Telemetry event exceeds the append limit.")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("Telemetry destination must be a regular file.")
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
        if _has_symlink_component(path.parent):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if _has_symlink_component(path.parent):
            return
        with _LOG_LOCK, _process_log_lock(path):
            if path.is_symlink() or not _regular_or_missing(path):
                return
            current_size = path.stat().st_size if path.exists() else 0
            if current_size + len(encoded) > LOG_MAX_BYTES:
                _rotate(path)
            if path.is_symlink() or not _regular_or_missing(path):
                return
            _append_line(path, encoded)
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
        bounded_owner = _safe_text(owner_id, maximum=500) if owner_id is not None else ""
        owner_hash = (
            hashlib.sha256(bounded_owner.encode("utf-8", errors="replace")).hexdigest()
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
