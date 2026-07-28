"""Failure-isolated, privacy-conscious bounded JSONL telemetry."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from tools.privacy import mask_metadata_text, sanitize_metadata_dict


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
_LOG_LOCK = threading.Lock()


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return "[TRUNCATED_DEPTH]"
    if isinstance(value, str):
        return value[:4000]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 100:
                result["__truncated_items__"] = True
                break
            result[str(key)[:200]] = _json_safe(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth=depth + 1) for item in value[:100]]
    return mask_metadata_text(repr(value))[:1000]


def _finite_nonnegative(value: Any, *, digits: int = 3) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return round(max(numeric, 0.0), digits)


def _nonnegative_integer(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _absolute_without_resolving(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _has_symlink_component(path: Path) -> bool:
    candidate = _absolute_without_resolving(path)
    for component in (candidate, *candidate.parents):
        try:
            if component.is_symlink():
                return True
        except OSError:
            return True
    return False


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


def _append_line(path: Path, line: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("Telemetry destination must be a regular file.")
        try:
            os.fchmod(descriptor, 0o600)
        except OSError:
            pass
        with os.fdopen(descriptor, "a", encoding="utf-8", closefd=False) as handle:
            handle.write(line)
            handle.flush()
    finally:
        os.close(descriptor)


def log_activity(activity_type: str, details: Dict[str, Any]) -> None:
    """Append one bounded event. Telemetry failure never fails the user request."""

    sanitized_details = sanitize_metadata_dict(details if isinstance(details, dict) else {})
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": mask_metadata_text(str(activity_type))[:100],
        "details": _json_safe(sanitized_details),
    }
    try:
        path = _absolute_without_resolving(LOG_FILE)
        if _has_symlink_component(path.parent):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if _has_symlink_component(path.parent):
            return
        line = json.dumps(
            entry,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
        encoded_length = len(line.encode("utf-8"))
        if encoded_length > LOG_MAX_BYTES:
            entry["details"] = {"telemetry_truncated": True}
            line = json.dumps(
                entry,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ) + "\n"
            encoded_length = len(line.encode("utf-8"))
        with _LOG_LOCK:
            if path.is_symlink() or not _regular_or_missing(path):
                return
            current_size = path.stat().st_size if path.exists() else 0
            if current_size + encoded_length > LOG_MAX_BYTES:
                _rotate(path)
            if path.is_symlink() or not _regular_or_missing(path):
                return
            _append_line(path, line)
    except Exception:
        return


def log_tool_call(
    tool_name: str,
    duration: float,
    success: bool,
    tokens: int = 0,
    error_type: str | None = None,
) -> None:
    log_activity(
        "tool_call",
        {
            "tool": str(tool_name)[:200],
            "duration_sec": _finite_nonnegative(duration),
            "success": bool(success),
            "estimated_tokens": _nonnegative_integer(tokens),
            "error_type": str(error_type)[:200] if error_type else None,
        },
    )


def log_agent_run(
    query: str,
    total_time: float,
    citation_count: int,
    *,
    success: bool = True,
    owner_id: str | None = None,
) -> None:
    query_bytes = (query or "").encode("utf-8")
    owner_hash = (
        hashlib.sha256(owner_id.encode("utf-8")).hexdigest()
        if owner_id
        else None
    )
    log_activity(
        "agent_run",
        {
            "query_sha256": hashlib.sha256(query_bytes).hexdigest(),
            "query_length": len(query or ""),
            "duration_sec": _finite_nonnegative(total_time),
            "citations": _nonnegative_integer(citation_count),
            "success": bool(success),
            "owner_sha256": owner_hash,
        },
    )
