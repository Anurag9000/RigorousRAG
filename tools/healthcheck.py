"""Dependency-light process and persistence readiness checks.

The probe intentionally does not import ``server`` or initialize the embedding model.
It verifies a small loopback HTTP response, both SQLite registries, and runtime
storage directories through bounded create/fsync/delete cycles.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import stat
import tempfile
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote, urlparse

_MAX_HTTP_BYTES = 64 * 1024
_MAX_PATH_CHARS = 4096
_MAX_HEALTH_URL_CHARS = 4096
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "localhost.localdomain"}
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_link_or_reparse(info: os.stat_result) -> bool:
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _identity(info: os.stat_result) -> tuple[int, int]:
    return int(info.st_dev), int(info.st_ino)


def _lexical_absolute(value: Any) -> Optional[Path]:
    if not isinstance(value, (str, os.PathLike)):
        return None
    try:
        rendered = os.fspath(value)
    except Exception:
        return None
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH_CHARS
        or _contains_ascii_control(rendered)
    ):
        return None
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        return Path(os.path.abspath(candidate))
    except Exception:
        return None


def _has_redirected_component(path: Path) -> bool:
    try:
        for candidate in (path, *path.parents):
            try:
                info = os.lstat(candidate)
            except FileNotFoundError:
                continue
            if _is_link_or_reparse(info):
                return True
        return False
    except Exception:
        return True


def _finite_timeout(value: object, default: float = 3.0) -> float:
    if isinstance(value, bool):
        parsed = default
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            parsed = default
    if not math.isfinite(parsed) or parsed <= 0:
        parsed = default
    return max(0.1, min(parsed, 60.0))


def check_http(url: str, timeout: float = 3.0) -> bool:
    if not isinstance(url, str):
        return False
    rendered = url.strip()
    if (
        not rendered
        or len(rendered) > _MAX_HEALTH_URL_CHARS
        or _contains_ascii_control(rendered)
    ):
        return False
    try:
        parsed = urlparse(rendered)
        hostname = (parsed.hostname or "").rstrip(".").lower()
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme.lower() not in {"http", "https"} or hostname not in _LOCAL_HOSTS:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if port is not None and not 1 <= port <= 65_535:
        return False
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
        )
        request = urllib.request.Request(
            parsed._replace(fragment="").geturl(),
            headers={"Accept": "application/json"},
            method="GET",
        )
        with opener.open(request, timeout=_finite_timeout(timeout)) as response:
            if int(response.status) != 200:
                return False
            raw = response.read(_MAX_HTTP_BYTES + 1)
        if not isinstance(raw, bytes) or len(raw) > _MAX_HTTP_BYTES:
            return False
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-standard JSON constant {value}")
            ),
        )
        return isinstance(payload, dict) and payload.get("status") == "ok"
    except Exception:
        return False


def _sqlite_uri(database: Path) -> str:
    encoded = quote(database.as_posix(), safe="/:")
    return f"file:{encoded}?mode=rw"


def check_sqlite(path: str | Path) -> bool:
    database = _lexical_absolute(path)
    if database is None or _has_redirected_component(database):
        return False
    try:
        before = os.lstat(database)
        parent_before = os.lstat(database.parent)
        if (
            not stat.S_ISREG(before.st_mode)
            or _is_link_or_reparse(before)
            or not stat.S_ISDIR(parent_before.st_mode)
            or _is_link_or_reparse(parent_before)
        ):
            return False
        with sqlite3.connect(_sqlite_uri(database), uri=True, timeout=2) as connection:
            connection.execute("PRAGMA query_only=ON")
            row = connection.execute("SELECT 1").fetchone()
        if _has_redirected_component(database):
            return False
        after = os.lstat(database)
        parent_after = os.lstat(database.parent)
        if (
            not stat.S_ISREG(after.st_mode)
            or _is_link_or_reparse(after)
            or _identity(before) != _identity(after)
            or not stat.S_ISDIR(parent_after.st_mode)
            or _is_link_or_reparse(parent_after)
            or _identity(parent_before) != _identity(parent_after)
        ):
            return False
        return bool(row and row[0] == 1)
    except Exception:
        return False


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("Readiness probe write made no progress.")
        offset += written


def _check_writable_directory_posix(directory: Path) -> bool:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(directory, flags)
    filename = f".health-{uuid.uuid4().hex}"
    descriptor = -1
    named_entry_exists = False
    descriptor_identity: tuple[int, int] | None = None
    try:
        directory_info = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_info.st_mode):
            return False
        descriptor = os.open(
            filename,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        named_entry_exists = True
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            return False
        descriptor_identity = _identity(opened)

        # Remove the name while the original descriptor is still open. This tests
        # directory deletion permission and prevents a later cleanup from unlinking a
        # replacement entry supplied by another process.
        entry = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if _is_link_or_reparse(entry) or _identity(entry) != descriptor_identity:
            return False
        os.unlink(filename, dir_fd=directory_fd)
        named_entry_exists = False

        _write_all(descriptor, b"ok")
        os.fsync(descriptor)
        os.fsync(directory_fd)
        return True
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if named_entry_exists and descriptor_identity is not None:
            try:
                entry = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
                if (
                    not _is_link_or_reparse(entry)
                    and _identity(entry) == descriptor_identity
                ):
                    os.unlink(filename, dir_fd=directory_fd)
            except OSError:
                pass
        os.close(directory_fd)


def _check_writable_directory_portable(directory: Path) -> bool:
    before = os.lstat(directory)
    try:
        with tempfile.TemporaryFile(prefix=".health-", dir=directory) as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                return False
            handle.write(b"ok")
            handle.flush()
            os.fsync(handle.fileno())
        after = os.lstat(directory)
        return (
            stat.S_ISDIR(after.st_mode)
            and not _is_link_or_reparse(after)
            and _identity(before) == _identity(after)
        )
    except Exception:
        return False


def check_writable_directory(path: str | Path) -> bool:
    directory = _lexical_absolute(path)
    if directory is None or _has_redirected_component(directory):
        return False
    try:
        info = os.lstat(directory)
        if not stat.S_ISDIR(info.st_mode) or _is_link_or_reparse(info):
            return False
        if os.name == "nt":  # pragma: no cover
            return _check_writable_directory_portable(directory)
        return _check_writable_directory_posix(directory)
    except Exception:
        return False


def run_checks() -> Dict[str, bool]:
    timeout = _finite_timeout(os.getenv("HEALTHCHECK_TIMEOUT_SECONDS", "3"))
    return {
        "http": check_http(
            os.getenv("HEALTHCHECK_URL", "http://127.0.0.1:8000/health"),
            timeout=timeout,
        ),
        "jobs": check_sqlite(os.getenv("JOB_DB_PATH", "data/jobs.sqlite3")),
        "documents": check_sqlite(
            os.getenv("DOCUMENT_DB_PATH", "data/documents.sqlite3")
        ),
        "uploads": check_writable_directory(os.getenv("UPLOAD_DIR", "uploads")),
        "vectors": check_writable_directory(os.getenv("CHROMA_PATH", "rag_storage")),
    }


def main() -> int:
    results = run_checks()
    ready = all(results.values())
    print(json.dumps({"status": "ready" if ready else "not_ready", "checks": results}))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
