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
from urllib.parse import urlparse

_MAX_HTTP_BYTES = 64 * 1024
_MAX_PATH_CHARS = 4096
_MAX_HEALTH_URL_CHARS = 4096
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "localhost.localdomain"}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _lexical_absolute(value: Any) -> Optional[Path]:
    if not isinstance(value, (str, os.PathLike)):
        return None
    try:
        rendered = os.fspath(value)
    except TypeError:
        return None
    if not rendered or len(rendered) > _MAX_PATH_CHARS or "\x00" in rendered:
        return None
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _has_symlink_component(path: Path) -> bool:
    try:
        return any(candidate.is_symlink() for candidate in (path, *path.parents))
    except OSError:
        return True


def _finite_timeout(value: object, default: float = 3.0) -> float:
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
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
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


def check_sqlite(path: str | Path) -> bool:
    database = _lexical_absolute(path)
    if database is None or _has_symlink_component(database):
        return False
    try:
        before = database.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            return False
        uri = f"file:{database.as_posix()}?mode=rw"
        with sqlite3.connect(uri, uri=True, timeout=2) as connection:
            row = connection.execute("SELECT 1").fetchone()
        after = database.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(after.st_mode)
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        ):
            return False
        return bool(row and row[0] == 1)
    except (sqlite3.Error, OSError, ValueError):
        return False


def _check_writable_directory_posix(directory: Path) -> bool:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(directory, flags)
    filename = f".health-{uuid.uuid4().hex}"
    descriptor = -1
    try:
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
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
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return False
        os.write(descriptor, b"ok")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.unlink(filename, dir_fd=directory_fd)
        return True
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                os.unlink(filename, dir_fd=directory_fd)
            except OSError:
                pass
        os.close(directory_fd)


def _check_writable_directory_portable(directory: Path) -> bool:
    before = directory.stat()
    probe_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix=".health-", dir=directory)
        probe_path = Path(raw_path)
        with os.fdopen(descriptor, "wb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return False
            handle.write(b"ok")
            handle.flush()
            os.fsync(handle.fileno())
        after = directory.stat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            return False
        probe_path.unlink()
        return True
    finally:
        if probe_path is not None:
            try:
                if not probe_path.is_symlink():
                    probe_path.unlink(missing_ok=True)
            except OSError:
                pass


def check_writable_directory(path: str | Path) -> bool:
    directory = _lexical_absolute(path)
    if directory is None or _has_symlink_component(directory):
        return False
    try:
        if not directory.exists() or not directory.is_dir():
            return False
        if os.name == "nt":  # pragma: no cover
            return _check_writable_directory_portable(directory)
        return _check_writable_directory_posix(directory)
    except (OSError, ValueError):
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
