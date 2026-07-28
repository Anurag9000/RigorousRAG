"""Dependency-light process and persistence readiness checks.

The probe intentionally does not import ``server`` or initialize the embedding model.
It verifies a small loopback HTTP response, both SQLite registries, and runtime
storage directories through create/fsync/delete cycles.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse

_MAX_HTTP_BYTES = 64 * 1024
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "localhost.localdomain"}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def check_http(url: str, timeout: float = 3.0) -> bool:
    parsed = urlparse(str(url or "").strip())
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if parsed.scheme.lower() not in {"http", "https"} or hostname not in _LOCAL_HOSTS:
        return False
    if parsed.username or parsed.password:
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
        with opener.open(request, timeout=max(0.1, float(timeout))) as response:
            if response.status != 200:
                return False
            raw = response.read(_MAX_HTTP_BYTES + 1)
        if len(raw) > _MAX_HTTP_BYTES:
            return False
        payload = json.loads(raw.decode("utf-8"))
        return isinstance(payload, dict) and payload.get("status") == "ok"
    except Exception:
        return False


def check_sqlite(path: str | Path) -> bool:
    raw_database = Path(path)
    if _has_symlink_component(raw_database):
        return False
    database = raw_database.resolve()
    if not database.exists() or not database.is_file():
        return False
    try:
        uri = f"file:{database.as_posix()}?mode=rw"
        with sqlite3.connect(uri, uri=True, timeout=2) as connection:
            row = connection.execute("SELECT 1").fetchone()
        return bool(row and row[0] == 1)
    except sqlite3.Error:
        return False


def check_writable_directory(path: str | Path) -> bool:
    raw_directory = Path(path)
    if _has_symlink_component(raw_directory):
        return False
    directory = raw_directory.resolve()
    if not directory.exists() or not directory.is_dir():
        return False
    probe_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix=".health-", dir=directory)
        probe_path = Path(raw_path)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"ok")
            handle.flush()
            os.fsync(handle.fileno())
        probe_path.unlink()
        return True
    except OSError:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)
        return False


def run_checks() -> Dict[str, bool]:
    return {
        "http": check_http(
            os.getenv("HEALTHCHECK_URL", "http://127.0.0.1:8000/health"),
            timeout=float(os.getenv("HEALTHCHECK_TIMEOUT_SECONDS", "3")),
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
