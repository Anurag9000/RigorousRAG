"""Bounded immutable snapshots for parser-facing ingestion work."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Tuple

from tools.security import DEFAULT_MAX_UPLOAD_BYTES, safe_upload_suffix
from tools.upload_storage import UploadStorageError, read_owner_file


@contextmanager
def materialize_ingestion_snapshot(
    *,
    upload_root: str | Path,
    source_path: str | Path,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> Iterator[Tuple[Path, bytes]]:
    """Yield a private parser path containing one anchored upload byte snapshot.

    The retained owner path is opened with the descriptor-relative no-follow reader.
    Parsers then operate on a private `0600` temporary file containing those exact
    bytes, so a later owner-directory replacement cannot redirect parser input.
    """

    limit = int(max_bytes)
    if limit <= 0:
        raise UploadStorageError("max_bytes must be positive.")
    payload = read_owner_file(upload_root, source_path, max_bytes=limit)
    if payload is None:
        raise UploadStorageError(
            "The ingestion source was missing, oversized, non-regular, or symlinked."
        )
    suffix = safe_upload_suffix(Path(source_path).name)
    descriptor, raw_path = tempfile.mkstemp(
        prefix="rigorousrag-ingest-",
        suffix=suffix,
    )
    snapshot = Path(raw_path)
    try:
        try:
            os.fchmod(descriptor, 0o600)
        except (AttributeError, OSError):
            pass
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        yield snapshot, payload
    finally:
        try:
            if not snapshot.is_symlink():
                snapshot.unlink(missing_ok=True)
        except OSError:
            pass
