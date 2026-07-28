"""Bounded immutable snapshots for parser-facing ingestion work."""

from __future__ import annotations

import os
import stat
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
    """Yield a private parser path containing one anchored upload byte snapshot."""

    payload = read_owner_file(
        upload_root,
        source_path,
        max_bytes=max_bytes,
    )
    if payload is None:
        raise UploadStorageError(
            "The ingestion source was missing, oversized, non-regular, or symlinked."
        )
    try:
        source_name = os.fspath(source_path)
    except TypeError as exc:
        raise UploadStorageError("source_path must be a filesystem path.") from exc
    suffix = safe_upload_suffix(Path(source_name).name)
    descriptor, raw_path = tempfile.mkstemp(
        prefix="rigorousrag-ingest-",
        suffix=suffix,
    )
    snapshot = Path(raw_path)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise UploadStorageError(
                "The immutable ingestion snapshot is not a regular file."
            )
        try:
            os.fchmod(descriptor, 0o600)
        except (AttributeError, OSError):
            pass
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = -1
        if snapshot.is_symlink() or not snapshot.is_file():
            raise UploadStorageError(
                "The immutable ingestion snapshot changed before parser use."
            )
        yield snapshot, payload
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            if not snapshot.is_symlink():
                snapshot.unlink(missing_ok=True)
        except OSError:
            pass
