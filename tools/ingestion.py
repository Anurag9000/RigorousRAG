"""Failure-safe, immutable public boundary over document ingestion.

The parser, OCR, redaction, and archive implementation remains in
``ingestion_legacy``. This module normalizes parser budgets and consumes every direct
source through a bounded no-follow byte snapshot before invoking that implementation.
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from tools.config import bounded_float_env, bounded_int_env
from tools.ingestion_models import IngestionResult
from tools.security import SecurityError, normalize_owner_id, safe_upload_suffix

_INTEGER_BUDGETS = {
    "MAX_UPLOAD_BYTES": (50_000_000, 1, 1_000_000_000),
    "OCR_MAX_PAGES": (50, 1, 500),
    "OCR_DPI": (200, 100, 400),
    "OCR_TIMEOUT_SECONDS": (30, 1, 300),
    "OCR_MIN_TEXT_CHARS": (40, 0, 2000),
    "MAX_PDF_PAGES": (2000, 1, 10_000),
    "MAX_PDF_RENDER_PIXELS": (40_000_000, 1_000_000, 250_000_000),
    "MAX_EXTRACTED_CHARS": (5_000_000, 100_000, 50_000_000),
    "MAX_DOCX_MEMBERS": (10_000, 10, 100_000),
    "MAX_DOCX_UNCOMPRESSED_BYTES": (200_000_000, 1, 2_000_000_000),
}
for _name, (_default, _minimum, _maximum) in _INTEGER_BUDGETS.items():
    bounded_int_env(
        _name,
        _default,
        minimum=_minimum,
        maximum=_maximum,
        write_back=True,
    )
bounded_float_env(
    "MAX_DOCX_COMPRESSION_RATIO",
    1000.0,
    minimum=10.0,
    maximum=100_000.0,
    write_back=True,
)

from tools import ingestion_legacy as _implementation

_original_ingest_file = _implementation.ingest_file
_MAX_PATH_CHARS = 4096


def _source_path(value: Any) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("file_path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not rendered
        or len(rendered) > _MAX_PATH_CHARS
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("file_path is invalid or too long.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(
                "Input paths may not contain symbolic-link components."
            )
    return absolute


def _read_source_bytes(path: Path, maximum: int) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("The input must be a regular file.")
        if metadata.st_size <= 0:
            raise ValueError("The input file is empty.")
        if metadata.st_size > maximum:
            raise ValueError(
                f"The input file exceeds the {maximum}-byte upload limit."
            )
        payload = bytearray()
        while True:
            remaining = maximum + 1 - len(payload)
            if remaining <= 0:
                raise ValueError(
                    f"The input file exceeds the {maximum}-byte upload limit."
                )
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            payload.extend(chunk)
        if not payload:
            raise ValueError("The input file is empty.")
        if len(payload) > maximum:
            raise ValueError(
                f"The input file exceeds the {maximum}-byte upload limit."
            )
        return bytes(payload)
    finally:
        os.close(descriptor)


def ingest_file(
    file_path: str | os.PathLike[str],
    owner_id: str = "default_user",
) -> IngestionResult:
    """Parse one immutable source snapshot without reopening the caller's path."""

    try:
        owner = normalize_owner_id(owner_id)
        source = _source_path(file_path)
        suffix = safe_upload_suffix(source.name)
        payload = _read_source_bytes(
            source,
            _implementation.DEFAULT_MAX_UPLOAD_BYTES,
        )
    except (SecurityError, ValueError) as exc:
        return IngestionResult(success=False, error=str(exc))
    except Exception as exc:
        return IngestionResult(
            success=False,
            error=f"Input validation failed ({type(exc).__name__}).",
        )

    with tempfile.TemporaryDirectory(prefix="rigorousrag-parser-") as directory:
        snapshot = Path(directory) / f"source{suffix}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(snapshot, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                return IngestionResult(
                    success=False,
                    error="The parser snapshot was not a regular file.",
                )
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        result = _original_ingest_file(str(snapshot), owner_id=owner)

    if result.success and result.document is not None:
        result.document.filename = source.name
        result.document.file_path = str(source)
    return result


_implementation.ingest_file = ingest_file
_implementation.__doc__ = __doc__
sys.modules[__name__] = _implementation
