"""Stable immutable public boundary over the legacy document parser.

``ingestion_legacy`` continues to own format parsing and OCR.  This module remains a
real module instead of replacing itself in ``sys.modules``.  Public monkeypatches and
runtime configuration are forwarded explicitly before each parse, and every successful
legacy result is reconstructed as a new privacy-safe document from a bounded no-follow
source snapshot.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import uuid
from pathlib import Path
from typing import Any

from tools.config import bounded_float_env, bounded_int_env
from tools.ingestion_models import (
    DocumentSection,
    IngestedDocument,
    IngestionResult,
)
from tools.privacy import mask_metadata_text, sanitize_metadata_dict
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
_MAX_SECTIONS = 10_000

# Re-export the complete legacy helper surface without replacing this module object.
# This keeps existing imports and test overrides compatible while ensuring that every
# caller reaches the wrapper ``ingest_file`` below.
_FORWARDED_NAMES = tuple(
    name
    for name in dir(_implementation)
    if not name.startswith("__") and name != "ingest_file"
)
for _forwarded_name in _FORWARDED_NAMES:
    globals().setdefault(
        _forwarded_name,
        getattr(_implementation, _forwarded_name),
    )


def __getattr__(name: str) -> Any:
    """Expose future legacy helpers without changing module identity."""

    try:
        return getattr(_implementation, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def _sync_legacy_runtime() -> None:
    """Forward public overrides to the parser immediately before one parse."""

    for name in _FORWARDED_NAMES:
        if name in globals():
            setattr(_implementation, name, globals()[name])
    # Always bind parser model construction to the current privacy-safe classes.
    _implementation.DocumentSection = DocumentSection
    _implementation.IngestedDocument = IngestedDocument
    _implementation.IngestionResult = IngestionResult


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
            raise ValueError("Input paths may not contain symbolic-link components.")
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


def _redacted_sections(sections: Any) -> list[DocumentSection]:
    if sections is None or isinstance(sections, (str, bytes, bytearray)):
        return []
    try:
        values = list(sections)
    except Exception:
        return []
    if len(values) > _MAX_SECTIONS:
        values = values[:_MAX_SECTIONS]

    chunker = getattr(_implementation, "_chunk_text_semantically", None)
    result: list[DocumentSection] = []
    for index, section in enumerate(values):
        try:
            raw_content = section.content
            raw_title = section.title
            page_number = section.page_number
        except Exception:
            continue
        if not isinstance(raw_content, str):
            continue
        content = mask_metadata_text(raw_content).strip()
        if not content:
            continue
        if callable(chunker):
            try:
                chunks = chunker(content, max_chars=6000)
            except Exception:
                chunks = []
        else:
            chunks = []
        chunks = chunks or [
            content[position:position + 6000]
            for position in range(0, len(content), 6000)
        ]
        base_title = mask_metadata_text(
            raw_title or f"Section {index + 1}"
        ).strip() or f"Section {index + 1}"
        for chunk_index, chunk in enumerate(chunks):
            masked_chunk = mask_metadata_text(chunk).strip()
            if not masked_chunk:
                continue
            title = base_title
            if len(chunks) > 1:
                title = f"{title} — Part {chunk_index + 1}"
            result.append(
                DocumentSection(
                    title=title[:500],
                    content=masked_chunk,
                    page_number=page_number,
                )
            )
    return result


def _finalize_public_result(
    result: Any,
    *,
    owner: str,
    source: Path,
    payload: bytes,
) -> IngestionResult:
    """Reconstruct, rather than mutate, the parser result at the public boundary."""

    try:
        success = result.success
        legacy_document = result.document
        legacy_error = result.error
    except Exception:
        return IngestionResult(
            success=False,
            error="Document parser returned an invalid result.",
        )
    if not success or legacy_document is None:
        return IngestionResult(
            success=False,
            error=legacy_error or "Document ingestion failed.",
        )

    try:
        redacted_text = mask_metadata_text(legacy_document.text).strip()
        redacted_sections = _redacted_sections(legacy_document.sections)
        if not redacted_text or not redacted_sections:
            return IngestionResult(
                success=False,
                error="No indexable text remained after parsing.",
            )
        source_hash = hashlib.sha256(payload).hexdigest()
        content_hash = hashlib.sha256(redacted_text.encode("utf-8")).hexdigest()
        extracted = _implementation.extract_academic_metadata(redacted_text)
        fallback_title = source.stem.replace("_", " ").replace("-", " ")
        title = mask_metadata_text(
            getattr(legacy_document, "title", None)
            or extracted.get("extracted_title")
            or fallback_title
        ).strip() or fallback_title
        metadata = sanitize_metadata_dict(
            getattr(legacy_document, "metadata", {})
        )
        metadata.update(sanitize_metadata_dict(extracted))
        metadata.update(
            {
                "owner_id": owner,
                "content_sha256": content_hash,
                "file_size_bytes": len(payload),
                "redaction": "best_effort_regex_masking",
                "document_identity": "owner_and_source_sha256",
            }
        )
        document = IngestedDocument(
            id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"rigorousrag:{owner}:{source_hash}",
                )
            ),
            filename=source.name,
            file_path=str(source),
            mime_type=getattr(
                legacy_document,
                "mime_type",
                _implementation.detect_mime_type(str(source)),
            ),
            created_at=getattr(legacy_document, "created_at", None),
            title=title[:1000],
            text=redacted_text,
            sections=redacted_sections,
            metadata=metadata,
        )
        return IngestionResult(success=True, document=document)
    except Exception as exc:
        return IngestionResult(
            success=False,
            error=f"Document privacy finalization failed ({type(exc).__name__}).",
        )


def ingest_file(
    file_path: str | os.PathLike[str],
    owner_id: str = "default_user",
) -> IngestionResult:
    """Parse one immutable source snapshot without reopening the caller path."""

    try:
        owner = normalize_owner_id(owner_id)
        source = _source_path(file_path)
        suffix = safe_upload_suffix(source.name)
        maximum = int(getattr(_implementation, "DEFAULT_MAX_UPLOAD_BYTES"))
        payload = _read_source_bytes(source, maximum)
    except (SecurityError, ValueError) as exc:
        return IngestionResult(success=False, error=str(exc))
    except Exception as exc:
        return IngestionResult(
            success=False,
            error=f"Input validation failed ({type(exc).__name__}).",
        )

    _sync_legacy_runtime()
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

    return _finalize_public_result(
        result,
        owner=owner,
        source=source,
        payload=payload,
    )
