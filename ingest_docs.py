"""Batch document ingestion through the shared authoritative indexing lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from tools.authoritative_document_index import (
    AuthoritativeDocumentSnapshot,
    capture_authoritative_document,
    restore_authoritative_document,
)
from tools.document_service import IndexedDocument, index_document
from tools.document_store import get_document_store
from tools.ingestion import ingest_file
from tools.privacy import mask_metadata_text
from tools.rag import get_rag_layer
from tools.security import normalize_owner_id
from tools.vector_generation import (
    VectorGenerationSnapshot,
    capture_vector_generation,
)

_ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}
_MAX_INPUT_FILES = 10_000
_MAX_PATH_CHARS = 4_096
_MAX_MANIFEST_BYTES = 50_000_000
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
# Compatibility limits for callers that still inspect a prior vector generation
# through this batch-ingestion module. The authoritative capture function remains
# the single backend reader and enforces the same owner/document query boundary.
_MAX_VECTOR_ROWS = 100_000
_MAX_VECTOR_TEXT_CHARS = 5_000_000
_MAX_VECTOR_METADATA_ITEMS = 1_000



def _capture_generation(
    rag: Any,
    owner_id: str,
    doc_id: str,
) -> VectorGenerationSnapshot:
    """Compatibility wrapper over authoritative bounded vector capture.

    Public callers historically used this helper during batch replacement. It now
    delegates the backend request to ``capture_vector_generation`` and only adds
    caller-configurable defensive limits for already captured row payloads.
    """

    try:
        snapshot = capture_vector_generation(
            rag,
            owner_id=owner_id,
            doc_id=doc_id,
        )
    except ValueError as exc:
        message = str(exc)
        if "row ID" in message or "vector row ID" in message:
            raise RuntimeError(
                "Prior vector generation contains an invalid ID."
            ) from exc
        raise RuntimeError(
            "Prior vector generation contains invalid rows."
        ) from exc
    if snapshot.row_count > _MAX_VECTOR_ROWS:
        raise RuntimeError("Prior vector generation contains invalid rows.")
    for text, metadata in zip(snapshot.documents, snapshot.metadatas):
        if len(text) > _MAX_VECTOR_TEXT_CHARS or len(metadata) > _MAX_VECTOR_METADATA_ITEMS:
            raise RuntimeError("Prior vector generation contains invalid rows.")
    return snapshot

def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _redirecting(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        int(getattr(metadata, "st_file_attributes", 0))
        & _WINDOWS_REPARSE_POINT
    )


def _lexical_absolute(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("Paths must be strings or path-like values.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH_CHARS
        or _contains_ascii_control(rendered)
    ):
        raise ValueError("A path is invalid or too long.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _has_redirecting_component(path: Path) -> bool:
    absolute = _lexical_absolute(path)
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            return True
        if _redirecting(metadata):
            return True
    return False


def _has_symlink_component(path: Path) -> bool:
    """Compatibility alias retained for existing callers and tests."""
    return _has_redirecting_component(path)


def _regular_supported_file(path: Path) -> bool:
    try:
        if _has_redirecting_component(path):
            return False
        info = path.lstat()
    except (OSError, ValueError):
        return False
    return stat.S_ISREG(info.st_mode) and path.suffix.lower() in _ALLOWED_SUFFIXES


def _directory_files(directory: Path, recursive: bool) -> Iterator[Path]:
    stack = [directory]
    inspected = 0
    inspection_limit = max(_MAX_INPUT_FILES * 20, _MAX_INPUT_FILES)
    while stack:
        current = stack.pop()
        try:
            if _has_redirecting_component(current):
                continue
            scanner = os.scandir(current)
        except (OSError, ValueError):
            continue
        try:
            with scanner as entries:
                for entry in entries:
                    inspected += 1
                    if inspected > inspection_limit:
                        raise ValueError(
                            "Directory traversal exceeded the bounded inspection limit."
                        )
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_file(follow_symlinks=False):
                            yield _lexical_absolute(entry.path)
                        elif recursive and entry.is_dir(follow_symlinks=False):
                            stack.append(_lexical_absolute(entry.path))
                    except (OSError, ValueError):
                        continue
        except OSError:
            continue


def _collect_files(
    paths: list[str],
    recursive: bool,
    output_path: Optional[Path],
) -> list[Path]:
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise ValueError("paths must be a list of strings.")
    if len(paths) > _MAX_INPUT_FILES:
        raise ValueError(f"At most {_MAX_INPUT_FILES} input paths may be supplied.")
    if not isinstance(recursive, bool):
        raise ValueError("recursive must be a boolean.")
    excluded = _lexical_absolute(output_path) if output_path is not None else None
    collected: dict[str, Path] = {}
    for raw in paths:
        try:
            path = _lexical_absolute(raw)
        except ValueError:
            continue
        if _regular_supported_file(path):
            candidates: Iterable[Path] = (path,)
        else:
            try:
                info = path.lstat()
            except OSError:
                continue
            if not stat.S_ISDIR(info.st_mode) or _has_redirecting_component(path):
                continue
            candidates = _directory_files(path, recursive)
        for candidate in candidates:
            if not _regular_supported_file(candidate):
                continue
            if excluded is not None and candidate == excluded:
                continue
            collected[str(candidate)] = candidate
            if len(collected) > _MAX_INPUT_FILES:
                raise ValueError(
                    f"At most {_MAX_INPUT_FILES} supported files may be ingested."
                )
    return [collected[key] for key in sorted(collected)]


def _provider_value(name: str) -> Optional[str]:
    if not isinstance(name, str) or not name or len(name) > 200:
        raise ValueError("Provider setting names must be valid strings.")
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    if (
        not isinstance(raw, str)
        or raw != raw.strip()
        or len(raw) > 4_096
        or _contains_ascii_control(raw)
    ):
        raise ValueError(f"{name} is invalid.")
    return raw


def _llm_client() -> Optional[Any]:
    if OpenAI is None:
        return None
    api_key = _provider_value("OPENAI_API_KEY")
    base_url = _provider_value("OPENAI_BASE_URL")
    if not api_key and not base_url:
        return None
    return OpenAI(
        api_key=api_key or "local-no-key",
        base_url=base_url,
        timeout=60,
        max_retries=2,
    )


def _atomic_manifest(path: Path, manifest: list[dict[str, Any]]) -> None:
    destination = _lexical_absolute(path)
    if _has_redirecting_component(destination.parent):
        raise ValueError("The manifest parent path may not contain redirects.")
    if not destination.parent.exists() or not destination.parent.is_dir():
        raise ValueError("The manifest parent directory does not exist.")
    payload = json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise ValueError("The ingestion manifest exceeds the byte limit.")
    descriptor = -1
    temporary: Optional[Path] = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(raw_path)
        try:
            os.fchmod(descriptor, 0o600)
        except OSError:
            pass
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, destination)
        temporary = None
        try:
            parent_descriptor = os.open(
                destination.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        except OSError:
            pass
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse, redact, and index PDF, DOCX, Markdown, and text files."
    )
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest.")
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument("-o", "--output", default="ingestion_manifest.json")
    parser.add_argument(
        "--owner-id",
        default=os.getenv("SINGLE_USER_OWNER_ID", "default_user"),
    )
    parser.add_argument("--retain-sources", action="store_true")
    parser.add_argument("--include-redacted-text", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _strict_flags(args: argparse.Namespace) -> None:
    for name in (
        "recursive",
        "retain_sources",
        "include_redacted_text",
        "fail_fast",
    ):
        if not isinstance(getattr(args, name, None), bool):
            raise ValueError(f"{name} must be a boolean.")


def _manifest_payload(
    indexed: IndexedDocument,
    *,
    include_redacted_text: bool,
    source_retained: bool,
) -> dict[str, Any]:
    excluded = set() if include_redacted_text else {"text", "sections"}
    payload = indexed.document.model_dump(
        mode="json",
        exclude_none=True,
        exclude=excluded,
    )
    payload.pop("file_path", None)
    payload["chunk_count"] = indexed.chunk_count
    payload["source_retained"] = source_retained
    return payload


def _remove_new_source(document_store: Any, retained_copy: Optional[Path]) -> None:
    if retained_copy is None:
        return
    try:
        document_store.remove_source(retained_copy)
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    try:
        _strict_flags(args)
        owner_id = normalize_owner_id(args.owner_id)
        output_path = _lexical_absolute(args.output) if args.output else None
        files = _collect_files(args.paths, args.recursive, output_path)
    except Exception:
        print("Invalid batch-ingestion arguments or paths.", file=sys.stderr)
        return 2
    if not files:
        print("No supported non-redirected input files were found.", file=sys.stderr)
        return 1
    try:
        rag = get_rag_layer()
        client = _llm_client()
        document_store = get_document_store()
    except Exception:
        print("Batch-ingestion dependencies are unavailable.", file=sys.stderr)
        return 1

    manifest: list[dict[str, Any]] = []
    failures = 0
    for path in files:
        print(
            f"Ingesting {mask_metadata_text(path.name)[:500]} ...",
            end=" ",
            flush=True,
        )
        retained_copy: Optional[Path] = None
        prior: AuthoritativeDocumentSnapshot | None = None
        indexed: IndexedDocument | None = None
        registry_committed = False
        try:
            result = ingest_file(str(path), owner_id=owner_id)
            if not result.success or result.document is None:
                raise ValueError("Document ingestion failed.")
            document = result.document
            prior = capture_authoritative_document(
                owner_id=owner_id,
                doc_id=document.id,
                rag=rag,
            )
            if args.retain_sources:
                retained_copy = document_store.copy_source(
                    owner_id=owner_id,
                    source_path=path,
                )
            indexed = index_document(
                document,
                owner_id=owner_id,
                rag=rag,
                client=client,
            )
            payload = _manifest_payload(
                indexed,
                include_redacted_text=args.include_redacted_text,
                source_retained=retained_copy is not None,
            )
            previous_path = document_store.register(
                owner_id=owner_id,
                doc_id=document.id,
                filename=document.filename,
                mime_type=document.mime_type,
                source_path=retained_copy,
            )
            registry_committed = True
            cleanup_pending = False
            if previous_path:
                try:
                    cleanup_pending = document_store.remove_source(previous_path) is not True
                except Exception:
                    cleanup_pending = True
            if cleanup_pending:
                payload["replacement_cleanup_pending"] = True
            manifest.append(payload)
            print(
                f"OK ({indexed.chunk_count} chunks, "
                f"{'source retained' if retained_copy else 'text evidence only'})"
            )
        except Exception:
            if not registry_committed:
                _remove_new_source(document_store, retained_copy)
                if indexed is not None and prior is not None:
                    try:
                        restore_authoritative_document(prior, rag=rag)
                    except Exception:
                        pass
            failures += 1
            print("FAILED")
            if args.fail_fast:
                break

    if output_path is not None:
        try:
            _atomic_manifest(output_path, manifest)
            print(
                f"Manifest written to "
                f"{mask_metadata_text(output_path.name)[:500]}"
            )
        except Exception:
            print("The ingestion manifest could not be published.", file=sys.stderr)
            failures += 1
    print(f"Completed: {len(manifest)} succeeded, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
