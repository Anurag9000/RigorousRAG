"""Batch document ingestion CLI using the same parsing/indexing services as the API."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from tools.document_service import IndexedDocument, index_document
from tools.document_store import get_document_store
from tools.ingestion import ingest_file
from tools.privacy import mask_metadata_text
from tools.rag import get_rag_layer
from tools.security import normalize_owner_id

_ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}
_MAX_INPUT_FILES = 10_000
_MAX_PATH_CHARS = 4096
_MAX_VECTOR_ROWS = 100_000
_VECTOR_BATCH_SIZE = 128
_MAX_MANIFEST_BYTES = 50_000_000
_MAX_VECTOR_TEXT_CHARS = 50_000_000
_MAX_VECTOR_METADATA_ITEMS = 2000


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _lexical_absolute(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("Paths must be strings or path-like values.")
    rendered = os.fspath(value)
    if (
        not rendered
        or len(rendered) > _MAX_PATH_CHARS
        or _contains_ascii_control(rendered)
    ):
        raise ValueError("A path is invalid or too long.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _has_symlink_component(path: Path) -> bool:
    absolute = _lexical_absolute(path)
    for component in (absolute, *absolute.parents):
        try:
            if component.is_symlink():
                return True
        except OSError:
            return True
    return False


def _regular_supported_file(path: Path) -> bool:
    try:
        if _has_symlink_component(path):
            return False
        info = os.stat(path, follow_symlinks=False)
    except (OSError, ValueError):
        return False
    return stat.S_ISREG(info.st_mode) and path.suffix.lower() in _ALLOWED_SUFFIXES


def _directory_files(directory: Path, recursive: bool) -> Iterator[Path]:
    """Yield directory files without materializing an unbounded directory listing."""

    stack = [directory]
    inspected = 0
    inspection_limit = max(_MAX_INPUT_FILES * 20, _MAX_INPUT_FILES)
    while stack:
        current = stack.pop()
        try:
            if _has_symlink_component(current):
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
                            "Directory traversal exceeded the bounded entry-inspection limit."
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
    paths: List[str],
    recursive: bool,
    output_path: Optional[Path],
) -> List[Path]:
    if not isinstance(paths, list):
        raise ValueError("paths must be a list.")
    if len(paths) > _MAX_INPUT_FILES:
        raise ValueError(f"At most {_MAX_INPUT_FILES} input paths may be supplied.")
    if any(not isinstance(path, str) for path in paths):
        raise ValueError("Every input path must be a string.")
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
                info = os.stat(path, follow_symlinks=False)
            except OSError:
                continue
            if not stat.S_ISDIR(info.st_mode) or _has_symlink_component(path):
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
                    f"At most {_MAX_INPUT_FILES} supported input files may be ingested."
                )
    return [collected[key] for key in sorted(collected)]


def _provider_value(name: str) -> Optional[str]:
    if not isinstance(name, str) or not name or len(name) > 200:
        raise ValueError("Provider setting names must be valid strings.")
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    value = raw.strip()
    if len(value) > 4096 or _contains_ascii_control(value):
        raise ValueError(f"{name} is invalid.")
    return value or None


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


@dataclass(frozen=True)
class _VectorGeneration:
    ids: List[str]
    documents: List[str]
    metadatas: List[dict[str, Any]]


def _vector_filter(owner_id: str, doc_id: str) -> dict[str, Any]:
    return {
        "$and": [
            {"owner_id": {"$eq": owner_id}},
            {"doc_id": {"$eq": doc_id}},
        ]
    }


def _capture_generation(rag: Any, owner_id: str, doc_id: str) -> _VectorGeneration:
    result = rag.collection.get(
        where=_vector_filter(owner_id, doc_id),
        include=["documents", "metadatas"],
        limit=_MAX_VECTOR_ROWS + 1,
    )
    if not isinstance(result, dict):
        raise RuntimeError("The vector backend returned an invalid snapshot.")
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    if not all(isinstance(value, list) for value in (ids, documents, metadatas)):
        raise RuntimeError("The vector backend returned an invalid snapshot.")
    if len(ids) > _MAX_VECTOR_ROWS:
        raise RuntimeError("The prior vector generation exceeds the restoration limit.")
    if len(documents) != len(ids) or len(metadatas) != len(ids):
        raise RuntimeError("The prior vector generation is incomplete.")

    clean_ids: List[str] = []
    clean_documents: List[str] = []
    clean_metadatas: List[dict[str, Any]] = []
    for vector_id, text, metadata in zip(ids, documents, metadatas):
        if (
            not isinstance(vector_id, str)
            or not vector_id
            or len(vector_id) > 1000
            or _contains_ascii_control(vector_id)
        ):
            raise RuntimeError("The prior vector generation contains an invalid ID.")
        if (
            not isinstance(text, str)
            or len(text) > _MAX_VECTOR_TEXT_CHARS
            or not isinstance(metadata, dict)
            or len(metadata) > _MAX_VECTOR_METADATA_ITEMS
        ):
            raise RuntimeError("The prior vector generation contains invalid rows.")
        if metadata.get("owner_id") != owner_id or metadata.get("doc_id") != doc_id:
            raise RuntimeError("The prior vector generation violates owner scope.")
        clean_ids.append(vector_id)
        clean_documents.append(text)
        clean_metadatas.append(dict(metadata))
    return _VectorGeneration(clean_ids, clean_documents, clean_metadatas)


def _restore_generation(
    rag: Any,
    owner_id: str,
    doc_id: str,
    previous: _VectorGeneration,
) -> None:
    errors: List[str] = []
    try:
        rag.delete_document(owner_id=owner_id, doc_id=doc_id)
    except Exception as exc:
        errors.append(f"delete:{type(exc).__name__}")
    if previous.ids:
        for start in range(0, len(previous.ids), _VECTOR_BATCH_SIZE):
            stop = start + _VECTOR_BATCH_SIZE
            try:
                rag.collection.upsert(
                    ids=previous.ids[start:stop],
                    documents=previous.documents[start:stop],
                    metadatas=previous.metadatas[start:stop],
                )
            except Exception as exc:
                errors.append(f"restore:{type(exc).__name__}")
                break
    if errors:
        raise RuntimeError("Vector rollback was incomplete: " + ", ".join(errors))


def _atomic_manifest(path: Path, manifest: list[dict[str, Any]]) -> None:
    destination = _lexical_absolute(path)
    if _has_symlink_component(destination.parent):
        raise ValueError("The manifest parent path may not contain symbolic links.")
    if not destination.parent.exists() or not destination.parent.is_dir():
        raise ValueError("The manifest parent directory does not exist.")
    payload = json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise ValueError("The ingestion manifest exceeds the configured byte limit.")

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
    parser.add_argument(
        "--retain-sources",
        action="store_true",
        help=(
            "Copy source files into the private owner-scoped store so figure tools can "
            "use them. The manifest never contains the retained path."
        ),
    )
    parser.add_argument(
        "--include-redacted-text",
        action="store_true",
        help="Include redacted full text and sections in the output manifest.",
    )
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
        print("No supported non-symlink input files were found.", file=sys.stderr)
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
        previous: Optional[_VectorGeneration] = None
        indexed: Optional[IndexedDocument] = None
        document = None
        registry_committed = False
        try:
            result = ingest_file(str(path), owner_id=owner_id)
            if not result.success or result.document is None:
                raise ValueError("Document ingestion failed.")
            document = result.document
            previous = _capture_generation(rag, owner_id, document.id)
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
            payload = document.model_dump(
                mode="json",
                exclude_none=True,
                exclude=(set() if args.include_redacted_text else {"text", "sections"}),
            )
            payload["chunk_count"] = indexed.chunk_count
            payload["source_retained"] = retained_copy is not None

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
                if retained_copy is not None:
                    try:
                        document_store.remove_source(retained_copy)
                    except Exception:
                        pass
                if indexed is not None and document is not None and previous is not None:
                    try:
                        _restore_generation(rag, owner_id, document.id, previous)
                    except Exception:
                        pass
            failures += 1
            print("FAILED")
            if args.fail_fast:
                break

    if output_path is not None:
        try:
            _atomic_manifest(output_path, manifest)
            print(f"Manifest written to {mask_metadata_text(output_path.name)[:500]}")
        except Exception:
            print("The ingestion manifest could not be published.", file=sys.stderr)
            failures += 1
    print(f"Completed: {len(manifest)} succeeded, {failures} failed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
