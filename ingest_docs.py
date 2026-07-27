"""Batch document ingestion CLI using the same service as the API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from tools.document_service import ingest_and_index
from tools.rag import get_rag_layer
from tools.security import normalize_owner_id

_ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


def _collect_files(paths: List[str], recursive: bool, output_path: Optional[Path]) -> List[Path]:
    collected: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = list(path.rglob("*") if recursive else path.glob("*"))
        else:
            continue
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in _ALLOWED_SUFFIXES:
                continue
            if output_path and candidate.resolve() == output_path.resolve():
                continue
            collected.append(candidate)
    return sorted(set(collected), key=lambda item: str(item.resolve()))


def _llm_client() -> Optional[Any]:
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key and not base_url:
        return None
    return OpenAI(
        api_key=api_key or "local-no-key",
        base_url=base_url,
        timeout=60,
        max_retries=2,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse, redact, and index PDF, DOCX, Markdown, and text files."
    )
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest.")
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument("-o", "--output", default="ingestion_manifest.json")
    parser.add_argument("--owner-id", default=os.getenv("SINGLE_USER_OWNER_ID", "default_user"))
    parser.add_argument(
        "--include-redacted-text",
        action="store_true",
        help="Include redacted full text and sections in the output manifest.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        owner_id = normalize_owner_id(args.owner_id)
    except Exception as exc:
        print(f"Invalid owner ID: {exc}", file=sys.stderr)
        return 2

    output_path = Path(args.output) if args.output else None
    files = _collect_files(args.paths, args.recursive, output_path)
    if not files:
        print("No supported input files were found.", file=sys.stderr)
        return 1

    rag = get_rag_layer()
    client = _llm_client()
    manifest = []
    failures = 0
    for path in files:
        print(f"Ingesting {path} ...", end=" ", flush=True)
        try:
            indexed = ingest_and_index(
                str(path),
                owner_id=owner_id,
                rag=rag,
                client=client,
            )
            document = indexed.document
            payload = document.model_dump(
                mode="json",
                exclude_none=True,
                exclude=set() if args.include_redacted_text else {"text", "sections"},
            )
            payload["chunk_count"] = indexed.chunk_count
            manifest.append(payload)
            print(f"OK ({indexed.chunk_count} chunks)")
        except Exception as exc:
            failures += 1
            print(f"FAILED ({exc})")
            if args.fail_fast:
                break

    if output_path:
        output_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Manifest written to {output_path}")
    print(f"Completed: {len(manifest)} succeeded, {failures} failed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
