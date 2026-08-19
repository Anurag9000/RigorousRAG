"""Strict read-side verification for governed retrieval benchmark corpora."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from evaluation.governed_benchmark_corpus import BenchmarkCorpusDocument, GovernedBenchmarkCorpusReceipt
from training.advanced_path_authority import safe_advanced_path

_MAX_LINE_BYTES = 128 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block: break
            digest.update(block)
    return digest.hexdigest()


def _strict(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8", errors="strict"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc


def _id_digest(values: list[str]) -> str:
    selected = sorted(set(values))
    return hashlib.sha256(("\n".join(selected) + ("\n" if selected else "")).encode("utf-8")).hexdigest()


def iter_verified_benchmark_corpus(path: str | Path, *, expected_sha256: str) -> Iterator[BenchmarkCorpusDocument]:
    source = safe_advanced_path(path, label="canonical benchmark corpus", must_exist=True, require_file=True)
    if _stream_sha(source) != _sha(expected_sha256, "expected_sha256"):
        raise ValueError("canonical benchmark corpus digest differs from receipt")
    seen: set[str] = set()
    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip(): continue
            if len(raw) > _MAX_LINE_BYTES:
                raise ValueError(f"corpus line {line_number} exceeds safety bound")
            value = _strict(raw, f"corpus line {line_number}")
            required = {"schema", "document_id", "title", "text", "source_group_id", "metadata"}
            if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "rigorousrag-benchmark-corpus-document/v1":
                raise ValueError(f"corpus line {line_number} has unsupported schema")
            document = BenchmarkCorpusDocument(value["document_id"], value["text"], value["title"], value["source_group_id"], value["metadata"])
            if document.document_id in seen:
                raise ValueError(f"canonical corpus contains duplicate document id {document.document_id!r}")
            seen.add(document.document_id)
            yield document


def verify_governed_benchmark_corpus_receipt(path: str | Path) -> GovernedBenchmarkCorpusReceipt:
    receipt_path = safe_advanced_path(path, label="benchmark corpus receipt", must_exist=True, require_file=True)
    raw = _strict(receipt_path.read_bytes(), "benchmark corpus receipt")
    required = {"schema", "source_sha256", "transformation_sha256", "output_path", "output_sha256", "record_count", "document_id_sha256", "source_group_sha256", "receipt_sha256"}
    if not isinstance(raw, Mapping) or set(raw) != required or raw.get("schema") != "rigorousrag-governed-benchmark-corpus-receipt/v1":
        raise ValueError("unsupported benchmark corpus receipt schema")
    receipt = GovernedBenchmarkCorpusReceipt(
        source_sha256=raw["source_sha256"], transformation_sha256=raw["transformation_sha256"], output_path=raw["output_path"], output_sha256=raw["output_sha256"],
        record_count=raw["record_count"], document_id_sha256=raw["document_id_sha256"], source_group_sha256=raw["source_group_sha256"], receipt_sha256=raw["receipt_sha256"],
    )
    ids: list[str] = []; groups: list[str] = []; count = 0
    for document in iter_verified_benchmark_corpus(receipt.output_path, expected_sha256=receipt.output_sha256):
        count += 1; ids.append(document.document_id)
        if document.source_group_id is not None: groups.append(document.source_group_id)
    if count != receipt.record_count or _id_digest(ids) != receipt.document_id_sha256:
        raise ValueError("canonical corpus count/document-id digest differs from receipt")
    actual_groups = _id_digest(groups) if groups else None
    if actual_groups != receipt.source_group_sha256:
        raise ValueError("canonical corpus source-group digest differs from receipt")
    return receipt


__all__ = ["iter_verified_benchmark_corpus", "verify_governed_benchmark_corpus_receipt"]
