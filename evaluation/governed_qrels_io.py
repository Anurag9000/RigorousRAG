"""Strict persistence and reconstruction for governed qrels v2 receipts."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from evaluation.governed_qrels import GovernedQrels, GovernedQrelsReceipt, load_governed_qrels
from training.advanced_path_authority import safe_advanced_path

_MAX_BYTES = 16 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _read(path: str | Path, label: str) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label=label, must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES:
        raise ValueError(f"{label} exceeds byte safety bound")
    try:
        raw = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must contain an object")
    return raw


def _atomic(path: str | Path, payload: Mapping[str, Any], label: str) -> None:
    destination = safe_advanced_path(path, label=label, must_exist=False)
    if destination.exists() and destination.is_dir():
        raise ValueError(f"{label} destination must be a file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical(payload) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_governed_qrels_receipt(path: str | Path, qrels: GovernedQrels) -> None:
    if not isinstance(qrels, GovernedQrels):
        raise ValueError("qrels must be GovernedQrels")
    _atomic(path, {**qrels.receipt.unsigned(), "receipt_sha256": qrels.receipt.receipt_sha256}, "governed qrels receipt")


def read_governed_qrels_receipt(path: str | Path) -> GovernedQrelsReceipt:
    raw = _read(path, "governed qrels receipt")
    required = {"schema", "source_path", "source_sha256", "input_format", "minimum_relevance", "query_field", "document_field", "relevance_field", "pair_count", "query_count", "document_count", "relevant_pair_sha256", "receipt_sha256"}
    if set(raw) != required or raw.get("schema") != "rigorousrag-governed-qrels-receipt/v2":
        raise ValueError("unsupported governed qrels receipt schema")
    return GovernedQrelsReceipt(
        source_path=raw["source_path"], source_sha256=raw["source_sha256"], input_format=raw["input_format"], minimum_relevance=raw["minimum_relevance"],
        query_field=raw["query_field"], document_field=raw["document_field"], relevance_field=raw["relevance_field"], pair_count=raw["pair_count"],
        query_count=raw["query_count"], document_count=raw["document_count"], relevant_pair_sha256=raw["relevant_pair_sha256"], receipt_sha256=raw["receipt_sha256"],
    )


def load_governed_qrels_from_receipt(path: str | Path) -> GovernedQrels:
    receipt = read_governed_qrels_receipt(path)
    qrels = load_governed_qrels(
        receipt.source_path,
        expected_sha256=receipt.source_sha256,
        input_format=receipt.input_format,
        minimum_relevance=receipt.minimum_relevance,
        query_field=receipt.query_field,
        document_field=receipt.document_field,
        relevance_field=receipt.relevance_field,
    )
    if qrels.receipt != receipt or qrels.receipt.receipt_sha256 != receipt.receipt_sha256:
        raise ValueError("reloaded qrels semantics differ from persisted receipt")
    return qrels


__all__ = ["load_governed_qrels_from_receipt", "read_governed_qrels_receipt", "write_governed_qrels_receipt"]
