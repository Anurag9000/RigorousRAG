"""Governed standalone qrels for retrieval benchmarks.

Many IR datasets keep queries, corpus and relevance judgments in separate local files. Binding
qrels bytes as an opaque auxiliary artifact is not enough: evaluation examples must consume the
judgments. This module parses exact local qrels bytes, builds a content-addressed relevance map,
and overlays it onto governed ``BenchmarkExample`` values. Existing non-empty relevant ids must
match the qrels set by default; conflicting gold labels fail closed.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from tools.benchmark_adapters import BenchmarkExample
from training.advanced_path_authority import safe_advanced_path

_MAX_BYTES = 2 * 1024 * 1024 * 1024
_MAX_RECORDS = 200_000_000
_MAX_LINE_BYTES = 16 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected): raise ValueError(f"{label} must be SHA-256")
    return selected


def _identifier(value: Any, label: str, maximum: int = 10_000) -> str:
    selected = str(value).strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected): raise ValueError(f"{label} is invalid")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool): raise ValueError(f"{label} must be finite")
    selected = float(value)
    if not math.isfinite(selected): raise ValueError(f"{label} must be finite")
    return selected


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block: break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class QrelsEntry:
    query_id: str
    document_id: str
    relevance: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _identifier(self.query_id, "qrels query_id"))
        object.__setattr__(self, "document_id", _identifier(self.document_id, "qrels document_id"))
        object.__setattr__(self, "relevance", _finite(self.relevance, "qrels relevance"))


@dataclass(frozen=True)
class GovernedQrelsReceipt:
    source_path: str
    source_sha256: str
    input_format: str
    minimum_relevance: float
    pair_count: int
    query_count: int
    document_count: int
    relevant_pair_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_sha256", _sha(self.source_sha256, "source_sha256")); object.__setattr__(self, "relevant_pair_sha256", _sha(self.relevant_pair_sha256, "relevant_pair_sha256")); object.__setattr__(self, "receipt_sha256", _sha(self.receipt_sha256, "receipt_sha256"))
        if self.input_format not in {"json", "jsonl", "trec"}: raise ValueError("input_format must be json, jsonl or trec")
        object.__setattr__(self, "minimum_relevance", _finite(self.minimum_relevance, "minimum_relevance"))
        for name in ("pair_count", "query_count", "document_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0: raise ValueError(f"{name} must be non-negative")
        if _digest(self.unsigned()) != self.receipt_sha256: raise ValueError("governed qrels receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {"schema": "rigorousrag-governed-qrels-receipt/v1", "source_path": self.source_path, "source_sha256": self.source_sha256, "input_format": self.input_format, "minimum_relevance": self.minimum_relevance, "pair_count": self.pair_count, "query_count": self.query_count, "document_count": self.document_count, "relevant_pair_sha256": self.relevant_pair_sha256}


@dataclass(frozen=True)
class GovernedQrels:
    receipt: GovernedQrelsReceipt
    relevant_by_query: Mapping[str, tuple[str, ...]]

    @property
    def contract_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-governed-qrels-contract/v1", "receipt_sha256": self.receipt.receipt_sha256})

    def relevant_ids(self, query_id: str) -> tuple[str, ...]:
        return tuple(self.relevant_by_query.get(_identifier(query_id, "query_id"), ()))


def _strict_json(raw: bytes, label: str) -> Any:
    try: return json.loads(raw.decode("utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc: raise ValueError(f"{label} is not strict UTF-8 JSON") from exc


def _entry_from_mapping(value: Any, label: str, *, query_field: str, document_field: str, relevance_field: str) -> QrelsEntry:
    if not isinstance(value, Mapping): raise ValueError(f"{label} must be an object")
    for field in (query_field, document_field, relevance_field):
        if field not in value: raise ValueError(f"{label} lacks field {field!r}")
    return QrelsEntry(value[query_field], value[document_field], value[relevance_field])


def _entries(path: Path, *, input_format: str, query_field: str, document_field: str, relevance_field: str) -> Iterator[QrelsEntry]:
    if input_format == "trec":
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip(): continue
                if len(line.encode("utf-8")) > _MAX_LINE_BYTES: raise ValueError(f"qrels line {line_number} exceeds safety bound")
                fields = line.strip().split()
                if len(fields) == 4: query_id, _, document_id, relevance = fields
                elif len(fields) == 3: query_id, document_id, relevance = fields
                else: raise ValueError(f"qrels TREC line {line_number} must have 3 or 4 whitespace-separated fields")
                yield QrelsEntry(query_id, document_id, relevance)
        return
    if input_format == "jsonl":
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip(): continue
                if len(raw) > _MAX_LINE_BYTES: raise ValueError(f"qrels line {line_number} exceeds safety bound")
                yield _entry_from_mapping(_strict_json(raw, f"qrels line {line_number}"), f"qrels line {line_number}", query_field=query_field, document_field=document_field, relevance_field=relevance_field)
        return
    if path.stat().st_size <= 0 or path.stat().st_size > _MAX_BYTES: raise ValueError("qrels JSON exceeds whole-file safety bound")
    value = _strict_json(path.read_bytes(), "qrels JSON")
    if not isinstance(value, list): raise ValueError("qrels JSON must contain an array")
    for index, item in enumerate(value): yield _entry_from_mapping(item, f"qrels[{index}]", query_field=query_field, document_field=document_field, relevance_field=relevance_field)


def load_governed_qrels(
    path: str | Path, *, expected_sha256: str, input_format: str = "trec", minimum_relevance: float = 1.0,
    query_field: str = "query_id", document_field: str = "document_id", relevance_field: str = "relevance",
) -> GovernedQrels:
    source = safe_advanced_path(path, label="qrels source", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES: raise ValueError("qrels source exceeds byte safety bound")
    source_sha = _sha(expected_sha256, "expected_sha256")
    if _stream_sha(source) != source_sha: raise ValueError("qrels source digest differs from expected immutable bytes")
    selected_format = _identifier(input_format, "input_format", 20).lower()
    if selected_format not in {"json", "jsonl", "trec"}: raise ValueError("input_format must be json, jsonl or trec")
    threshold = _finite(minimum_relevance, "minimum_relevance"); relevant: dict[str, set[str]] = {}; pairs: set[tuple[str, str]] = set(); documents: set[str] = set(); count = 0
    for entry in _entries(source, input_format=selected_format, query_field=query_field, document_field=document_field, relevance_field=relevance_field):
        count += 1
        if count > _MAX_RECORDS: raise ValueError("qrels source exceeds record safety bound")
        if entry.relevance < threshold: continue
        pair = (entry.query_id, entry.document_id)
        if pair in pairs: continue
        pairs.add(pair); relevant.setdefault(entry.query_id, set()).add(entry.document_id); documents.add(entry.document_id)
    ordered_pairs = sorted(f"{query}\t{document}" for query, document in pairs); pair_sha = hashlib.sha256(("\n".join(ordered_pairs) + ("\n" if ordered_pairs else "")).encode("utf-8")).hexdigest()
    unsigned = {"schema": "rigorousrag-governed-qrels-receipt/v1", "source_path": str(source), "source_sha256": source_sha, "input_format": selected_format, "minimum_relevance": threshold, "pair_count": len(pairs), "query_count": len(relevant), "document_count": len(documents), "relevant_pair_sha256": pair_sha}
    receipt = GovernedQrelsReceipt(str(source), source_sha, selected_format, threshold, len(pairs), len(relevant), len(documents), pair_sha, _digest(unsigned))
    return GovernedQrels(receipt, {query: tuple(sorted(ids)) for query, ids in relevant.items()})


def overlay_qrels(
    examples: Iterable[BenchmarkExample], qrels: GovernedQrels, *, require_query_labels: bool = True, require_existing_equal: bool = True,
) -> Iterator[BenchmarkExample]:
    if not isinstance(qrels, GovernedQrels): raise ValueError("qrels must be GovernedQrels")
    for example in examples:
        if not isinstance(example, BenchmarkExample): raise ValueError("examples must contain BenchmarkExample values")
        gold = qrels.relevant_ids(example.example_id); existing = tuple(example.relevant_ids)
        if require_query_labels and not gold: raise ValueError(f"qrels contain no relevant documents for query {example.example_id!r}")
        if existing and gold and require_existing_equal and set(existing) != set(gold): raise ValueError(f"embedded relevant ids conflict with governed qrels for query {example.example_id!r}")
        selected = gold or existing
        metadata = dict(example.metadata); metadata["qrels_receipt_sha256"] = qrels.receipt.receipt_sha256
        yield BenchmarkExample(example.example_id, example.query, example.answers, tuple(selected), example.contexts, metadata)


__all__ = ["GovernedQrels", "GovernedQrelsReceipt", "QrelsEntry", "load_governed_qrels", "overlay_qrels"]
