"""Persistence and restart reconstruction for authoritative retrieval benchmarks."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from evaluation.authoritative_governed_benchmark_corpus import (
    verify_authoritative_benchmark_corpus_receipt,
)
from evaluation.authoritative_governed_benchmark_io import (
    verify_authoritative_governed_benchmark_import,
)
from evaluation.authoritative_governed_retrieval_benchmark import (
    AuthoritativeGovernedRetrievalBenchmark,
    build_authoritative_governed_retrieval_benchmark,
)
from evaluation.authoritative_governed_qrels import close_authoritative_governed_qrels
from evaluation.governed_benchmark_qualification import (
    GovernedBenchmarkLeakageReceipt,
    LeakageFindingEvidence,
)
from evaluation.governed_qrels_io import load_governed_qrels_from_receipt
from training.advanced_path_authority import safe_advanced_path

_MAX_BYTES = 32 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block: break
            digest.update(block)
    return digest.hexdigest()


def _read(path: str | Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    source = safe_advanced_path(path, label=label, must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES:
        raise ValueError(f"{label} exceeds byte safety bound")
    try:
        raw = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must contain an object")
    return source, raw


def _atomic(path: Path, payload: bytes) -> None:
    if path.exists() and path.is_dir():
        raise ValueError("retrieval benchmark receipt destination must be a file")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _read_leakage(path: str | Path) -> GovernedBenchmarkLeakageReceipt:
    _, raw = _read(path, "authoritative benchmark leakage receipt")
    required = {"schema", "dataset_manifest_sha256", "import_receipt_sha256", "split_key_sha256", "blocking_key_kinds", "findings", "passed", "receipt_sha256"}
    if set(raw) != required or raw.get("schema") != "rigorousrag-governed-benchmark-leakage-receipt/v1":
        raise ValueError("unsupported benchmark leakage receipt schema")
    split_keys = raw["split_key_sha256"]
    if not isinstance(split_keys, Mapping):
        raise ValueError("leakage split_key_sha256 must be an object")
    findings_raw = raw["findings"]
    if not isinstance(findings_raw, list):
        raise ValueError("leakage findings must be an array")
    findings = []
    expected_finding = {"left_split", "right_split", "key_kind", "severity", "overlap_count", "overlap_sha256", "overlap_sample"}
    for item in findings_raw:
        if not isinstance(item, Mapping) or set(item) != expected_finding:
            raise ValueError("leakage finding fields are invalid")
        sample = item["overlap_sample"]
        if not isinstance(sample, list):
            raise ValueError("leakage overlap_sample must be an array")
        findings.append(LeakageFindingEvidence(
            left_split=item["left_split"], right_split=item["right_split"], key_kind=item["key_kind"], severity=item["severity"],
            overlap_count=item["overlap_count"], overlap_sha256=item["overlap_sha256"], overlap_sample=tuple(sample),
        ))
    blocking = raw["blocking_key_kinds"]
    if not isinstance(blocking, list):
        raise ValueError("blocking_key_kinds must be an array")
    return GovernedBenchmarkLeakageReceipt(
        dataset_manifest_sha256=raw["dataset_manifest_sha256"], import_receipt_sha256=raw["import_receipt_sha256"],
        split_key_sha256={str(split): {str(kind): str(value) for kind, value in groups.items()} for split, groups in split_keys.items()},
        blocking_key_kinds=tuple(blocking), findings=tuple(findings), passed=raw["passed"], receipt_sha256=raw["receipt_sha256"],
    )


@dataclass(frozen=True)
class AuthoritativeRetrievalBenchmarkReceipt:
    query_import_receipt_path: str
    query_import_receipt_file_sha256: str
    leakage_receipt_path: str
    leakage_receipt_file_sha256: str
    qrels_receipt_path: str
    qrels_receipt_file_sha256: str
    corpus_receipt_path: str
    corpus_receipt_file_sha256: str
    query_manifest_sha256: str
    qrels_coverage_sha256: str
    retrieval_contract_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("query_import_receipt_path", "leakage_receipt_path", "qrels_receipt_path", "corpus_receipt_path"):
            path = safe_advanced_path(getattr(self, name), label=name, must_exist=True, require_file=True)
            object.__setattr__(self, name, str(path))
        for name in ("query_import_receipt_file_sha256", "leakage_receipt_file_sha256", "qrels_receipt_file_sha256", "corpus_receipt_file_sha256", "query_manifest_sha256", "qrels_coverage_sha256", "retrieval_contract_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("authoritative retrieval benchmark receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-authoritative-retrieval-benchmark-receipt/v2",
            "query_import_receipt_path": self.query_import_receipt_path,
            "query_import_receipt_file_sha256": self.query_import_receipt_file_sha256,
            "leakage_receipt_path": self.leakage_receipt_path,
            "leakage_receipt_file_sha256": self.leakage_receipt_file_sha256,
            "qrels_receipt_path": self.qrels_receipt_path,
            "qrels_receipt_file_sha256": self.qrels_receipt_file_sha256,
            "corpus_receipt_path": self.corpus_receipt_path,
            "corpus_receipt_file_sha256": self.corpus_receipt_file_sha256,
            "query_manifest_sha256": self.query_manifest_sha256,
            "qrels_coverage_sha256": self.qrels_coverage_sha256,
            "retrieval_contract_sha256": self.retrieval_contract_sha256,
        }


def build_authoritative_retrieval_benchmark_receipt(
    benchmark: AuthoritativeGovernedRetrievalBenchmark,
    *,
    query_import_receipt_path: str | Path,
    leakage_receipt_path: str | Path,
    qrels_receipt_path: str | Path,
    corpus_receipt_path: str | Path,
) -> AuthoritativeRetrievalBenchmarkReceipt:
    if not isinstance(benchmark, AuthoritativeGovernedRetrievalBenchmark):
        raise ValueError("benchmark must be AuthoritativeGovernedRetrievalBenchmark")
    query_path = safe_advanced_path(query_import_receipt_path, label="query import receipt", must_exist=True, require_file=True)
    leakage_path = safe_advanced_path(leakage_receipt_path, label="leakage receipt", must_exist=True, require_file=True)
    qrels_path = safe_advanced_path(qrels_receipt_path, label="qrels receipt", must_exist=True, require_file=True)
    corpus_path = safe_advanced_path(corpus_receipt_path, label="corpus receipt", must_exist=True, require_file=True)
    unsigned = {
        "schema": "rigorousrag-authoritative-retrieval-benchmark-receipt/v2",
        "query_import_receipt_path": str(query_path),
        "query_import_receipt_file_sha256": _file_sha(query_path),
        "leakage_receipt_path": str(leakage_path),
        "leakage_receipt_file_sha256": _file_sha(leakage_path),
        "qrels_receipt_path": str(qrels_path),
        "qrels_receipt_file_sha256": _file_sha(qrels_path),
        "corpus_receipt_path": str(corpus_path),
        "corpus_receipt_file_sha256": _file_sha(corpus_path),
        "query_manifest_sha256": benchmark.queries.manifest.manifest_digest,
        "qrels_coverage_sha256": benchmark.qrels_coverage_sha256,
        "retrieval_contract_sha256": benchmark.contract_sha256,
    }
    return AuthoritativeRetrievalBenchmarkReceipt(**{key: value for key, value in unsigned.items() if key != "schema"}, receipt_sha256=_digest(unsigned))


def write_authoritative_retrieval_benchmark_receipt(path: str | Path, receipt: AuthoritativeRetrievalBenchmarkReceipt) -> None:
    if not isinstance(receipt, AuthoritativeRetrievalBenchmarkReceipt):
        raise ValueError("receipt must be AuthoritativeRetrievalBenchmarkReceipt")
    destination = safe_advanced_path(path, label="authoritative retrieval benchmark receipt", must_exist=False)
    _atomic(destination, _canonical({**receipt.unsigned(), "receipt_sha256": receipt.receipt_sha256}) + b"\n")


def load_authoritative_retrieval_benchmark_receipt(path: str | Path) -> AuthoritativeRetrievalBenchmarkReceipt:
    _, raw = _read(path, "authoritative retrieval benchmark receipt")
    required = {"schema", "query_import_receipt_path", "query_import_receipt_file_sha256", "leakage_receipt_path", "leakage_receipt_file_sha256", "qrels_receipt_path", "qrels_receipt_file_sha256", "corpus_receipt_path", "corpus_receipt_file_sha256", "query_manifest_sha256", "qrels_coverage_sha256", "retrieval_contract_sha256", "receipt_sha256"}
    if set(raw) != required or raw.get("schema") != "rigorousrag-authoritative-retrieval-benchmark-receipt/v2":
        raise ValueError("unsupported authoritative retrieval benchmark receipt schema")
    return AuthoritativeRetrievalBenchmarkReceipt(**{key: value for key, value in raw.items() if key != "schema"})


def reconstruct_authoritative_retrieval_benchmark(path: str | Path) -> tuple[AuthoritativeGovernedRetrievalBenchmark, AuthoritativeRetrievalBenchmarkReceipt]:
    receipt = load_authoritative_retrieval_benchmark_receipt(path)
    file_checks = (
        (receipt.query_import_receipt_path, receipt.query_import_receipt_file_sha256, "query import receipt"),
        (receipt.leakage_receipt_path, receipt.leakage_receipt_file_sha256, "leakage receipt"),
        (receipt.qrels_receipt_path, receipt.qrels_receipt_file_sha256, "qrels receipt"),
        (receipt.corpus_receipt_path, receipt.corpus_receipt_file_sha256, "corpus receipt"),
    )
    for raw_path, expected, label in file_checks:
        source = safe_advanced_path(raw_path, label=label, must_exist=True, require_file=True)
        if _file_sha(source) != expected:
            raise ValueError(f"{label} bytes changed after retrieval receipt publication")
    queries = verify_authoritative_governed_benchmark_import(receipt.query_import_receipt_path, require_promotable=True)
    leakage = _read_leakage(receipt.leakage_receipt_path)
    if not leakage.passed or leakage.dataset_manifest_sha256 != queries.manifest.manifest_digest or leakage.import_receipt_sha256 != queries.receipt.receipt_sha256:
        raise ValueError("persisted leakage receipt is not passed/bound to authoritative queries")
    qrels = load_governed_qrels_from_receipt(receipt.qrels_receipt_path)
    try:
        corpus = verify_authoritative_benchmark_corpus_receipt(receipt.corpus_receipt_path)
        benchmark = build_authoritative_governed_retrieval_benchmark(
            queries, leakage=leakage, qrels=qrels, corpus_receipt_path=receipt.corpus_receipt_path,
        )
        if corpus.receipt_sha256 != benchmark.corpus.receipt_sha256:
            raise RuntimeError("corpus reconstruction identity differs")
        if benchmark.queries.manifest.manifest_digest != receipt.query_manifest_sha256 or benchmark.qrels_coverage_sha256 != receipt.qrels_coverage_sha256 or benchmark.contract_sha256 != receipt.retrieval_contract_sha256:
            raise ValueError("reconstructed retrieval benchmark differs from persisted receipt")
        return benchmark, receipt
    except Exception:
        close_authoritative_governed_qrels(qrels)
        raise


def close_reconstructed_authoritative_retrieval_benchmark(benchmark: AuthoritativeGovernedRetrievalBenchmark) -> None:
    if not isinstance(benchmark, AuthoritativeGovernedRetrievalBenchmark):
        raise ValueError("benchmark must be AuthoritativeGovernedRetrievalBenchmark")
    close_authoritative_governed_qrels(benchmark.qrels)


__all__ = ["AuthoritativeRetrievalBenchmarkReceipt", "build_authoritative_retrieval_benchmark_receipt", "close_reconstructed_authoritative_retrieval_benchmark", "load_authoritative_retrieval_benchmark_receipt", "reconstruct_authoritative_retrieval_benchmark", "write_authoritative_retrieval_benchmark_receipt"]
