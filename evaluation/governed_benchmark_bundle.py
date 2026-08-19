"""Complete identity for governed retrieval benchmarks with separate corpora/auxiliary bytes.

The v1 query import remains backward compatible. This companion bundle proves that its query
manifest/import receipt, passed leakage qualification, separately imported full-text corpus,
and optional role-labelled auxiliary artifacts (for example standalone qrels) all refer to one
immutable evaluation input. Relevant-document coverage is checked against the corpus with a
temporary SQLite set so large corpora need not be retained in Python memory.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.benchmark_run_evidence import BenchmarkResultArtifactReceipt, materialize_benchmark_run_evidence
from evaluation.benchmark_suite import BenchmarkSuiteResult
from evaluation.advanced_rag_receipts import AdvancedEvaluationRun
from evaluation.governed_benchmark_corpus import GovernedBenchmarkCorpusReceipt
from evaluation.governed_benchmark_corpus_io import iter_verified_benchmark_corpus, verify_governed_benchmark_corpus_receipt
from evaluation.governed_benchmark_io import VerifiedGovernedBenchmark
from evaluation.governed_benchmark_qualification import GovernedBenchmarkLeakageReceipt
from training.advanced_path_authority import safe_advanced_path

_HEX = frozenset("0123456789abcdef")
_MAX_AUXILIARY = 100
_MAX_MISSING_SAMPLE = 100


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block: break
            digest.update(block)
    return digest.hexdigest()


def _id_digest(values: Sequence[str]) -> str:
    selected = sorted(set(values))
    return hashlib.sha256(("\n".join(selected) + ("\n" if selected else "")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuxiliaryBenchmarkArtifact:
    role: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _identifier(self.role, "auxiliary role", 200))
        source = safe_advanced_path(self.path, label=f"benchmark auxiliary {self.role}", must_exist=True, require_file=True)
        object.__setattr__(self, "path", str(source))
        object.__setattr__(self, "sha256", _sha(self.sha256, f"auxiliary {self.role} sha256"))
        if _stream_sha(source) != self.sha256:
            raise ValueError(f"benchmark auxiliary {self.role} bytes differ from configured SHA-256")


@dataclass(frozen=True)
class GovernedBenchmarkBundleReceipt:
    dataset_manifest_sha256: str
    query_import_receipt_sha256: str
    leakage_receipt_sha256: str
    corpus_receipt_sha256: str
    corpus_output_sha256: str
    relevant_id_sha256: str
    relevant_id_count: int
    auxiliary_artifacts: tuple[AuxiliaryBenchmarkArtifact, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("dataset_manifest_sha256", "query_import_receipt_sha256", "leakage_receipt_sha256", "corpus_receipt_sha256", "corpus_output_sha256", "relevant_id_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if isinstance(self.relevant_id_count, bool) or not isinstance(self.relevant_id_count, int) or self.relevant_id_count < 0:
            raise ValueError("relevant_id_count must be non-negative")
        auxiliary = tuple(self.auxiliary_artifacts)
        if len(auxiliary) > _MAX_AUXILIARY or any(not isinstance(item, AuxiliaryBenchmarkArtifact) for item in auxiliary):
            raise ValueError("auxiliary_artifacts must be a bounded AuxiliaryBenchmarkArtifact tuple")
        if len({item.role for item in auxiliary}) != len(auxiliary):
            raise ValueError("auxiliary benchmark roles must be unique")
        object.__setattr__(self, "auxiliary_artifacts", auxiliary)
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("governed benchmark bundle receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-governed-benchmark-bundle-receipt/v1",
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "query_import_receipt_sha256": self.query_import_receipt_sha256,
            "leakage_receipt_sha256": self.leakage_receipt_sha256,
            "corpus_receipt_sha256": self.corpus_receipt_sha256,
            "corpus_output_sha256": self.corpus_output_sha256,
            "relevant_id_sha256": self.relevant_id_sha256,
            "relevant_id_count": self.relevant_id_count,
            "auxiliary_artifacts": [asdict(item) for item in self.auxiliary_artifacts],
        }


def _verify_relevant_coverage(benchmark: VerifiedGovernedBenchmark, corpus: GovernedBenchmarkCorpusReceipt) -> tuple[str, int]:
    database_fd, database_name = tempfile.mkstemp(prefix="rigorousrag-benchmark-corpus-", suffix=".sqlite3")
    os.close(database_fd)
    relevant: set[str] = set()
    missing: list[str] = []
    try:
        connection = sqlite3.connect(database_name)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE corpus_ids(document_id TEXT PRIMARY KEY) WITHOUT ROWID")
            for document in iter_verified_benchmark_corpus(corpus.output_path, expected_sha256=corpus.output_sha256):
                try:
                    connection.execute("INSERT INTO corpus_ids(document_id) VALUES (?)", (document.document_id,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"corpus contains duplicate document id {document.document_id!r}") from exc
            connection.commit()
            for split in benchmark.manifest.splits:
                for example in benchmark.split(split.name):
                    for document_id in example.relevant_ids:
                        relevant.add(document_id)
                        if connection.execute("SELECT 1 FROM corpus_ids WHERE document_id=?", (document_id,)).fetchone() is None and len(missing) < _MAX_MISSING_SAMPLE:
                            missing.append(document_id)
            if missing:
                raise ValueError(f"benchmark relevant ids are absent from corpus; sample={missing}")
        finally:
            connection.close()
    finally:
        try: os.unlink(database_name)
        except FileNotFoundError: pass
    return _id_digest(tuple(relevant)), len(relevant)


def build_governed_benchmark_bundle(
    benchmark: VerifiedGovernedBenchmark,
    *,
    leakage_receipt: GovernedBenchmarkLeakageReceipt,
    corpus_receipt_path: str | Path,
    auxiliary_artifacts: Sequence[AuxiliaryBenchmarkArtifact] = (),
) -> GovernedBenchmarkBundleReceipt:
    if not isinstance(benchmark, VerifiedGovernedBenchmark):
        raise ValueError("benchmark must be VerifiedGovernedBenchmark")
    if not isinstance(leakage_receipt, GovernedBenchmarkLeakageReceipt) or not leakage_receipt.passed:
        raise ValueError("bundle requires a passed GovernedBenchmarkLeakageReceipt")
    if leakage_receipt.dataset_manifest_sha256 != benchmark.manifest.manifest_digest or leakage_receipt.import_receipt_sha256 != benchmark.receipt.receipt_sha256:
        raise ValueError("leakage receipt differs from governed query benchmark")
    corpus = verify_governed_benchmark_corpus_receipt(corpus_receipt_path)
    relevant_sha, relevant_count = _verify_relevant_coverage(benchmark, corpus)
    auxiliary = tuple(auxiliary_artifacts)
    unsigned = {
        "schema": "rigorousrag-governed-benchmark-bundle-receipt/v1",
        "dataset_manifest_sha256": benchmark.manifest.manifest_digest,
        "query_import_receipt_sha256": benchmark.receipt.receipt_sha256,
        "leakage_receipt_sha256": leakage_receipt.receipt_sha256,
        "corpus_receipt_sha256": corpus.receipt_sha256,
        "corpus_output_sha256": corpus.output_sha256,
        "relevant_id_sha256": relevant_sha,
        "relevant_id_count": relevant_count,
        "auxiliary_artifacts": [asdict(item) for item in auxiliary],
    }
    return GovernedBenchmarkBundleReceipt(
        dataset_manifest_sha256=unsigned["dataset_manifest_sha256"], query_import_receipt_sha256=unsigned["query_import_receipt_sha256"],
        leakage_receipt_sha256=unsigned["leakage_receipt_sha256"], corpus_receipt_sha256=unsigned["corpus_receipt_sha256"], corpus_output_sha256=unsigned["corpus_output_sha256"],
        relevant_id_sha256=relevant_sha, relevant_id_count=relevant_count, auxiliary_artifacts=auxiliary, receipt_sha256=_digest(unsigned),
    )


def bundled_evaluator_contract_sha256(base_evaluator_contract_sha256: str, bundle: GovernedBenchmarkBundleReceipt) -> str:
    base = _sha(base_evaluator_contract_sha256, "base_evaluator_contract_sha256")
    if not isinstance(bundle, GovernedBenchmarkBundleReceipt):
        raise ValueError("bundle must be GovernedBenchmarkBundleReceipt")
    return _digest({"schema": "rigorousrag-bundled-benchmark-evaluator-contract/v1", "base_evaluator_contract_sha256": base, "benchmark_bundle_receipt_sha256": bundle.receipt_sha256})


def materialize_bundled_benchmark_run_evidence(
    result: BenchmarkSuiteResult,
    *,
    benchmark: VerifiedGovernedBenchmark,
    bundle: GovernedBenchmarkBundleReceipt,
    base_evaluator_contract_sha256: str,
    seed: int,
    repeat_index: int,
    output_path: str | Path,
) -> tuple[AdvancedEvaluationRun, BenchmarkResultArtifactReceipt]:
    if bundle.dataset_manifest_sha256 != benchmark.manifest.manifest_digest or bundle.query_import_receipt_sha256 != benchmark.receipt.receipt_sha256:
        raise ValueError("benchmark bundle differs from governed query benchmark")
    evaluator_sha = bundled_evaluator_contract_sha256(base_evaluator_contract_sha256, bundle)
    return materialize_benchmark_run_evidence(result, benchmark_manifest=benchmark.manifest, evaluator_contract_sha256=evaluator_sha, seed=seed, repeat_index=repeat_index, output_path=output_path)


def write_governed_benchmark_bundle(path: str | Path, bundle: GovernedBenchmarkBundleReceipt) -> None:
    if not isinstance(bundle, GovernedBenchmarkBundleReceipt):
        raise ValueError("bundle must be GovernedBenchmarkBundleReceipt")
    destination = safe_advanced_path(path, label="governed benchmark bundle receipt", must_exist=False)
    if destination.exists() and destination.is_dir():
        raise ValueError("bundle destination must be a file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical({**bundle.unsigned(), "receipt_sha256": bundle.receipt_sha256}) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


__all__ = ["AuxiliaryBenchmarkArtifact", "GovernedBenchmarkBundleReceipt", "build_governed_benchmark_bundle", "bundled_evaluator_contract_sha256", "materialize_bundled_benchmark_run_evidence", "write_governed_benchmark_bundle"]
