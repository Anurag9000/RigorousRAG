"""Complete governed retrieval benchmark composition: queries + qrels + corpus.

This is the evaluation-facing object for standard IR layouts. It overlays semantic qrels into
query examples, proves every relevant document exists in the governed corpus, and binds query
manifest/import, leakage, qrels parsing semantics and corpus receipt into one evaluator
contract. No retrieval/model execution occurs here.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from evaluation.advanced_rag_receipts import AdvancedEvaluationRun
from evaluation.benchmark_run_evidence import BenchmarkResultArtifactReceipt, materialize_benchmark_run_evidence
from evaluation.benchmark_suite import BenchmarkSuiteResult
from evaluation.governed_benchmark_bundle import AuxiliaryBenchmarkArtifact, GovernedBenchmarkBundleReceipt, build_governed_benchmark_bundle
from evaluation.governed_benchmark_corpus import GovernedBenchmarkCorpusReceipt
from evaluation.governed_benchmark_corpus_io import iter_verified_benchmark_corpus, verify_governed_benchmark_corpus_receipt
from evaluation.governed_benchmark_io import VerifiedGovernedBenchmark
from evaluation.governed_benchmark_qualification import GovernedBenchmarkLeakageReceipt
from evaluation.governed_qrels import GovernedQrels, overlay_qrels
from tools.benchmark_adapters import BenchmarkExample

_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected): raise ValueError(f"{label} must be SHA-256")
    return selected


@dataclass(frozen=True)
class GovernedRetrievalBenchmark:
    queries: VerifiedGovernedBenchmark
    leakage: GovernedBenchmarkLeakageReceipt
    qrels: GovernedQrels
    corpus: GovernedBenchmarkCorpusReceipt
    bundle: GovernedBenchmarkBundleReceipt
    qrels_coverage_sha256: str
    contract_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "qrels_coverage_sha256", _sha(self.qrels_coverage_sha256, "qrels_coverage_sha256")); object.__setattr__(self, "contract_sha256", _sha(self.contract_sha256, "contract_sha256"))
        expected = _digest({"schema": "rigorousrag-governed-retrieval-benchmark/v1", "query_manifest_sha256": self.queries.manifest.manifest_digest, "query_import_receipt_sha256": self.queries.receipt.receipt_sha256, "leakage_receipt_sha256": self.leakage.receipt_sha256, "qrels_receipt_sha256": self.qrels.receipt.receipt_sha256, "corpus_receipt_sha256": self.corpus.receipt_sha256, "bundle_receipt_sha256": self.bundle.receipt_sha256, "qrels_coverage_sha256": self.qrels_coverage_sha256})
        if expected != self.contract_sha256: raise ValueError("governed retrieval benchmark contract digest mismatch")

    def split(self, name: str) -> Iterator[BenchmarkExample]:
        return overlay_qrels(self.queries.split(name), self.qrels, require_query_labels=True, require_existing_equal=True)


def _qrels_corpus_coverage(qrels: GovernedQrels, corpus: GovernedBenchmarkCorpusReceipt) -> str:
    fd, database = tempfile.mkstemp(prefix="rigorousrag-qrels-corpus-", suffix=".sqlite3"); os.close(fd)
    missing: list[str] = []
    try:
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA journal_mode=OFF"); connection.execute("PRAGMA synchronous=OFF"); connection.execute("CREATE TABLE ids(document_id TEXT PRIMARY KEY) WITHOUT ROWID")
            for document in iter_verified_benchmark_corpus(corpus.output_path, expected_sha256=corpus.output_sha256):
                connection.execute("INSERT INTO ids(document_id) VALUES (?)", (document.document_id,))
            connection.commit()
            pairs = []
            for query_id in sorted(qrels.relevant_by_query):
                for document_id in qrels.relevant_by_query[query_id]:
                    if connection.execute("SELECT 1 FROM ids WHERE document_id=?", (document_id,)).fetchone() is None:
                        if len(missing) < 100: missing.append(document_id)
                    pairs.append(f"{query_id}\t{document_id}")
            if missing: raise ValueError(f"qrels reference documents absent from governed corpus; sample={missing}")
        finally: connection.close()
    finally:
        try: os.unlink(database)
        except FileNotFoundError: pass
    return hashlib.sha256(("\n".join(pairs) + ("\n" if pairs else "")).encode("utf-8")).hexdigest()


def build_governed_retrieval_benchmark(
    queries: VerifiedGovernedBenchmark, *, leakage: GovernedBenchmarkLeakageReceipt, qrels: GovernedQrels, corpus_receipt_path: str | Path,
) -> GovernedRetrievalBenchmark:
    if not isinstance(queries, VerifiedGovernedBenchmark) or not isinstance(leakage, GovernedBenchmarkLeakageReceipt) or not isinstance(qrels, GovernedQrels): raise ValueError("queries/leakage/qrels have incorrect types")
    if not leakage.passed or leakage.dataset_manifest_sha256 != queries.manifest.manifest_digest or leakage.import_receipt_sha256 != queries.receipt.receipt_sha256: raise ValueError("retrieval benchmark requires passed leakage evidence bound to query import")
    corpus = verify_governed_benchmark_corpus_receipt(corpus_receipt_path)
    qrels_aux = AuxiliaryBenchmarkArtifact("qrels", qrels.receipt.source_path, qrels.receipt.source_sha256)
    bundle = build_governed_benchmark_bundle(queries, leakage_receipt=leakage, corpus_receipt_path=corpus_receipt_path, auxiliary_artifacts=(qrels_aux,))
    coverage = _qrels_corpus_coverage(qrels, corpus)
    contract = _digest({"schema": "rigorousrag-governed-retrieval-benchmark/v1", "query_manifest_sha256": queries.manifest.manifest_digest, "query_import_receipt_sha256": queries.receipt.receipt_sha256, "leakage_receipt_sha256": leakage.receipt_sha256, "qrels_receipt_sha256": qrels.receipt.receipt_sha256, "corpus_receipt_sha256": corpus.receipt_sha256, "bundle_receipt_sha256": bundle.receipt_sha256, "qrels_coverage_sha256": coverage})
    return GovernedRetrievalBenchmark(queries, leakage, qrels, corpus, bundle, coverage, contract)


def retrieval_evaluator_contract_sha256(base_evaluator_contract_sha256: str, benchmark: GovernedRetrievalBenchmark) -> str:
    return _digest({"schema": "rigorousrag-governed-retrieval-evaluator-contract/v1", "base_evaluator_contract_sha256": _sha(base_evaluator_contract_sha256, "base_evaluator_contract_sha256"), "governed_retrieval_benchmark_sha256": benchmark.contract_sha256})


def materialize_governed_retrieval_run_evidence(
    result: BenchmarkSuiteResult, *, benchmark: GovernedRetrievalBenchmark, base_evaluator_contract_sha256: str, seed: int, repeat_index: int, output_path: str | Path,
) -> tuple[AdvancedEvaluationRun, BenchmarkResultArtifactReceipt]:
    evaluator = retrieval_evaluator_contract_sha256(base_evaluator_contract_sha256, benchmark)
    return materialize_benchmark_run_evidence(result, benchmark_manifest=benchmark.queries.manifest, evaluator_contract_sha256=evaluator, seed=seed, repeat_index=repeat_index, output_path=output_path)


__all__ = ["GovernedRetrievalBenchmark", "build_governed_retrieval_benchmark", "materialize_governed_retrieval_run_evidence", "retrieval_evaluator_contract_sha256"]
