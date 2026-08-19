"""Authoritative retrieval benchmark composition for promotion-grade evaluation.

The v2 path binds only authoritative components: v2 query import, passed disk-backed leakage
qualification, disk-backed qrels, and the v2 closed corpus publication. Relevant-document
coverage and qrels pair identity are proved with SQLite/streaming hashes. Promotion-grade run
evidence is emitted through the streaming v2 result-artifact authority.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from evaluation.advanced_rag_receipts import AdvancedEvaluationRun
from evaluation.authoritative_benchmark_run_evidence import (
    AuthoritativeBenchmarkResultReceipt,
    materialize_authoritative_benchmark_run_evidence,
)
from evaluation.authoritative_governed_benchmark_corpus import (
    AuthoritativeBenchmarkCorpusReceipt,
    iter_authoritative_benchmark_corpus,
    verify_authoritative_benchmark_corpus_receipt,
)
from evaluation.authoritative_governed_benchmark_io import VerifiedAuthoritativeGovernedBenchmark
from evaluation.authoritative_governed_qrels import _SQLiteRelevantMapping
from evaluation.benchmark_suite import BenchmarkSuiteResult
from evaluation.governed_benchmark_qualification import GovernedBenchmarkLeakageReceipt
from evaluation.governed_qrels import GovernedQrels, overlay_qrels
from tools.benchmark_adapters import BenchmarkExample

_HEX = frozenset("0123456789abcdef")
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


def _assert_authoritative_qrels(qrels: GovernedQrels) -> None:
    if not isinstance(qrels, GovernedQrels):
        raise ValueError("qrels must be GovernedQrels")
    if not isinstance(qrels.relevant_by_query, _SQLiteRelevantMapping):
        raise ValueError("promotion-grade retrieval requires disk-backed authoritative governed qrels")
    if qrels.receipt.pair_count <= 0 or qrels.receipt.query_count <= 0 or qrels.receipt.document_count <= 0:
        raise ValueError("authoritative qrels must contain at least one relevant pair/query/document")


def _coverage(qrels: GovernedQrels, corpus: AuthoritativeBenchmarkCorpusReceipt) -> str:
    descriptor, raw_database = tempfile.mkstemp(prefix="rigorousrag-retrieval-coverage-", suffix=".sqlite3")
    os.close(descriptor)
    missing: list[str] = []
    pair_digest = hashlib.sha256(); pair_count = 0
    try:
        connection = sqlite3.connect(raw_database)
        try:
            connection.execute("PRAGMA journal_mode=OFF"); connection.execute("PRAGMA synchronous=OFF"); connection.execute("PRAGMA temp_store=FILE")
            connection.execute("CREATE TABLE corpus_ids(document_id TEXT PRIMARY KEY) WITHOUT ROWID")
            count = 0
            for document in iter_authoritative_benchmark_corpus(corpus):
                try:
                    connection.execute("INSERT INTO corpus_ids(document_id) VALUES (?)", (document.document_id,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"authoritative corpus contains duplicate document id {document.document_id!r}") from exc
                count += 1
                if count % 20_000 == 0: connection.commit()
            connection.commit()
            if count != corpus.record_count:
                raise ValueError("authoritative corpus iteration count differs from corpus receipt")
            for query_id in qrels.relevant_by_query:
                for document_id in qrels.relevant_by_query[query_id]:
                    if connection.execute("SELECT 1 FROM corpus_ids WHERE document_id=?", (document_id,)).fetchone() is None:
                        if len(missing) < _MAX_MISSING_SAMPLE: missing.append(document_id)
                    pair_digest.update(query_id.encode("utf-8")); pair_digest.update(b"\t"); pair_digest.update(document_id.encode("utf-8")); pair_digest.update(b"\n"); pair_count += 1
            if missing:
                raise ValueError(f"qrels reference documents absent from authoritative corpus; sample={missing}")
        finally:
            connection.close()
    finally:
        try: os.unlink(raw_database)
        except FileNotFoundError: pass
    if pair_count != qrels.receipt.pair_count:
        raise ValueError("authoritative qrels mapping pair count differs from receipt")
    coverage_sha = pair_digest.hexdigest()
    if coverage_sha != qrels.receipt.relevant_pair_sha256:
        raise ValueError("authoritative qrels mapping pair digest differs from qrels receipt")
    return coverage_sha


@dataclass(frozen=True)
class AuthoritativeGovernedRetrievalBenchmark:
    queries: VerifiedAuthoritativeGovernedBenchmark
    leakage: GovernedBenchmarkLeakageReceipt
    qrels: GovernedQrels
    corpus: AuthoritativeBenchmarkCorpusReceipt
    qrels_coverage_sha256: str
    contract_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.queries, VerifiedAuthoritativeGovernedBenchmark):
            raise ValueError("queries must be VerifiedAuthoritativeGovernedBenchmark")
        if not isinstance(self.leakage, GovernedBenchmarkLeakageReceipt) or not self.leakage.passed:
            raise ValueError("leakage must be a passed GovernedBenchmarkLeakageReceipt")
        _assert_authoritative_qrels(self.qrels)
        if not isinstance(self.corpus, AuthoritativeBenchmarkCorpusReceipt):
            raise ValueError("corpus must be AuthoritativeBenchmarkCorpusReceipt")
        object.__setattr__(self, "qrels_coverage_sha256", _sha(self.qrels_coverage_sha256, "qrels_coverage_sha256"))
        object.__setattr__(self, "contract_sha256", _sha(self.contract_sha256, "contract_sha256"))
        expected = _digest({
            "schema": "rigorousrag-authoritative-governed-retrieval-benchmark/v2",
            "query_manifest_sha256": self.queries.manifest.manifest_digest,
            "query_import_receipt_sha256": self.queries.receipt.receipt_sha256,
            "leakage_receipt_sha256": self.leakage.receipt_sha256,
            "qrels_receipt_sha256": self.qrels.receipt.receipt_sha256,
            "corpus_receipt_sha256": self.corpus.receipt_sha256,
            "corpus_publication_contract_sha256": self.corpus.publication_contract_sha256,
            "qrels_coverage_sha256": self.qrels_coverage_sha256,
        })
        if expected != self.contract_sha256:
            raise ValueError("authoritative retrieval benchmark contract digest mismatch")

    def split(self, name: str) -> Iterator[BenchmarkExample]:
        return overlay_qrels(self.queries.split(name), self.qrels, require_query_labels=True, require_existing_equal=True)


def build_authoritative_governed_retrieval_benchmark(
    queries: VerifiedAuthoritativeGovernedBenchmark,
    *,
    leakage: GovernedBenchmarkLeakageReceipt,
    qrels: GovernedQrels,
    corpus_receipt_path: str | Path,
) -> AuthoritativeGovernedRetrievalBenchmark:
    if not isinstance(queries, VerifiedAuthoritativeGovernedBenchmark):
        raise ValueError("queries must be VerifiedAuthoritativeGovernedBenchmark")
    if not isinstance(leakage, GovernedBenchmarkLeakageReceipt) or not leakage.passed:
        raise ValueError("retrieval benchmark requires passed leakage evidence")
    if leakage.dataset_manifest_sha256 != queries.manifest.manifest_digest or leakage.import_receipt_sha256 != queries.receipt.receipt_sha256:
        raise ValueError("leakage receipt differs from authoritative query benchmark")
    _assert_authoritative_qrels(qrels)
    corpus = verify_authoritative_benchmark_corpus_receipt(corpus_receipt_path)
    coverage = _coverage(qrels, corpus)
    contract = _digest({
        "schema": "rigorousrag-authoritative-governed-retrieval-benchmark/v2",
        "query_manifest_sha256": queries.manifest.manifest_digest,
        "query_import_receipt_sha256": queries.receipt.receipt_sha256,
        "leakage_receipt_sha256": leakage.receipt_sha256,
        "qrels_receipt_sha256": qrels.receipt.receipt_sha256,
        "corpus_receipt_sha256": corpus.receipt_sha256,
        "corpus_publication_contract_sha256": corpus.publication_contract_sha256,
        "qrels_coverage_sha256": coverage,
    })
    return AuthoritativeGovernedRetrievalBenchmark(queries, leakage, qrels, corpus, coverage, contract)


def authoritative_retrieval_evaluator_contract_sha256(base_evaluator_contract_sha256: str, benchmark: AuthoritativeGovernedRetrievalBenchmark) -> str:
    if not isinstance(benchmark, AuthoritativeGovernedRetrievalBenchmark):
        raise ValueError("benchmark must be AuthoritativeGovernedRetrievalBenchmark")
    return _digest({
        "schema": "rigorousrag-authoritative-retrieval-evaluator-contract/v2",
        "base_evaluator_contract_sha256": _sha(base_evaluator_contract_sha256, "base_evaluator_contract_sha256"),
        "authoritative_retrieval_benchmark_sha256": benchmark.contract_sha256,
    })


def materialize_authoritative_retrieval_run_evidence(
    result: BenchmarkSuiteResult,
    *,
    benchmark: AuthoritativeGovernedRetrievalBenchmark,
    base_evaluator_contract_sha256: str,
    seed: int,
    repeat_index: int,
    output_dir: str | Path,
) -> tuple[AdvancedEvaluationRun, AuthoritativeBenchmarkResultReceipt]:
    evaluator = authoritative_retrieval_evaluator_contract_sha256(base_evaluator_contract_sha256, benchmark)
    return materialize_authoritative_benchmark_run_evidence(
        result,
        benchmark_manifest=benchmark.queries.manifest,
        evaluator_contract_sha256=evaluator,
        seed=seed,
        repeat_index=repeat_index,
        output_dir=output_dir,
    )


__all__ = ["AuthoritativeGovernedRetrievalBenchmark", "authoritative_retrieval_evaluator_contract_sha256", "build_authoritative_governed_retrieval_benchmark", "materialize_authoritative_retrieval_run_evidence"]
