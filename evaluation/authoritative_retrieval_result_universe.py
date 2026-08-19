"""Exact result-row/query-universe proof for authoritative retrieval evaluation."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from typing import TYPE_CHECKING

from evaluation.benchmark_run_evidence import _normalize_row
from evaluation.benchmark_suite import BenchmarkSuiteResult

if TYPE_CHECKING:
    from evaluation.authoritative_governed_retrieval_benchmark import (
        AuthoritativeGovernedRetrievalBenchmark,
    )

_MAX_SAMPLE = 100


def assert_authoritative_retrieval_result_universe(
    result: BenchmarkSuiteResult,
    benchmark: "AuthoritativeGovernedRetrievalBenchmark",
) -> str:
    """Require one canonical result row for every governed benchmark query, and no others."""
    from evaluation.authoritative_governed_retrieval_benchmark import (
        AuthoritativeGovernedRetrievalBenchmark,
    )

    if not isinstance(result, BenchmarkSuiteResult):
        raise ValueError("result must be BenchmarkSuiteResult")
    if not isinstance(benchmark, AuthoritativeGovernedRetrievalBenchmark):
        raise ValueError("benchmark must be AuthoritativeGovernedRetrievalBenchmark")
    descriptor, raw_database = tempfile.mkstemp(
        prefix="rigorousrag-result-universe-",
        suffix=".sqlite3",
    )
    os.close(descriptor)
    try:
        connection = sqlite3.connect(raw_database)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("CREATE TABLE queries(id TEXT PRIMARY KEY) WITHOUT ROWID")
            connection.execute("CREATE TABLE results(id TEXT PRIMARY KEY) WITHOUT ROWID")
            query_count = 0
            for split in benchmark.queries.manifest.splits:
                for example in benchmark.queries.split(split.name):
                    try:
                        connection.execute("INSERT INTO queries(id) VALUES (?)", (example.example_id,))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(
                            f"governed query id {example.example_id!r} is duplicated across benchmark splits"
                        ) from exc
                    query_count += 1
                    if query_count % 20_000 == 0:
                        connection.commit()
            connection.commit()
            if query_count <= 0:
                raise ValueError("authoritative retrieval benchmark contains no queries")

            result_count = 0
            for row in result.rows:
                normalized = _normalize_row(row)
                example_id = str(normalized["example_id"])
                try:
                    connection.execute("INSERT INTO results(id) VALUES (?)", (example_id,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        f"retrieval result repeats example id {example_id!r}"
                    ) from exc
                result_count += 1
                if result_count % 20_000 == 0:
                    connection.commit()
            connection.commit()

            missing = [
                str(row[0])
                for row in connection.execute(
                    "SELECT q.id FROM queries q LEFT JOIN results r ON q.id=r.id "
                    "WHERE r.id IS NULL ORDER BY q.id COLLATE BINARY LIMIT ?",
                    (_MAX_SAMPLE,),
                )
            ]
            extra = [
                str(row[0])
                for row in connection.execute(
                    "SELECT r.id FROM results r LEFT JOIN queries q ON r.id=q.id "
                    "WHERE q.id IS NULL ORDER BY r.id COLLATE BINARY LIMIT ?",
                    (_MAX_SAMPLE,),
                )
            ]
            if query_count != result_count or missing or extra:
                raise ValueError(
                    "retrieval result/query universes differ; "
                    f"query_count={query_count} result_count={result_count} "
                    f"missing_result_sample={missing} extra_result_sample={extra}"
                )
            digest = hashlib.sha256()
            for (query_id,) in connection.execute(
                "SELECT id FROM results ORDER BY id COLLATE BINARY"
            ):
                digest.update(str(query_id).encode("utf-8"))
                digest.update(b"\n")
            universe_sha = digest.hexdigest()
            if universe_sha != benchmark.query_universe_sha256:
                raise ValueError(
                    "retrieval result universe digest differs from authoritative benchmark query universe"
                )
            return universe_sha
        finally:
            connection.close()
    finally:
        try:
            os.unlink(raw_database)
        except FileNotFoundError:
            pass


__all__ = ["assert_authoritative_retrieval_result_universe"]
