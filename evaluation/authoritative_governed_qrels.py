"""Disk-backed governed qrels loading for large retrieval benchmarks.

The historical qrels loader materializes all relevant pairs in Python sets.  This module keeps
the same ``GovernedQrels``/receipt contract while backing ``relevant_by_query`` with a temporary
SQLite database.  Exact qrels semantics and receipt digests therefore remain compatible, but
the advertised large-record safety bound no longer implies corpus-sized Python memory.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from evaluation.governed_qrels import (
    GovernedQrels,
    GovernedQrelsReceipt,
    _digest,
    _entries,
    _finite,
    _identifier,
    _sha,
    _stream_sha,
)
from training.advanced_path_authority import safe_advanced_path

_MAX_BYTES = 2 * 1024 * 1024 * 1024
_MAX_RECORDS = 200_000_000


class _SQLiteRelevantMapping(Mapping[str, tuple[str, ...]]):
    def __init__(self, connection: sqlite3.Connection, database_path: Path) -> None:
        self._connection = connection
        self._database_path = database_path
        self._closed = False

    def _require_open(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("authoritative qrels mapping is closed")
        return self._connection

    def __getitem__(self, query_id: str) -> tuple[str, ...]:
        query = _identifier(query_id, "query_id")
        rows = self._require_open().execute(
            "SELECT document_id FROM pairs WHERE query_id=? ORDER BY document_id COLLATE BINARY",
            (query,),
        ).fetchall()
        if not rows:
            raise KeyError(query)
        return tuple(str(row[0]) for row in rows)

    def get(self, query_id: str, default: Any = None) -> Any:
        try:
            return self[query_id]
        except KeyError:
            return default

    def __iter__(self) -> Iterator[str]:
        cursor = self._require_open().execute(
            "SELECT DISTINCT query_id FROM pairs ORDER BY query_id COLLATE BINARY"
        )
        for (query_id,) in cursor:
            yield str(query_id)

    def __len__(self) -> int:
        return int(
            self._require_open()
            .execute("SELECT COUNT(DISTINCT query_id) FROM pairs")
            .fetchone()[0]
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._connection.close()
        finally:
            try:
                self._database_path.unlink()
            except FileNotFoundError:
                pass

    def __del__(self) -> None:  # pragma: no cover - explicit close is preferred
        try:
            self.close()
        except Exception:
            pass


def _open_store() -> tuple[sqlite3.Connection, Path]:
    descriptor, raw_path = tempfile.mkstemp(
        prefix="rigorousrag-qrels-",
        suffix=".sqlite3",
    )
    os.close(descriptor)
    path = Path(raw_path)
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE pairs ("
        "query_id TEXT NOT NULL, document_id TEXT NOT NULL, "
        "PRIMARY KEY(query_id,document_id)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE INDEX pairs_document_idx ON pairs(document_id)"
    )
    return connection, path


def load_authoritative_governed_qrels(
    path: str | Path,
    *,
    expected_sha256: str,
    input_format: str = "trec",
    minimum_relevance: float = 1.0,
    query_field: str = "query_id",
    document_field: str = "document_id",
    relevance_field: str = "relevance",
) -> GovernedQrels:
    source = safe_advanced_path(
        path,
        label="authoritative qrels source",
        must_exist=True,
        require_file=True,
    )
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES:
        raise ValueError("qrels source exceeds byte safety bound")
    source_sha = _sha(expected_sha256, "expected_sha256")
    if _stream_sha(source) != source_sha:
        raise ValueError("qrels source digest differs from expected immutable bytes")
    selected_format = _identifier(input_format, "input_format", 20).lower()
    if selected_format not in {"json", "jsonl", "trec"}:
        raise ValueError("input_format must be json, jsonl or trec")
    selected_query_field = _identifier(query_field, "query_field", 300)
    selected_document_field = _identifier(document_field, "document_field", 300)
    selected_relevance_field = _identifier(relevance_field, "relevance_field", 300)
    threshold = _finite(minimum_relevance, "minimum_relevance")

    connection, database = _open_store()
    try:
        count = 0
        for entry in _entries(
            source,
            input_format=selected_format,
            query_field=selected_query_field,
            document_field=selected_document_field,
            relevance_field=selected_relevance_field,
        ):
            count += 1
            if count > _MAX_RECORDS:
                raise ValueError("qrels source exceeds record safety bound")
            if entry.relevance < threshold:
                continue
            connection.execute(
                "INSERT OR IGNORE INTO pairs(query_id,document_id) VALUES (?,?)",
                (entry.query_id, entry.document_id),
            )
            if count % 20_000 == 0:
                connection.commit()
        connection.commit()

        pair_count = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
        query_count = int(
            connection.execute("SELECT COUNT(DISTINCT query_id) FROM pairs").fetchone()[0]
        )
        document_count = int(
            connection.execute("SELECT COUNT(DISTINCT document_id) FROM pairs").fetchone()[0]
        )
        pair_digest = hashlib.sha256()
        for query_id, document_id in connection.execute(
            "SELECT query_id,document_id FROM pairs "
            "ORDER BY query_id COLLATE BINARY,document_id COLLATE BINARY"
        ):
            pair_digest.update(str(query_id).encode("utf-8"))
            pair_digest.update(b"\t")
            pair_digest.update(str(document_id).encode("utf-8"))
            pair_digest.update(b"\n")
        pair_sha = pair_digest.hexdigest()
        unsigned = {
            "schema": "rigorousrag-governed-qrels-receipt/v2",
            "source_path": str(source),
            "source_sha256": source_sha,
            "input_format": selected_format,
            "minimum_relevance": threshold,
            "query_field": selected_query_field,
            "document_field": selected_document_field,
            "relevance_field": selected_relevance_field,
            "pair_count": pair_count,
            "query_count": query_count,
            "document_count": document_count,
            "relevant_pair_sha256": pair_sha,
        }
        receipt = GovernedQrelsReceipt(
            str(source),
            source_sha,
            selected_format,
            threshold,
            selected_query_field,
            selected_document_field,
            selected_relevance_field,
            pair_count,
            query_count,
            document_count,
            pair_sha,
            _digest(unsigned),
        )
        return GovernedQrels(receipt, _SQLiteRelevantMapping(connection, database))
    except Exception:
        connection.close()
        try:
            database.unlink()
        except FileNotFoundError:
            pass
        raise


def close_authoritative_governed_qrels(qrels: GovernedQrels) -> None:
    if not isinstance(qrels, GovernedQrels):
        raise ValueError("qrels must be GovernedQrels")
    mapping = qrels.relevant_by_query
    if isinstance(mapping, _SQLiteRelevantMapping):
        mapping.close()


__all__ = [
    "close_authoritative_governed_qrels",
    "load_authoritative_governed_qrels",
]
