"""Durable owner-scoped store for finalized server-owned research results.

The store persists the final answer and authoritative ``Citation`` objects produced by
the server after all agent/gating steps. Raw user queries are not persisted here; only a
SHA-256 query fingerprint is stored. Result IDs are owner-bound content identities so a
cross-tenant equality oracle is not exposed through identifiers.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.models import AgentAnswer, Citation
from tools.security import normalize_owner_id

_MAX_ANSWER_CHARS = 100_000
_MAX_WARNINGS = 100
_MAX_CITATIONS = 500
_MAX_METADATA_BYTES = 1_000_000
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _safe_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    if len(str(absolute)) > 4096:
        raise ValueError("research result database path is too long")
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT):
            raise RuntimeError("research result database path may not traverse symlinks/reparse points")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _text(value: Any, label: str, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.replace("\x00", " ").strip()
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _canonical(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )
    if len(encoded.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValueError("research result payload exceeds the metadata limit")
    return encoded


def _citation_identity(citation: Citation) -> str:
    payload = {
        "source": citation.source_id or citation.url,
        "doc": citation.doc_id or "",
        "chunk": citation.chunk_id or "",
        "page": citation.page_number,
        "quote": citation.quote or citation.snippet or "",
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StoredResearchResult:
    owner_id: str
    result_id: str
    query_sha256: str
    answer: str
    citations: tuple[Citation, ...]
    warnings: tuple[str, ...]
    metadata: Mapping[str, Any]
    strategy: str
    model: str
    created_at: float

    @property
    def citation_ids(self) -> tuple[str, ...]:
        return tuple(_citation_identity(item) for item in self.citations)

    @property
    def fingerprint(self) -> str:
        return self.result_id


class ResearchResultStore:
    def __init__(self, path: str | Path) -> None:
        self.path = _safe_path(path)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_results (
                    owner_id TEXT NOT NULL,
                    result_id CHAR(64) NOT NULL,
                    query_sha256 CHAR(64) NOT NULL,
                    answer TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    warnings_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, result_id)
                );
                CREATE INDEX IF NOT EXISTS research_results_owner_time_idx
                  ON research_results(owner_id, created_at DESC, result_id);
                CREATE INDEX IF NOT EXISTS research_results_owner_query_idx
                  ON research_results(owner_id, query_sha256, created_at DESC);
                """
            )

    def put(
        self,
        owner_id: str,
        *,
        query_sha256: str,
        answer: AgentAnswer,
        strategy: str,
        model: str = "",
    ) -> StoredResearchResult:
        owner = normalize_owner_id(owner_id)
        query_digest = _sha(query_sha256, "query_sha256")
        if not isinstance(answer, AgentAnswer):
            raise TypeError("answer must be AgentAnswer")
        final_text = _text(answer.answer, "answer", _MAX_ANSWER_CHARS, allow_empty=True)
        citations = tuple(answer.citations or ())
        if len(citations) > _MAX_CITATIONS or any(not isinstance(item, Citation) for item in citations):
            raise ValueError("authoritative citations are invalid")
        citation_payload = [item.model_dump(exclude_none=True) for item in citations]
        warnings = tuple(_text(item, "warning", 5000) for item in (answer.warnings or ())[:_MAX_WARNINGS])
        metadata = dict(answer.metadata or {})
        strategy_value = _text(strategy, "strategy", 128)
        model_value = _text(model, "model", 256, allow_empty=True)
        identity_payload = {
            "owner_id": owner,
            "query_sha256": query_digest,
            "answer": final_text,
            "citations": [_citation_identity(item) for item in citations],
            "warnings": warnings,
            "metadata": metadata,
            "strategy": strategy_value,
            "model": model_value,
        }
        result_id = hashlib.sha256(_canonical(identity_payload).encode("utf-8")).hexdigest()
        created_at = time.time()
        citations_json = _canonical(citation_payload)
        warnings_json = _canonical(list(warnings))
        metadata_json = _canonical(metadata)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT answer,citations_json,warnings_json,metadata_json,strategy,model,created_at "
                    "FROM research_results WHERE owner_id=? AND result_id=?",
                    (owner, result_id),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """INSERT INTO research_results
                           (owner_id,result_id,query_sha256,answer,citations_json,warnings_json,
                            metadata_json,strategy,model,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        (
                            owner, result_id, query_digest, final_text, citations_json,
                            warnings_json, metadata_json, strategy_value, model_value, created_at,
                        ),
                    )
                else:
                    created_at = float(existing["created_at"])
                    if (
                        str(existing["answer"]) != final_text
                        or str(existing["citations_json"]) != citations_json
                        or str(existing["warnings_json"]) != warnings_json
                        or str(existing["metadata_json"]) != metadata_json
                        or str(existing["strategy"]) != strategy_value
                        or str(existing["model"]) != model_value
                    ):
                        raise RuntimeError("research result identity collision")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return StoredResearchResult(
            owner, result_id, query_digest, final_text, citations, warnings,
            metadata, strategy_value, model_value, created_at,
        )

    def get(self, owner_id: str, result_id: str) -> StoredResearchResult:
        owner = normalize_owner_id(owner_id)
        identifier = _sha(result_id, "result_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_results WHERE owner_id=? AND result_id=?",
                (owner, identifier),
            ).fetchone()
        if row is None:
            raise KeyError(identifier)
        citations_raw = json.loads(str(row["citations_json"]))
        citations = tuple(Citation(**item) for item in citations_raw)
        warnings = tuple(str(item) for item in json.loads(str(row["warnings_json"])))
        metadata = json.loads(str(row["metadata_json"]))
        return StoredResearchResult(
            owner_id=owner,
            result_id=identifier,
            query_sha256=_sha(str(row["query_sha256"]), "query_sha256"),
            answer=str(row["answer"]),
            citations=citations,
            warnings=warnings,
            metadata=metadata,
            strategy=str(row["strategy"]),
            model=str(row["model"]),
            created_at=float(row["created_at"]),
        )

    def list(self, owner_id: str, *, limit: int = 100) -> tuple[StoredResearchResult, ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT result_id FROM research_results WHERE owner_id=? "
                "ORDER BY created_at DESC,result_id LIMIT ?",
                (owner, limit),
            ).fetchall()
        return tuple(self.get(owner, str(row["result_id"])) for row in rows)


__all__ = ["ResearchResultStore", "StoredResearchResult"]
