"""SQLite schema, path identity and shared preparation for the sparse index."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from tools.sparse_types import SparseField
from tools.sparse_utils import (
    _MAX_DOCUMENT_CHARS,
    _MAX_DOCUMENT_TOKENS,
    _MAX_FIELDS,
    _MAX_UNIQUE_TERMS_PER_FIELD,
    _SCHEMA_VERSION,
    _identity,
    _is_redirecting,
    _reject_redirecting_components,
    tokenize,
)


class SparseBackend:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        if not isinstance(path, (str, os.PathLike)):
            raise ValueError("Sparse index path must be a filesystem path.")
        rendered = os.fspath(path)
        if not isinstance(rendered, str) or not rendered or len(rendered) > 4_096 or any(
            ord(character) < 32 or ord(character) == 127 for character in rendered
        ):
            raise ValueError("Sparse index path is invalid.")
        raw = Path(rendered)
        if not raw.is_absolute():
            raw = Path.cwd() / raw
        self.path = Path(os.path.abspath(raw))
        _reject_redirecting_components(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _reject_redirecting_components(self.path)
        self._lock = threading.RLock()
        self._initialize()
        self._parent_identity = _identity(self.path.parent)
        self._database_identity = _identity(self.path)

    def _verify_identity(self) -> None:
        _reject_redirecting_components(self.path)
        try:
            parent_identity = _identity(self.path.parent)
            database_identity = _identity(self.path)
        except FileNotFoundError as exc:
            raise RuntimeError("Sparse index path disappeared.") from exc
        if parent_identity != self._parent_identity:
            raise RuntimeError("Sparse index parent directory was replaced.")
        if database_identity != self._database_identity:
            raise RuntimeError("Sparse index database file was replaced.")

    def _connect(self) -> sqlite3.Connection:
        self._verify_identity()
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sparse_schema (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    schema_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sparse_documents (
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    profile_fingerprint TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    field_count INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    schema_version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_id, doc_id)
                );
                CREATE TABLE IF NOT EXISTS sparse_fields (
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    field_id TEXT NOT NULL,
                    field_type TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    page_number INTEGER,
                    section TEXT,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY(owner_id, doc_id, field_id),
                    FOREIGN KEY(owner_id, doc_id)
                        REFERENCES sparse_documents(owner_id, doc_id)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS sparse_postings (
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    field_id TEXT NOT NULL,
                    term TEXT NOT NULL,
                    frequency INTEGER NOT NULL,
                    positions_json TEXT NOT NULL,
                    PRIMARY KEY(owner_id, doc_id, field_id, term),
                    FOREIGN KEY(owner_id, doc_id, field_id)
                        REFERENCES sparse_fields(owner_id, doc_id, field_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS sparse_postings_owner_term
                    ON sparse_postings(owner_id, term, doc_id);
                CREATE INDEX IF NOT EXISTS sparse_fields_owner_type
                    ON sparse_fields(owner_id, field_type, doc_id);
                """
            )
            row = connection.execute(
                "SELECT schema_version FROM sparse_schema WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO sparse_schema(singleton, schema_version) VALUES(1, ?)",
                    (_SCHEMA_VERSION,),
                )
            elif int(row[0]) != _SCHEMA_VERSION:
                raise RuntimeError("Sparse index schema version is incompatible.")
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _profile_fingerprint(value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("profile_fingerprint must be a string.")
        cleaned = value.strip().lower()
        if cleaned and (len(cleaned) != 64 or not all(ch in "0123456789abcdef" for ch in cleaned)):
            raise ValueError("profile_fingerprint must be an empty string or a SHA-256 hex digest.")
        return cleaned

    @staticmethod
    def _prepare_fields(fields: Iterable[SparseField]) -> list[tuple[SparseField, tuple[str, ...], dict[str, list[int]]]]:
        if isinstance(fields, (str, bytes, bytearray)):
            raise ValueError("fields must be an iterable of SparseField values.")
        prepared: list[tuple[SparseField, tuple[str, ...], dict[str, list[int]]]] = []
        seen_ids: set[str] = set()
        total_chars = 0
        total_tokens = 0
        for raw in fields:
            if len(prepared) >= _MAX_FIELDS:
                raise ValueError("Document exceeds the sparse field limit.")
            if not isinstance(raw, SparseField):
                raise ValueError("Every sparse field must be a SparseField.")
            if raw.field_id in seen_ids:
                raise ValueError(f"Duplicate sparse field ID: {raw.field_id}.")
            seen_ids.add(raw.field_id)
            tokens = tokenize(raw.text)
            if not tokens:
                raise ValueError(f"Sparse field {raw.field_id!r} produced no tokens.")
            total_chars += len(raw.text)
            total_tokens += len(tokens)
            if total_chars > _MAX_DOCUMENT_CHARS:
                raise ValueError("Document exceeds the sparse character limit.")
            if total_tokens > _MAX_DOCUMENT_TOKENS:
                raise ValueError("Document exceeds the sparse token limit.")
            positions: defaultdict[str, list[int]] = defaultdict(list)
            for index, token in enumerate(tokens):
                positions[token].append(index)
            if len(positions) > _MAX_UNIQUE_TERMS_PER_FIELD:
                raise ValueError("Sparse field exceeds the unique-term limit.")
            prepared.append((raw, tokens, dict(positions)))
        if not prepared:
            raise ValueError("At least one sparse field is required.")
        prepared.sort(key=lambda item: (item[0].position, item[0].field_id))
        return prepared

    def ping(self) -> bool:
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute("SELECT schema_version FROM sparse_schema WHERE singleton=1").fetchone()
                return row is not None and int(row[0]) == _SCHEMA_VERSION
        except Exception:
            return False

