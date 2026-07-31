"""Transactional sparse-document replacement and posting insertion."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from tools.sparse_types import SparseField
from tools.sparse_utils import (
    _SCHEMA_VERSION,
    _identifier,
    _json_text,
    _normalize_owner_id,
    _optional_int,
)


class SparseWriteMixin:
    def _insert_posting(
        self,
        connection: sqlite3.Connection,
        *,
        owner_id: str,
        doc_id: str,
        field_id: str,
        term: str,
        positions: Sequence[int],
    ) -> None:
        connection.execute(
            """INSERT INTO sparse_postings(
                owner_id, doc_id, field_id, term, frequency, positions_json
            ) VALUES(?, ?, ?, ?, ?, ?)""",
            (
                owner_id,
                doc_id,
                field_id,
                term,
                len(positions),
                json.dumps(list(positions), separators=(",", ":")),
            ),
        )

    def replace_document(
        self,
        *,
        owner_id: str,
        doc_id: str,
        fields: Iterable[SparseField],
        profile_fingerprint: str = "",
        metadata: Mapping[str, Any] | None = None,
        expected_generation: int | None = None,
    ) -> int:
        owner = _normalize_owner_id(owner_id)
        document_id = _identifier(doc_id, "doc_id")
        fingerprint = self._profile_fingerprint(profile_fingerprint)
        metadata_json = _json_text(metadata, "document metadata")
        prepared = self._prepare_fields(fields)
        expected = _optional_int(
            expected_generation,
            "expected_generation",
            minimum=0,
            maximum=2_147_483_647,
        )
        field_count = len(prepared)
        token_count = sum(len(tokens) for _, tokens, _ in prepared)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT generation FROM sparse_documents WHERE owner_id=? AND doc_id=?",
                    (owner, document_id),
                ).fetchone()
                current = int(row[0]) if row is not None else 0
                if expected is not None and current != expected:
                    raise RuntimeError(
                        f"Sparse generation changed: expected {expected}, found {current}."
                    )
                generation = current + 1
                connection.execute(
                    "DELETE FROM sparse_documents WHERE owner_id=? AND doc_id=?",
                    (owner, document_id),
                )
                connection.execute(
                    """INSERT INTO sparse_documents(
                        owner_id, doc_id, generation, profile_fingerprint,
                        metadata_json, field_count, token_count, schema_version, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        owner,
                        document_id,
                        generation,
                        fingerprint,
                        metadata_json,
                        field_count,
                        token_count,
                        _SCHEMA_VERSION,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                for sparse_field, tokens, positions in prepared:
                    connection.execute(
                        """INSERT INTO sparse_fields(
                            owner_id, doc_id, field_id, field_type, position, text,
                            token_count, page_number, section, metadata_json
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            owner,
                            document_id,
                            sparse_field.field_id,
                            sparse_field.field_type,
                            sparse_field.position,
                            sparse_field.text,
                            len(tokens),
                            sparse_field.page_number,
                            sparse_field.section,
                            _json_text(sparse_field.metadata, "field metadata"),
                        ),
                    )
                    for term in sorted(positions):
                        self._insert_posting(
                            connection,
                            owner_id=owner,
                            doc_id=document_id,
                            field_id=sparse_field.field_id,
                            term=term,
                            positions=positions[term],
                        )
                connection.commit()
                return generation
            except Exception:
                connection.rollback()
                raise

