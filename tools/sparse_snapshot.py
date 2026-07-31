"""Exact sparse generation snapshots, restore, deletion and enumeration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tools.sparse_types import (
    SparseDocumentSnapshot,
    SparseField,
    SparseFieldSnapshot,
)
from tools.sparse_utils import (
    _SCHEMA_VERSION,
    _exact_int,
    _field_type,
    _identifier,
    _json_text,
    _normalize_owner_id,
    _strict_json,
)


class SparseSnapshotMixin:
    def snapshot_document(
        self,
        *,
        owner_id: str,
        doc_id: str,
    ) -> SparseDocumentSnapshot | None:
        owner = _normalize_owner_id(owner_id)
        document_id = _identifier(doc_id, "doc_id")
        with self._lock, self._connect() as connection:
            document = connection.execute(
                """SELECT generation, profile_fingerprint, metadata_json,
                           field_count, token_count, schema_version
                    FROM sparse_documents WHERE owner_id=? AND doc_id=?""",
                (owner, document_id),
            ).fetchone()
            if document is None:
                return None
            if int(document[5]) != _SCHEMA_VERSION:
                raise RuntimeError("Stored sparse document schema is incompatible.")
            rows = connection.execute(
                """SELECT field_id, field_type, text, position, token_count,
                           page_number, section, metadata_json
                    FROM sparse_fields WHERE owner_id=? AND doc_id=?
                    ORDER BY position, field_id""",
                (owner, document_id),
            ).fetchall()
            if len(rows) != int(document[3]):
                raise RuntimeError("Stored sparse document field count is corrupt.")
            fields: list[SparseFieldSnapshot] = []
            observed_tokens = 0
            for row in rows:
                metadata = _strict_json(str(row[7]), "sparse field metadata")
                token_count = int(row[4])
                observed_tokens += token_count
                fields.append(
                    SparseFieldSnapshot(
                        field_id=_identifier(row[0], "stored field_id"),
                        field_type=_field_type(row[1]),
                        text=str(row[2]),
                        position=int(row[3]),
                        token_count=token_count,
                        page_number=int(row[5]) if row[5] is not None else None,
                        section=str(row[6]) if row[6] is not None else None,
                        metadata=metadata,
                    )
                )
            if observed_tokens != int(document[4]):
                raise RuntimeError("Stored sparse document token count is corrupt.")
            return SparseDocumentSnapshot(
                owner_id=owner,
                doc_id=document_id,
                generation=int(document[0]),
                profile_fingerprint=self._profile_fingerprint(str(document[1])),
                metadata=_strict_json(str(document[2]), "sparse document metadata"),
                fields=tuple(fields),
            )

    def restore_document(
        self,
        *,
        owner_id: str,
        doc_id: str,
        snapshot: SparseDocumentSnapshot | None,
    ) -> None:
        owner = _normalize_owner_id(owner_id)
        document_id = _identifier(doc_id, "doc_id")
        if snapshot is not None:
            if snapshot.owner_id != owner or snapshot.doc_id != document_id:
                raise ValueError("Snapshot owner/document identity does not match the restore target.")
            if snapshot.schema_version != _SCHEMA_VERSION:
                raise ValueError("Snapshot schema version is incompatible.")
            prepared = self._prepare_fields(
                SparseField(
                    field_id=item.field_id,
                    field_type=item.field_type,
                    text=item.text,
                    position=item.position,
                    page_number=item.page_number,
                    section=item.section,
                    metadata=item.metadata,
                )
                for item in snapshot.fields
            )
            snapshot_counts = {item.field_id: item.token_count for item in snapshot.fields}
            if len(snapshot_counts) != len(snapshot.fields) or any(
                len(tokens) != snapshot_counts.get(sparse_field.field_id)
                for sparse_field, tokens, _ in prepared
            ):
                raise ValueError("Snapshot token counts do not match the reconstructable fields.")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "DELETE FROM sparse_documents WHERE owner_id=? AND doc_id=?",
                    (owner, document_id),
                )
                if snapshot is not None:
                    connection.execute(
                        """INSERT INTO sparse_documents(
                            owner_id, doc_id, generation, profile_fingerprint,
                            metadata_json, field_count, token_count, schema_version, updated_at
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            owner,
                            document_id,
                            snapshot.generation,
                            self._profile_fingerprint(snapshot.profile_fingerprint),
                            _json_text(snapshot.metadata, "snapshot document metadata"),
                            len(prepared),
                            sum(len(tokens) for _, tokens, _ in prepared),
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
                                _json_text(sparse_field.metadata, "snapshot field metadata"),
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
            except Exception:
                connection.rollback()
                raise

    def delete_document(self, *, owner_id: str, doc_id: str) -> bool:
        owner = _normalize_owner_id(owner_id)
        document_id = _identifier(doc_id, "doc_id")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "DELETE FROM sparse_documents WHERE owner_id=? AND doc_id=?",
                (owner, document_id),
            )
            connection.commit()
            return cursor.rowcount == 1

    def document_exists(self, *, owner_id: str, doc_id: str) -> bool:
        owner = _normalize_owner_id(owner_id)
        document_id = _identifier(doc_id, "doc_id")
        with self._lock, self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM sparse_documents WHERE owner_id=? AND doc_id=?",
                (owner, document_id),
            ).fetchone() is not None

    def list_document_ids(self, *, owner_id: str, limit: int = 10_000) -> tuple[str, ...]:
        owner = _normalize_owner_id(owner_id)
        bounded = _exact_int(limit, "limit", minimum=1, maximum=100_000)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT doc_id FROM sparse_documents WHERE owner_id=? ORDER BY doc_id LIMIT ?",
                (owner, bounded),
            ).fetchall()
            return tuple(str(row[0]) for row in rows)
