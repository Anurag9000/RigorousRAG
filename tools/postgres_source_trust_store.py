"""PostgreSQL reviewed source-trust registry with transactional activation outbox."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from tools.postgres_research_stores import _PostgresMixin, _row
from tools.security import normalize_owner_id
from tools.source_trust import SourceTrustFeatures
from tools.source_trust_store import (
    SourceTrustActivation,
    SourceTrustRevision,
    SourceTrustStore,
    _bounded_source_id,
    _canonical,
)
from tools.sql_control_plane import ConnectionFactory, CursorLike


class PostgresSourceTrustStore(_PostgresMixin, SourceTrustStore):
    def __init__(
        self,
        connection_factory: ConnectionFactory,
        *,
        schema: str = "rigorousrag",
        initialize: bool = True,
    ) -> None:
        _PostgresMixin.__init__(
            self,
            connection_factory,
            schema=schema,
            initialize=initialize,
        )

    def _initialize_postgres(self) -> None:
        schema = self.schema
        self._initialize_tables(
            (
                f"""CREATE TABLE IF NOT EXISTS {schema}.source_trust_revisions (
                    owner_id TEXT NOT NULL,
                    revision_id CHAR(64) NOT NULL,
                    source_id TEXT NOT NULL,
                    features_json JSONB NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    review_basis TEXT NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY(owner_id,revision_id)
                )""",
                f"CREATE INDEX IF NOT EXISTS source_trust_source_idx ON {schema}.source_trust_revisions(owner_id,source_id,created_at DESC,revision_id DESC)",
                f"""CREATE TABLE IF NOT EXISTS {schema}.source_trust_heads (
                    owner_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    revision_id CHAR(64) NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY(owner_id,source_id)
                )""",
                f"CREATE INDEX IF NOT EXISTS source_trust_heads_updated_idx ON {schema}.source_trust_heads(owner_id,updated_at DESC,source_id)",
                f"""CREATE TABLE IF NOT EXISTS {schema}.source_trust_activations (
                    owner_id TEXT NOT NULL,
                    activation_id CHAR(64) NOT NULL,
                    source_id TEXT NOT NULL,
                    previous_revision_id TEXT NOT NULL DEFAULT '',
                    revision_id CHAR(64) NOT NULL,
                    activated_at DOUBLE PRECISION NOT NULL,
                    invalidation_completed_at DOUBLE PRECISION,
                    last_error TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(owner_id,activation_id)
                )""",
                f"CREATE INDEX IF NOT EXISTS source_trust_activations_pending_idx ON {schema}.source_trust_activations(owner_id,invalidation_completed_at,activated_at,activation_id)",
                f"CREATE INDEX IF NOT EXISTS source_trust_activations_source_idx ON {schema}.source_trust_activations(owner_id,source_id,activated_at,activation_id)",
            )
        )

        def backfill(cursor: CursorLike) -> None:
            cursor.execute(
                f"""INSERT INTO {schema}.source_trust_heads(owner_id,source_id,revision_id,updated_at)
                    SELECT DISTINCT ON (r.owner_id,r.source_id)
                           r.owner_id,r.source_id,r.revision_id,r.created_at
                    FROM {schema}.source_trust_revisions r
                    ORDER BY r.owner_id,r.source_id,r.created_at DESC,r.revision_id DESC
                    ON CONFLICT(owner_id,source_id) DO NOTHING"""
            )

        self._transaction(backfill)

    @staticmethod
    def _revision_from_row(row: Sequence[Any] | Mapping[str, Any]) -> SourceTrustRevision:
        raw = _row(row, "features_json", 3)
        if isinstance(raw, Mapping):
            value = dict(raw)
        else:
            value = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
        return SourceTrustRevision(
            owner_id=str(_row(row, "owner_id", 0)),
            revision_id=str(_row(row, "revision_id", 1)),
            features=SourceTrustFeatures(**value),
            reviewer_id=str(_row(row, "reviewer_id", 4)),
            review_basis=str(_row(row, "review_basis", 5)),
            created_at=float(_row(row, "created_at", 6)),
        )

    @staticmethod
    def _activation_from_row(
        row: Sequence[Any] | Mapping[str, Any],
    ) -> SourceTrustActivation:
        completed = _row(row, "invalidation_completed_at", 6)
        return SourceTrustActivation(
            owner_id=str(_row(row, "owner_id", 0)),
            activation_id=str(_row(row, "activation_id", 1)),
            source_id=str(_row(row, "source_id", 2)),
            previous_revision_id=str(_row(row, "previous_revision_id", 3)),
            revision_id=str(_row(row, "revision_id", 4)),
            activated_at=float(_row(row, "activated_at", 5)),
            invalidation_completed_at=float(completed) if completed is not None else None,
            last_error=str(_row(row, "last_error", 7) or ""),
        )

    @staticmethod
    def _activation_id(
        owner_id: str,
        source_id: str,
        previous_revision_id: str,
        revision_id: str,
    ) -> str:
        # Preserve occurrence semantics from the reference store without relying on a
        # process-global sequence. Time nanoseconds plus a random UUID are hashed into the
        # activation identity and never become authorization state.
        import uuid

        payload = {
            "owner_id": owner_id,
            "source_id": source_id,
            "previous_revision_id": previous_revision_id,
            "revision_id": revision_id,
            "nonce": f"{time.time_ns()}:{uuid.uuid4().hex}",
        }
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    def put(
        self,
        owner_id: str,
        features: SourceTrustFeatures,
        *,
        reviewer_id: str,
        review_basis: str,
    ) -> SourceTrustRevision:
        owner = normalize_owner_id(owner_id)
        if not isinstance(features, SourceTrustFeatures):
            raise TypeError("features must be SourceTrustFeatures")
        reviewer = str(reviewer_id or "").strip()
        basis = str(review_basis or "").strip()
        if not reviewer or len(reviewer) > 256 or not basis or len(basis) > 5000:
            raise ValueError("reviewer_id and review_basis are required and bounded")
        payload = {
            "owner_id": owner,
            "features": asdict(features),
            "reviewer_id": reviewer,
            "review_basis": basis,
        }
        revision_id = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        created_at = time.time()
        features_json = _canonical(asdict(features))

        def operation(cursor: CursorLike) -> SourceTrustRevision:
            cursor.execute(
                f"""SELECT revision_id FROM {self.schema}.source_trust_heads
                    WHERE owner_id=%s AND source_id=%s FOR UPDATE""",
                (owner, features.source_id),
            )
            head = cursor.fetchone()
            previous_revision_id = str(_row(head, "revision_id", 0)) if head is not None else ""
            cursor.execute(
                f"""INSERT INTO {self.schema}.source_trust_revisions
                    (owner_id,revision_id,source_id,features_json,reviewer_id,review_basis,created_at)
                    VALUES(%s,%s,%s,%s::jsonb,%s,%s,%s)
                    ON CONFLICT(owner_id,revision_id) DO NOTHING""",
                (
                    owner,
                    revision_id,
                    features.source_id,
                    features_json,
                    reviewer,
                    basis,
                    created_at,
                ),
            )
            cursor.execute(
                f"""SELECT owner_id,revision_id,source_id,features_json::text,reviewer_id,review_basis,created_at
                    FROM {self.schema}.source_trust_revisions
                    WHERE owner_id=%s AND revision_id=%s""",
                (owner, revision_id),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("source trust revision persistence failed")
            stored = self._revision_from_row(row)
            if stored.features != features or stored.reviewer_id != reviewer or stored.review_basis != basis:
                raise RuntimeError("source trust revision identity collision")
            if previous_revision_id != revision_id:
                activation_id = self._activation_id(
                    owner,
                    features.source_id,
                    previous_revision_id,
                    revision_id,
                )
                activated_at = time.time()
                cursor.execute(
                    f"""INSERT INTO {self.schema}.source_trust_activations
                        (owner_id,activation_id,source_id,previous_revision_id,revision_id,activated_at)
                        VALUES(%s,%s,%s,%s,%s,%s)""",
                    (
                        owner,
                        activation_id,
                        features.source_id,
                        previous_revision_id,
                        revision_id,
                        activated_at,
                    ),
                )
                cursor.execute(
                    f"""INSERT INTO {self.schema}.source_trust_heads(owner_id,source_id,revision_id,updated_at)
                        VALUES(%s,%s,%s,%s)
                        ON CONFLICT(owner_id,source_id) DO UPDATE SET
                          revision_id=EXCLUDED.revision_id,updated_at=EXCLUDED.updated_at""",
                    (owner, features.source_id, revision_id, activated_at),
                )
            return stored

        return self._transaction(operation)

    def latest(self, owner_id: str, source_id: str) -> SourceTrustRevision | None:
        owner = normalize_owner_id(owner_id)
        source = _bounded_source_id(source_id)

        def operation(cursor: CursorLike) -> SourceTrustRevision | None:
            cursor.execute(
                f"""SELECT r.owner_id,r.revision_id,r.source_id,r.features_json::text,
                           r.reviewer_id,r.review_basis,r.created_at
                    FROM {self.schema}.source_trust_heads h
                    JOIN {self.schema}.source_trust_revisions r
                      ON r.owner_id=h.owner_id AND r.revision_id=h.revision_id
                    WHERE h.owner_id=%s AND h.source_id=%s LIMIT 1""",
                (owner, source),
            )
            row = cursor.fetchone()
            return self._revision_from_row(row) if row is not None else None

        return self._transaction(operation)

    def history(
        self,
        owner_id: str,
        source_id: str,
        *,
        limit: int = 100,
    ) -> tuple[SourceTrustRevision, ...]:
        owner = normalize_owner_id(owner_id)
        source = _bounded_source_id(source_id)
        if not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")

        def operation(cursor: CursorLike) -> tuple[SourceTrustRevision, ...]:
            cursor.execute(
                f"""SELECT owner_id,revision_id,source_id,features_json::text,reviewer_id,review_basis,created_at
                    FROM {self.schema}.source_trust_revisions WHERE owner_id=%s AND source_id=%s
                    ORDER BY created_at DESC,revision_id DESC LIMIT %s""",
                (owner, source, limit),
            )
            return tuple(self._revision_from_row(row) for row in cursor.fetchall())

        return self._transaction(operation)

    def list_latest(
        self,
        owner_id: str,
        *,
        limit: int = 500,
    ) -> tuple[SourceTrustRevision, ...]:
        owner = normalize_owner_id(owner_id)
        if not 1 <= limit <= 5000:
            raise ValueError("limit is invalid")

        def operation(cursor: CursorLike) -> tuple[SourceTrustRevision, ...]:
            cursor.execute(
                f"""SELECT r.owner_id,r.revision_id,r.source_id,r.features_json::text,
                           r.reviewer_id,r.review_basis,r.created_at
                    FROM {self.schema}.source_trust_heads h
                    JOIN {self.schema}.source_trust_revisions r
                      ON r.owner_id=h.owner_id AND r.revision_id=h.revision_id
                    WHERE h.owner_id=%s ORDER BY h.updated_at DESC,h.source_id LIMIT %s""",
                (owner, limit),
            )
            return tuple(self._revision_from_row(row) for row in cursor.fetchall())

        return self._transaction(operation)

    def pending_activations(
        self,
        owner_id: str,
        *,
        source_id: str | None = None,
        limit: int = 1000,
    ) -> tuple[SourceTrustActivation, ...]:
        owner = normalize_owner_id(owner_id)
        if not 1 <= limit <= 10_000:
            raise ValueError("limit is invalid")
        clauses = ["owner_id=%s", "invalidation_completed_at IS NULL"]
        params: list[Any] = [owner]
        if source_id is not None:
            clauses.append("source_id=%s")
            params.append(_bounded_source_id(source_id))
        params.append(limit)

        def operation(cursor: CursorLike) -> tuple[SourceTrustActivation, ...]:
            cursor.execute(
                f"""SELECT owner_id,activation_id,source_id,previous_revision_id,revision_id,
                           activated_at,invalidation_completed_at,last_error
                    FROM {self.schema}.source_trust_activations
                    WHERE {' AND '.join(clauses)} ORDER BY activated_at,activation_id LIMIT %s""",
                tuple(params),
            )
            return tuple(self._activation_from_row(row) for row in cursor.fetchall())

        return self._transaction(operation)

    def mark_activation_completed(self, owner_id: str, activation_id: str) -> None:
        owner = normalize_owner_id(owner_id)
        activation = str(activation_id or "").strip().lower()
        if len(activation) != 64 or any(ch not in "0123456789abcdef" for ch in activation):
            raise ValueError("activation_id must be SHA-256")

        def operation(cursor: CursorLike) -> None:
            cursor.execute(
                f"""UPDATE {self.schema}.source_trust_activations
                    SET invalidation_completed_at=%s,last_error=''
                    WHERE owner_id=%s AND activation_id=%s AND invalidation_completed_at IS NULL""",
                (time.time(), owner, activation),
            )
            if cursor.rowcount not in {0, 1}:
                raise RuntimeError("source trust activation completion was ambiguous")

        self._transaction(operation)

    def mark_activation_failed(
        self,
        owner_id: str,
        activation_id: str,
        error_type: str,
    ) -> None:
        owner = normalize_owner_id(owner_id)
        activation = str(activation_id or "").strip().lower()
        if len(activation) != 64 or any(ch not in "0123456789abcdef" for ch in activation):
            raise ValueError("activation_id must be SHA-256")
        error = str(error_type or "unknown")[:200]

        def operation(cursor: CursorLike) -> None:
            cursor.execute(
                f"""UPDATE {self.schema}.source_trust_activations SET last_error=%s
                    WHERE owner_id=%s AND activation_id=%s AND invalidation_completed_at IS NULL""",
                (error, owner, activation),
            )

        self._transaction(operation)

    def activation_history(
        self,
        owner_id: str,
        source_id: str,
        *,
        limit: int = 100,
    ) -> tuple[SourceTrustActivation, ...]:
        owner = normalize_owner_id(owner_id)
        source = _bounded_source_id(source_id)
        if not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")

        def operation(cursor: CursorLike) -> tuple[SourceTrustActivation, ...]:
            cursor.execute(
                f"""SELECT owner_id,activation_id,source_id,previous_revision_id,revision_id,
                           activated_at,invalidation_completed_at,last_error
                    FROM {self.schema}.source_trust_activations
                    WHERE owner_id=%s AND source_id=%s
                    ORDER BY activated_at DESC,activation_id DESC LIMIT %s""",
                (owner, source, limit),
            )
            return tuple(self._activation_from_row(row) for row in cursor.fetchall())

        return self._transaction(operation)


__all__ = ["PostgresSourceTrustStore"]
