"""PostgreSQL human-review and feedback stores with shared lease semantics."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

from tools.feedback_store import (
    ActiveLearningExample,
    FeedbackEvent,
    FeedbackKind,
    FeedbackStore,
    _KINDS,
    _MAX_LIMIT as _FEEDBACK_MAX_LIMIT,
    _digest,
    _finite as _feedback_finite,
    _identifier as _feedback_identifier,
    _json as _feedback_json,
)
from tools.postgres_research_stores import _PostgresMixin, _row
from tools.review_routing import ReviewDecision
from tools.review_store import (
    ReviewRecord,
    ReviewStore,
    _MAX_LIMIT as _REVIEW_MAX_LIMIT,
    _MAX_TTL,
    _STATES,
    _finite as _review_finite,
    _identifier as _review_identifier,
    _metadata,
    _query_hash,
)
from tools.security import normalize_owner_id
from tools.sql_control_plane import ConnectionFactory, CursorLike


def _json_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return dict(json.loads(str(value)))


def _json_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return tuple(str(item) for item in json.loads(str(value)))


class PostgresReviewStore(_PostgresMixin, ReviewStore):
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
                f"""CREATE TABLE IF NOT EXISTS {schema}.reviews (
                    request_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    priority DOUBLE PRECISION NOT NULL,
                    reasons_json JSONB NOT NULL,
                    query_sha256 TEXT,
                    metadata_json JSONB NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    reviewer_id TEXT,
                    lease_token BIGINT NOT NULL DEFAULT 0,
                    lease_expires_at DOUBLE PRECISION NOT NULL DEFAULT 0,
                    resolution TEXT,
                    PRIMARY KEY(owner_id,request_id)
                )""",
                f"CREATE INDEX IF NOT EXISTS idx_reviews_queue ON {schema}.reviews(owner_id,state,priority DESC,created_at ASC)",
                f"CREATE INDEX IF NOT EXISTS idx_reviews_lease ON {schema}.reviews(state,lease_expires_at)",
            )
        )

    @staticmethod
    def _record(row: Sequence[Any] | Mapping[str, Any]) -> ReviewRecord:
        reviewer = _row(row, "reviewer_id", 9)
        resolution = _row(row, "resolution", 12)
        query_sha = _row(row, "query_sha256", 5)
        return ReviewRecord(
            request_id=str(_row(row, "request_id", 0)),
            owner_id=str(_row(row, "owner_id", 1)),
            state=str(_row(row, "state", 2)),
            priority=float(_row(row, "priority", 3)),
            reasons=_json_list(_row(row, "reasons_json", 4)),
            query_sha256=None if query_sha is None else str(query_sha),
            metadata=_json_mapping(_row(row, "metadata_json", 6)),
            created_at=float(_row(row, "created_at", 7)),
            updated_at=float(_row(row, "updated_at", 8)),
            reviewer_id=None if reviewer is None else str(reviewer),
            lease_token=int(_row(row, "lease_token", 10)),
            lease_expires_at=float(_row(row, "lease_expires_at", 11)),
            resolution=None if resolution is None else str(resolution),
        )

    @staticmethod
    def _columns() -> str:
        return (
            "request_id,owner_id,state,priority,reasons_json::text,query_sha256,"
            "metadata_json::text,created_at,updated_at,reviewer_id,lease_token,"
            "lease_expires_at,resolution"
        )

    def enqueue(
        self,
        *,
        owner_id: str,
        request_id: str,
        decision: ReviewDecision,
        query: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> ReviewRecord:
        owner = normalize_owner_id(owner_id)
        request = _review_identifier(request_id, "request_id")
        if not isinstance(decision, ReviewDecision) or decision.route != "human_review":
            raise ValueError("decision must be a human_review ReviewDecision.")
        selected_now = time.time() if now is None else _review_finite(now, "now")
        reasons_json = json.dumps(list(decision.reasons), separators=(",", ":"))
        metadata_json = _metadata(metadata)
        query_sha256 = _query_hash(query)

        def operation(cursor: CursorLike) -> ReviewRecord:
            cursor.execute(
                f"""INSERT INTO {self.schema}.reviews
                    (request_id,owner_id,state,priority,reasons_json,query_sha256,metadata_json,
                     created_at,updated_at,reviewer_id,lease_token,lease_expires_at,resolution)
                    VALUES(%s,%s,'pending',%s,%s::jsonb,%s,%s::jsonb,%s,%s,NULL,0,0,NULL)
                    ON CONFLICT(owner_id,request_id) DO NOTHING""",
                (
                    request,
                    owner,
                    decision.priority,
                    reasons_json,
                    query_sha256,
                    metadata_json,
                    selected_now,
                    selected_now,
                ),
            )
            cursor.execute(
                f"SELECT {self._columns()} FROM {self.schema}.reviews WHERE owner_id=%s AND request_id=%s",
                (owner, request),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("review enqueue failed.")
            return self._record(row)

        return self._transaction(operation)

    def claim_next(
        self,
        *,
        owner_id: str,
        reviewer_id: str,
        ttl_seconds: float = 300.0,
        now: float | None = None,
    ) -> ReviewRecord | None:
        owner = normalize_owner_id(owner_id)
        reviewer = _review_identifier(reviewer_id, "reviewer_id")
        ttl = _review_finite(ttl_seconds, "ttl_seconds", 0.001, _MAX_TTL)
        selected_now = time.time() if now is None else _review_finite(now, "now")
        expires = selected_now + ttl

        def operation(cursor: CursorLike) -> ReviewRecord | None:
            cursor.execute(
                f"""UPDATE {self.schema}.reviews
                    SET state='pending',reviewer_id=NULL,lease_expires_at=0,updated_at=%s
                    WHERE owner_id=%s AND state='claimed' AND lease_expires_at<=%s""",
                (selected_now, owner, selected_now),
            )
            cursor.execute(
                f"""SELECT {self._columns()} FROM {self.schema}.reviews
                    WHERE owner_id=%s AND state='pending'
                    ORDER BY priority DESC,created_at ASC,request_id ASC
                    FOR UPDATE SKIP LOCKED LIMIT 1""",
                (owner,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            request = str(_row(row, "request_id", 0))
            token = int(_row(row, "lease_token", 10)) + 1
            cursor.execute(
                f"""UPDATE {self.schema}.reviews
                    SET state='claimed',reviewer_id=%s,lease_token=%s,lease_expires_at=%s,updated_at=%s
                    WHERE owner_id=%s AND request_id=%s AND state='pending'""",
                (reviewer, token, expires, selected_now, owner, request),
            )
            if cursor.rowcount != 1:
                return None
            cursor.execute(
                f"SELECT {self._columns()} FROM {self.schema}.reviews WHERE owner_id=%s AND request_id=%s",
                (owner, request),
            )
            claimed = cursor.fetchone()
            return None if claimed is None else self._record(claimed)

        return self._transaction(operation)

    def renew(
        self,
        record: ReviewRecord,
        *,
        ttl_seconds: float = 300.0,
        now: float | None = None,
    ) -> ReviewRecord | None:
        if not isinstance(record, ReviewRecord) or record.state != "claimed" or record.reviewer_id is None:
            raise ValueError("record must be a claimed ReviewRecord.")
        selected_now = time.time() if now is None else _review_finite(now, "now")
        ttl = _review_finite(ttl_seconds, "ttl_seconds", 0.001, _MAX_TTL)
        expires = selected_now + ttl

        def operation(cursor: CursorLike) -> ReviewRecord | None:
            cursor.execute(
                f"""UPDATE {self.schema}.reviews SET lease_expires_at=%s,updated_at=%s
                    WHERE owner_id=%s AND request_id=%s AND state='claimed' AND reviewer_id=%s
                      AND lease_token=%s AND lease_expires_at>%s""",
                (
                    expires,
                    selected_now,
                    record.owner_id,
                    record.request_id,
                    record.reviewer_id,
                    record.lease_token,
                    selected_now,
                ),
            )
            if cursor.rowcount != 1:
                return None
            cursor.execute(
                f"SELECT {self._columns()} FROM {self.schema}.reviews WHERE owner_id=%s AND request_id=%s",
                (record.owner_id, record.request_id),
            )
            row = cursor.fetchone()
            return None if row is None else self._record(row)

        return self._transaction(operation)

    def resolve(
        self,
        record: ReviewRecord,
        *,
        resolution: str,
        now: float | None = None,
    ) -> bool:
        if not isinstance(record, ReviewRecord) or record.state != "claimed" or record.reviewer_id is None:
            raise ValueError("record must be a claimed ReviewRecord.")
        selected_resolution = _review_identifier(resolution, "resolution")
        selected_now = time.time() if now is None else _review_finite(now, "now")

        def operation(cursor: CursorLike) -> bool:
            cursor.execute(
                f"""UPDATE {self.schema}.reviews
                    SET state='resolved',resolution=%s,lease_expires_at=0,updated_at=%s
                    WHERE owner_id=%s AND request_id=%s AND state='claimed' AND reviewer_id=%s
                      AND lease_token=%s AND lease_expires_at>%s""",
                (
                    selected_resolution,
                    selected_now,
                    record.owner_id,
                    record.request_id,
                    record.reviewer_id,
                    record.lease_token,
                    selected_now,
                ),
            )
            return cursor.rowcount == 1

        return self._transaction(operation)

    def cancel(
        self,
        *,
        owner_id: str,
        request_id: str,
        now: float | None = None,
    ) -> bool:
        owner = normalize_owner_id(owner_id)
        request = _review_identifier(request_id, "request_id")
        selected_now = time.time() if now is None else _review_finite(now, "now")

        def operation(cursor: CursorLike) -> bool:
            cursor.execute(
                f"""UPDATE {self.schema}.reviews
                    SET state='cancelled',lease_expires_at=0,updated_at=%s
                    WHERE owner_id=%s AND request_id=%s AND state IN ('pending','claimed')""",
                (selected_now, owner, request),
            )
            return cursor.rowcount == 1

        return self._transaction(operation)

    def get(self, *, owner_id: str, request_id: str) -> ReviewRecord | None:
        owner = normalize_owner_id(owner_id)
        request = _review_identifier(request_id, "request_id")

        def operation(cursor: CursorLike) -> ReviewRecord | None:
            cursor.execute(
                f"SELECT {self._columns()} FROM {self.schema}.reviews WHERE owner_id=%s AND request_id=%s",
                (owner, request),
            )
            row = cursor.fetchone()
            return None if row is None else self._record(row)

        return self._transaction(operation)

    def list(
        self,
        *,
        owner_id: str,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[ReviewRecord, ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _REVIEW_MAX_LIMIT:
            raise ValueError("limit is invalid.")
        params: list[Any] = [owner]
        where = "owner_id=%s"
        if state is not None:
            if state not in _STATES:
                raise ValueError("state is invalid.")
            where += " AND state=%s"
            params.append(state)
        params.append(limit)

        def operation(cursor: CursorLike) -> tuple[ReviewRecord, ...]:
            cursor.execute(
                f"SELECT {self._columns()} FROM {self.schema}.reviews WHERE {where} ORDER BY priority DESC,created_at ASC LIMIT %s",
                tuple(params),
            )
            return tuple(self._record(row) for row in cursor.fetchall())

        return self._transaction(operation)

    def delete_owner(self, *, owner_id: str) -> int:
        owner = normalize_owner_id(owner_id)

        def operation(cursor: CursorLike) -> int:
            cursor.execute(f"DELETE FROM {self.schema}.reviews WHERE owner_id=%s", (owner,))
            return int(cursor.rowcount)

        return self._transaction(operation)


class PostgresFeedbackStore(_PostgresMixin, FeedbackStore):
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
                f"""CREATE TABLE IF NOT EXISTS {schema}.feedback (
                    owner_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    query_sha256 TEXT,
                    evidence_sha256 TEXT,
                    weight DOUBLE PRECISION NOT NULL,
                    metadata_json JSONB NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY(owner_id,event_id)
                )""",
                f"CREATE INDEX IF NOT EXISTS idx_feedback_kind ON {schema}.feedback(owner_id,kind,created_at DESC)",
            )
        )

    @staticmethod
    def _record(row: Sequence[Any] | Mapping[str, Any]) -> FeedbackEvent:
        query = _row(row, "query_sha256", 4)
        evidence = _row(row, "evidence_sha256", 5)
        return FeedbackEvent(
            event_id=str(_row(row, "event_id", 0)),
            owner_id=str(_row(row, "owner_id", 1)),
            kind=str(_row(row, "kind", 2)),
            subject_id=str(_row(row, "subject_id", 3)),
            query_sha256=None if query is None else str(query),
            evidence_sha256=None if evidence is None else str(evidence),
            weight=float(_row(row, "weight", 6)),
            metadata=_json_mapping(_row(row, "metadata_json", 7)),
            created_at=float(_row(row, "created_at", 8)),
        )

    @staticmethod
    def _columns() -> str:
        return (
            "event_id,owner_id,kind,subject_id,query_sha256,evidence_sha256,weight,"
            "metadata_json::text,created_at"
        )

    def put(
        self,
        *,
        owner_id: str,
        event_id: str,
        kind: FeedbackKind,
        subject_id: str,
        query: str | None = None,
        evidence: str | None = None,
        weight: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
        created_at: float | None = None,
    ) -> FeedbackEvent:
        owner = normalize_owner_id(owner_id)
        event = _feedback_identifier(event_id, "event_id")
        if kind not in _KINDS:
            raise ValueError("kind is unsupported.")
        subject = _feedback_identifier(subject_id, "subject_id")
        selected_weight = _feedback_finite(weight, "weight", 0.000001, 1_000.0)
        timestamp = (
            time.time()
            if created_at is None
            else _feedback_finite(created_at, "created_at", 0.0, 1e20)
        )
        query_hash = _digest(query, "query")
        evidence_hash = _digest(evidence, "evidence")
        metadata_json = _feedback_json(metadata)

        def operation(cursor: CursorLike) -> FeedbackEvent:
            cursor.execute(
                f"""INSERT INTO {self.schema}.feedback
                    (owner_id,event_id,kind,subject_id,query_sha256,evidence_sha256,weight,metadata_json,created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
                    ON CONFLICT(owner_id,event_id) DO NOTHING""",
                (
                    owner,
                    event,
                    kind,
                    subject,
                    query_hash,
                    evidence_hash,
                    selected_weight,
                    metadata_json,
                    timestamp,
                ),
            )
            cursor.execute(
                f"SELECT {self._columns()} FROM {self.schema}.feedback WHERE owner_id=%s AND event_id=%s",
                (owner, event),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("feedback write failed.")
            return self._record(row)

        return self._transaction(operation)

    def list(
        self,
        *,
        owner_id: str,
        kind: FeedbackKind | None = None,
        limit: int = 100,
    ) -> tuple[FeedbackEvent, ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _FEEDBACK_MAX_LIMIT:
            raise ValueError("limit is invalid.")
        if kind is not None and kind not in _KINDS:
            raise ValueError("kind is unsupported.")

        def operation(cursor: CursorLike) -> tuple[FeedbackEvent, ...]:
            if kind is None:
                cursor.execute(
                    f"SELECT {self._columns()} FROM {self.schema}.feedback WHERE owner_id=%s ORDER BY created_at DESC,event_id DESC LIMIT %s",
                    (owner, limit),
                )
            else:
                cursor.execute(
                    f"SELECT {self._columns()} FROM {self.schema}.feedback WHERE owner_id=%s AND kind=%s ORDER BY created_at DESC,event_id DESC LIMIT %s",
                    (owner, kind, limit),
                )
            return tuple(self._record(row) for row in cursor.fetchall())

        return self._transaction(operation)

    def export_active_learning(
        self,
        *,
        owner_id: str,
        limit: int = 1_000,
    ) -> tuple[ActiveLearningExample, ...]:
        return tuple(
            ActiveLearningExample(
                kind=row.kind,
                subject_id=row.subject_id,
                weight=row.weight,
                metadata=row.metadata,
                query_sha256=row.query_sha256,
                evidence_sha256=row.evidence_sha256,
            )
            for row in self.list(owner_id=owner_id, limit=limit)
        )

    def delete_owner(self, *, owner_id: str) -> int:
        owner = normalize_owner_id(owner_id)

        def operation(cursor: CursorLike) -> int:
            cursor.execute(f"DELETE FROM {self.schema}.feedback WHERE owner_id=%s", (owner,))
            return int(cursor.rowcount)

        return self._transaction(operation)


__all__ = ["PostgresFeedbackStore", "PostgresReviewStore"]
