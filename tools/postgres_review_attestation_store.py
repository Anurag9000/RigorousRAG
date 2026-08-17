"""PostgreSQL persistence for immutable signed human-review attestations."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from tools.postgres_research_stores import _PostgresMixin, _row
from tools.review_attestation import ReviewDecisionAttestation
from tools.review_attestation_store import (
    ReviewAttestationStore,
    StoredReviewAttestation,
    _finite,
    _manifest,
    _review_attestation_from_mapping,
    _sha,
    _text,
)
from tools.security import normalize_owner_id
from tools.sql_control_plane import ConnectionFactory, CursorLike

_MAX_LIMIT = 10_000


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    parsed = json.loads(str(value))
    if not isinstance(parsed, Mapping):
        raise RuntimeError("stored review attestation JSON is invalid")
    return dict(parsed)


class PostgresReviewAttestationStore(_PostgresMixin, ReviewAttestationStore):
    """Shared immutable review-attestation store using an injected DB-API factory."""

    def __init__(
        self,
        connection_factory: ConnectionFactory,
        *,
        schema: str = "rigorousrag",
        initialize: bool = True,
    ) -> None:
        _PostgresMixin.__init__(self, connection_factory, schema=schema, initialize=initialize)

    def _initialize_postgres(self) -> None:
        schema = self.schema
        self._initialize_tables(
            (
                f"""CREATE TABLE IF NOT EXISTS {schema}.review_attestations (
                    owner_id TEXT NOT NULL,
                    attestation_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    lease_token BIGINT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    resolution TEXT NOT NULL,
                    captured_manifest_json JSONB NOT NULL,
                    signed_json JSONB NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY(owner_id,attestation_id)
                )""",
                f"CREATE INDEX IF NOT EXISTS idx_review_attestation_request ON {schema}.review_attestations(owner_id,request_id,created_at DESC,attestation_id DESC)",
            )
        )

    @staticmethod
    def _columns() -> str:
        return (
            "owner_id,attestation_id,request_id,lease_token,reviewer_id,resolution,"
            "captured_manifest_json::text,signed_json::text,created_at"
        )

    @staticmethod
    def _record(row: Sequence[Any] | Mapping[str, Any]) -> StoredReviewAttestation:
        captured = _mapping(_row(row, "captured_manifest_json", 6))
        signed_raw = _mapping(_row(row, "signed_json", 7))
        return StoredReviewAttestation(
            owner_id=str(_row(row, "owner_id", 0)),
            attestation_id=str(_row(row, "attestation_id", 1)),
            request_id=str(_row(row, "request_id", 2)),
            lease_token=int(_row(row, "lease_token", 3)),
            reviewer_id=str(_row(row, "reviewer_id", 4)),
            resolution=str(_row(row, "resolution", 5)),
            captured_manifest=captured,
            signed=_review_attestation_from_mapping(signed_raw),
            created_at=float(_row(row, "created_at", 8)),
        )

    def put(
        self,
        *,
        owner_id: str,
        captured_manifest: Mapping[str, Any],
        signed: ReviewDecisionAttestation,
        now: float | None = None,
    ) -> StoredReviewAttestation:
        owner = normalize_owner_id(owner_id)
        if not isinstance(signed, ReviewDecisionAttestation) or signed.owner_id != owner:
            raise ValueError("signed review attestation owner does not match store owner")
        manifest = _manifest(captured_manifest)
        request_id = _text(str(manifest.get("request_id", "")), "request_id", 500)
        reviewer_id = _text(str(manifest.get("reviewer_id", "")), "reviewer_id", 500)
        resolution = _text(str(manifest.get("resolution", "")), "resolution", 500)
        try:
            lease_token = int(manifest.get("lease_token", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("captured manifest lease token is invalid") from exc
        selected_now = time.time() if now is None else _finite(now, "now")
        item = StoredReviewAttestation(
            attestation_id=signed.fingerprint,
            owner_id=owner,
            request_id=request_id,
            lease_token=lease_token,
            reviewer_id=reviewer_id,
            resolution=resolution,
            captured_manifest=manifest,
            signed=signed,
            created_at=selected_now,
        )
        manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        signed_json = json.dumps(asdict(signed), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

        def operation(cursor: CursorLike) -> StoredReviewAttestation:
            cursor.execute(
                f"""INSERT INTO {self.schema}.review_attestations
                    (owner_id,attestation_id,request_id,lease_token,reviewer_id,resolution,
                     captured_manifest_json,signed_json,created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)
                    ON CONFLICT(owner_id,attestation_id) DO NOTHING""",
                (
                    item.owner_id,
                    item.attestation_id,
                    item.request_id,
                    item.lease_token,
                    item.reviewer_id,
                    item.resolution,
                    manifest_json,
                    signed_json,
                    item.created_at,
                ),
            )
            cursor.execute(
                f"SELECT {self._columns()} FROM {self.schema}.review_attestations WHERE owner_id=%s AND attestation_id=%s",
                (owner, item.attestation_id),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("review attestation persistence failed")
            stored = self._record(row)
            if stored != item:
                raise RuntimeError("review attestation identity collision")
            return stored

        return self._transaction(operation)

    def get(self, *, owner_id: str, attestation_id: str) -> StoredReviewAttestation | None:
        owner = normalize_owner_id(owner_id)
        identity = _sha(attestation_id, "attestation_id")

        def operation(cursor: CursorLike) -> StoredReviewAttestation | None:
            cursor.execute(
                f"SELECT {self._columns()} FROM {self.schema}.review_attestations WHERE owner_id=%s AND attestation_id=%s",
                (owner, identity),
            )
            row = cursor.fetchone()
            return None if row is None else self._record(row)

        return self._transaction(operation)

    def list(
        self,
        *,
        owner_id: str,
        request_id: str | None = None,
        limit: int = 100,
    ) -> tuple[StoredReviewAttestation, ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIMIT:
            raise ValueError("limit is invalid")
        request = None if request_id is None else _text(request_id, "request_id", 500)

        def operation(cursor: CursorLike) -> tuple[StoredReviewAttestation, ...]:
            if request is None:
                cursor.execute(
                    f"SELECT {self._columns()} FROM {self.schema}.review_attestations WHERE owner_id=%s ORDER BY created_at DESC,attestation_id DESC LIMIT %s",
                    (owner, limit),
                )
            else:
                cursor.execute(
                    f"SELECT {self._columns()} FROM {self.schema}.review_attestations WHERE owner_id=%s AND request_id=%s ORDER BY created_at DESC,attestation_id DESC LIMIT %s",
                    (owner, request, limit),
                )
            return tuple(self._record(row) for row in cursor.fetchall())

        return self._transaction(operation)


__all__ = ["PostgresReviewAttestationStore"]
