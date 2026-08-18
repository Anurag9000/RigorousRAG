"""Atomic fenced publication ledger for authoritative grounded generation.

Generation can take long enough for runtime-stack authority to rotate while a request is in
flight.  ``orchestration.authoritative_generation`` re-checks the fence after model
inference, but a separate persistence/API layer could otherwise race after that check.

This ledger intentionally shares the *same SQLite database* as
``SQLiteRuntimeStackAuthorityStore``.  Publication executes ``BEGIN IMMEDIATE``, reads the
current runtime authority under that write serialization point, and inserts an immutable,
digest-only publication receipt before releasing the transaction.  Runtime promotion,
rollback and publication therefore have one database serialization order.

No answer text, prompt text, evidence text or raw model output is persisted here.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from orchestration.authoritative_generation import AuthoritativeGenerationResult
from orchestration.runtime_stack_authority import SQLiteRuntimeStackAuthorityStore

_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _time(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return selected


def _scope_sha256(owner_id: str, service_id: str, domain_id: str) -> str:
    return _digest(
        {
            "schema": "rigorousrag-serving-scope/v1",
            "owner_id": _text(owner_id, "owner_id"),
            "service_id": _text(service_id, "service_id"),
            "domain_id": _text(domain_id, "domain_id"),
        }
    )


@dataclass(frozen=True)
class GenerationPublicationReceipt:
    request_sha256: str
    scope_sha256: str
    serving_receipt_sha256: str
    grounded_output_sha256: str
    stack_sha256: str
    fencing_token: int
    authority_revision: int
    published_at: float
    publication_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "request_sha256",
            "scope_sha256",
            "serving_receipt_sha256",
            "grounded_output_sha256",
            "stack_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "fencing_token", _positive_int(self.fencing_token, "fencing_token"))
        object.__setattr__(self, "authority_revision", _positive_int(self.authority_revision, "authority_revision"))
        object.__setattr__(self, "published_at", _time(self.published_at, "published_at"))
        expected = _digest(self._payload())
        provided = _sha(self.publication_sha256, "publication_sha256")
        if provided != expected:
            raise ValueError("publication_sha256 does not match publication receipt")
        object.__setattr__(self, "publication_sha256", provided)

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-generation-publication/v1",
            "request_sha256": self.request_sha256,
            "scope_sha256": self.scope_sha256,
            "serving_receipt_sha256": self.serving_receipt_sha256,
            "grounded_output_sha256": self.grounded_output_sha256,
            "stack_sha256": self.stack_sha256,
            "fencing_token": self.fencing_token,
            "authority_revision": self.authority_revision,
            "published_at": self.published_at,
        }

    @classmethod
    def build(
        cls,
        *,
        request_sha256: str,
        scope_sha256: str,
        serving_receipt_sha256: str,
        grounded_output_sha256: str,
        stack_sha256: str,
        fencing_token: int,
        authority_revision: int,
        published_at: float,
    ) -> "GenerationPublicationReceipt":
        payload = {
            "schema": "rigorousrag-generation-publication/v1",
            "request_sha256": _sha(request_sha256, "request_sha256"),
            "scope_sha256": _sha(scope_sha256, "scope_sha256"),
            "serving_receipt_sha256": _sha(serving_receipt_sha256, "serving_receipt_sha256"),
            "grounded_output_sha256": _sha(grounded_output_sha256, "grounded_output_sha256"),
            "stack_sha256": _sha(stack_sha256, "stack_sha256"),
            "fencing_token": _positive_int(fencing_token, "fencing_token"),
            "authority_revision": _positive_int(authority_revision, "authority_revision"),
            "published_at": _time(published_at, "published_at"),
        }
        return cls(**{key: value for key, value in payload.items() if key != "schema"}, publication_sha256=_digest(payload))


class SQLiteGenerationPublicationLedger:
    """Digest-only publication authority sharing the runtime-authority database."""

    def __init__(self, runtime_authority: SQLiteRuntimeStackAuthorityStore) -> None:
        if not isinstance(runtime_authority, SQLiteRuntimeStackAuthorityStore):
            raise ValueError("runtime_authority must be SQLiteRuntimeStackAuthorityStore")
        self.runtime_authority = runtime_authority
        self.path = runtime_authority.path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS generation_publication_ledger (
                    owner_id TEXT NOT NULL,
                    service_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    scope_sha256 TEXT NOT NULL,
                    serving_receipt_sha256 TEXT NOT NULL,
                    grounded_output_sha256 TEXT NOT NULL,
                    stack_sha256 TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    authority_revision INTEGER NOT NULL,
                    published_at REAL NOT NULL,
                    publication_sha256 TEXT NOT NULL,
                    PRIMARY KEY(owner_id,service_id,domain_id,request_sha256),
                    UNIQUE(publication_sha256)
                )"""
            )

    @staticmethod
    def _decode(row: sqlite3.Row) -> GenerationPublicationReceipt:
        return GenerationPublicationReceipt(
            request_sha256=row["request_sha256"],
            scope_sha256=row["scope_sha256"],
            serving_receipt_sha256=row["serving_receipt_sha256"],
            grounded_output_sha256=row["grounded_output_sha256"],
            stack_sha256=row["stack_sha256"],
            fencing_token=int(row["fencing_token"]),
            authority_revision=int(row["authority_revision"]),
            published_at=float(row["published_at"]),
            publication_sha256=row["publication_sha256"],
        )

    def get(
        self,
        *,
        owner_id: str,
        service_id: str,
        domain_id: str,
        request_sha256: str,
    ) -> GenerationPublicationReceipt | None:
        owner, service, domain = (
            _text(owner_id, "owner_id"),
            _text(service_id, "service_id"),
            _text(domain_id, "domain_id"),
        )
        request = _sha(request_sha256, "request_sha256")
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM generation_publication_ledger
                   WHERE owner_id=? AND service_id=? AND domain_id=? AND request_sha256=?""",
                (owner, service, domain, request),
            ).fetchone()
        return None if row is None else self._decode(row)

    def publish(
        self,
        result: AuthoritativeGenerationResult,
        *,
        owner_id: str,
        service_id: str,
        domain_id: str,
        request_sha256: str,
        now: float,
    ) -> GenerationPublicationReceipt:
        """Atomically fence and record a publishable grounded result.

        Idempotent replay of the exact request/result returns the original receipt.  A
        conflicting result for the same request identity is rejected.
        """

        if not isinstance(result, AuthoritativeGenerationResult):
            raise ValueError("result must be AuthoritativeGenerationResult")
        if result.receipt.action != "published" or result.grounded_output is None:
            raise ValueError("only a published authoritative generation result may be published")
        owner, service, domain = (
            _text(owner_id, "owner_id"),
            _text(service_id, "service_id"),
            _text(domain_id, "domain_id"),
        )
        request = _sha(request_sha256, "request_sha256")
        timestamp = _time(now, "now")
        scope = _scope_sha256(owner, service, domain)
        if result.receipt.scope_sha256 != scope:
            raise ValueError("serving result belongs to a different publication scope")
        if result.receipt.grounded_output_sha256 != result.grounded_output.grounded_output_sha256:
            raise RuntimeError("serving result grounded-output identity is inconsistent")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_row = connection.execute(
                """SELECT * FROM generation_publication_ledger
                   WHERE owner_id=? AND service_id=? AND domain_id=? AND request_sha256=?""",
                (owner, service, domain, request),
            ).fetchone()
            if existing_row is not None:
                existing = self._decode(existing_row)
                if (
                    existing.scope_sha256 != scope
                    or existing.serving_receipt_sha256 != result.receipt.receipt_sha256
                    or existing.grounded_output_sha256 != result.grounded_output.grounded_output_sha256
                    or existing.stack_sha256 != result.receipt.stack_sha256
                    or existing.fencing_token != result.receipt.fencing_token
                ):
                    raise RuntimeError("request identity already has a different publication result")
                return existing

            authority = connection.execute(
                """SELECT stack_sha256,authority_revision,fencing_token FROM runtime_stack_authority
                   WHERE owner_id=? AND service_id=? AND domain_id=?""",
                (owner, service, domain),
            ).fetchone()
            if authority is None:
                raise RuntimeError("cannot publish before runtime-stack authority is established")
            current_stack = _sha(authority["stack_sha256"], "persisted stack_sha256")
            current_revision = _positive_int(int(authority["authority_revision"]), "persisted authority_revision")
            current_fence = _positive_int(int(authority["fencing_token"]), "persisted fencing_token")
            if current_stack != result.receipt.stack_sha256 or current_fence != result.receipt.fencing_token:
                raise RuntimeError("generation result is stale under current runtime-stack authority")

            receipt = GenerationPublicationReceipt.build(
                request_sha256=request,
                scope_sha256=scope,
                serving_receipt_sha256=result.receipt.receipt_sha256,
                grounded_output_sha256=result.grounded_output.grounded_output_sha256,
                stack_sha256=current_stack,
                fencing_token=current_fence,
                authority_revision=current_revision,
                published_at=timestamp,
            )
            connection.execute(
                """INSERT INTO generation_publication_ledger(
                    owner_id,service_id,domain_id,request_sha256,scope_sha256,
                    serving_receipt_sha256,grounded_output_sha256,stack_sha256,
                    fencing_token,authority_revision,published_at,publication_sha256
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    owner,
                    service,
                    domain,
                    request,
                    scope,
                    receipt.serving_receipt_sha256,
                    receipt.grounded_output_sha256,
                    receipt.stack_sha256,
                    receipt.fencing_token,
                    receipt.authority_revision,
                    receipt.published_at,
                    receipt.publication_sha256,
                ),
            )
            return receipt


__all__ = ["GenerationPublicationReceipt", "SQLiteGenerationPublicationLedger"]
