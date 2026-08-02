"""Durable one-decision reservation for signed relation-review assertions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_relation_review import (
    CrossDocumentRelationProposal,
    RelationReviewDecision,
)
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_STATES = frozenset({"reserved", "committed"})


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    selected = value.strip()
    if (
        not selected
        or len(selected) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in selected)
    ):
        raise ValueError(f"{label} is invalid.")
    return selected


def _digest(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
    if len(selected) != 64 or any(character not in "0123456789abcdef" for character in selected):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return selected


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(selected) or selected < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return selected


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("actor-use database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("actor-use database path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if _redirecting(info):
            raise ValueError("actor-use database path may not contain redirects.")
    return absolute


@dataclass(frozen=True)
class SignedActorUseRecord:
    assertion_digest: str
    decision_id: str
    proposal_id: str
    owner_id: str
    graph_set_key: str
    decision: str
    actor_id: str
    issuer: str
    binding_digest: str
    assertion_expires_at: float
    use_digest: str
    state: str
    reserved_at: float
    committed_at: float | None
    updated_at: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "assertion_digest",
            "decision_id",
            "proposal_id",
            "binding_digest",
            "use_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(
            self,
            "graph_set_key",
            _identifier(self.graph_set_key, "graph_set_key", 500),
        )
        object.__setattr__(self, "decision", _identifier(self.decision, "decision", 20))
        object.__setattr__(self, "actor_id", _identifier(self.actor_id, "actor_id", 200))
        object.__setattr__(self, "issuer", _identifier(self.issuer, "issuer", 200))
        object.__setattr__(
            self,
            "assertion_expires_at",
            _timestamp(self.assertion_expires_at, "assertion_expires_at"),
        )
        state = _identifier(self.state, "state", 20)
        if state not in _STATES:
            raise ValueError("actor-use state is unsupported.")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "reserved_at", _timestamp(self.reserved_at, "reserved_at"))
        if self.committed_at is not None:
            object.__setattr__(
                self,
                "committed_at",
                _timestamp(self.committed_at, "committed_at"),
            )
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at, "updated_at"))
        if self.updated_at < self.reserved_at:
            raise ValueError("actor-use updated_at may not precede reserved_at.")
        if self.state == "reserved" and self.committed_at is not None:
            raise ValueError("reserved actor use may not contain committed_at.")
        if self.state == "committed" and self.committed_at is None:
            raise ValueError("committed actor use requires committed_at.")
        if self.committed_at is not None and (
            self.committed_at < self.reserved_at
            or self.updated_at < self.committed_at
        ):
            raise ValueError("actor-use timestamps are not monotonic.")
        expected = _sha256(
            {
                "scope": "rigorousrag-signed-review-actor-use-v1",
                "assertion_digest": self.assertion_digest,
                "decision_id": self.decision_id,
                "proposal_id": self.proposal_id,
                "owner_id": self.owner_id,
                "graph_set_key": self.graph_set_key,
                "decision": self.decision,
                "actor_id": self.actor_id,
                "issuer": self.issuer,
                "binding_digest": self.binding_digest,
                "assertion_expires_at": self.assertion_expires_at,
            }
        )
        if self.use_digest != expected:
            raise ValueError("use_digest differs from signed actor-use identity.")
        if self.schema_version != 1:
            raise ValueError("signed actor-use schema is unsupported.")

    @classmethod
    def create(
        cls,
        *,
        binding: ReviewActorBinding,
        proposal: CrossDocumentRelationProposal,
        decision: RelationReviewDecision,
        reserved_at: float,
    ) -> "SignedActorUseRecord":
        if not isinstance(binding, ReviewActorBinding):
            raise ValueError("binding must be ReviewActorBinding.")
        if binding.binding_method != "hmac_assertion":
            raise ValueError("only signed actor assertions require use reservation.")
        if binding.assertion_digest is None or binding.issuer is None or binding.expires_at is None:
            raise ValueError("signed actor binding is incomplete.")
        if not isinstance(proposal, CrossDocumentRelationProposal):
            raise ValueError("proposal must be CrossDocumentRelationProposal.")
        if not isinstance(decision, RelationReviewDecision):
            raise ValueError("decision must be RelationReviewDecision.")
        timestamp = _timestamp(reserved_at, "reserved_at")
        if binding.expires_at < timestamp:
            raise PermissionError("signed actor assertion expired before reservation.")
        if (
            decision.proposal_id != proposal.proposal_id
            or decision.owner_id != proposal.owner_id
            or decision.reviewer_id != binding.actor_id
        ):
            raise PermissionError("signed actor use differs from decision scope.")
        stable = {
            "scope": "rigorousrag-signed-review-actor-use-v1",
            "assertion_digest": binding.assertion_digest,
            "decision_id": decision.decision_id,
            "proposal_id": proposal.proposal_id,
            "owner_id": proposal.owner_id,
            "graph_set_key": proposal.graph_set_key,
            "decision": decision.decision,
            "actor_id": binding.actor_id,
            "issuer": binding.issuer,
            "binding_digest": binding.binding_digest,
            "assertion_expires_at": binding.expires_at,
        }
        return cls(
            assertion_digest=binding.assertion_digest,
            decision_id=decision.decision_id,
            proposal_id=proposal.proposal_id,
            owner_id=proposal.owner_id,
            graph_set_key=proposal.graph_set_key,
            decision=decision.decision,
            actor_id=binding.actor_id,
            issuer=binding.issuer,
            binding_digest=binding.binding_digest,
            assertion_expires_at=binding.expires_at,
            use_digest=_sha256(stable),
            state="reserved",
            reserved_at=timestamp,
            committed_at=None,
            updated_at=timestamp,
        )


class SignedActorUseStore:
    """Append-only assertion reservation with monotonic commit state."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("actor-use database parent must be a regular directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("actor-use database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("actor-use database parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("actor-use database identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(self.path, timeout=30.0, isolation_level=None) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS signed_review_actor_uses (
                    assertion_digest TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    proposal_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    graph_set_key TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    issuer TEXT NOT NULL,
                    binding_digest TEXT NOT NULL,
                    assertion_expires_at REAL NOT NULL,
                    use_digest TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    reserved_at REAL NOT NULL,
                    committed_at REAL,
                    updated_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS signed_review_actor_use_decision
                    ON signed_review_actor_uses(
                        owner_id, decision_id, state, updated_at, assertion_digest
                    );
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> SignedActorUseRecord:
        if int(row["schema_version"]) != 1:
            raise RuntimeError("stored signed actor-use schema is unsupported.")
        try:
            value = SignedActorUseRecord(**json.loads(row["payload_json"]))
        except Exception as exc:
            raise RuntimeError("stored signed actor use is corrupt.") from exc
        for name in (
            "assertion_digest",
            "decision_id",
            "proposal_id",
            "owner_id",
            "graph_set_key",
            "decision",
            "actor_id",
            "issuer",
            "binding_digest",
            "assertion_expires_at",
            "use_digest",
            "state",
            "reserved_at",
            "committed_at",
            "updated_at",
        ):
            if getattr(value, name) != row[name]:
                raise RuntimeError("stored signed actor-use row identity is corrupt.")
        return value

    def reserve(self, value: SignedActorUseRecord) -> SignedActorUseRecord:
        if not isinstance(value, SignedActorUseRecord):
            raise ValueError("value must be SignedActorUseRecord.")
        payload = json.dumps(
            asdict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM signed_review_actor_uses WHERE assertion_digest=?",
                    (value.assertion_digest,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO signed_review_actor_uses VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1
                        )
                        """,
                        (
                            value.assertion_digest,
                            value.decision_id,
                            value.proposal_id,
                            value.owner_id,
                            value.graph_set_key,
                            value.decision,
                            value.actor_id,
                            value.issuer,
                            value.binding_digest,
                            value.assertion_expires_at,
                            value.use_digest,
                            payload,
                            value.state,
                            value.reserved_at,
                            value.committed_at,
                            value.updated_at,
                        ),
                    )
                else:
                    existing = self._record(row)
                    if existing.use_digest != value.use_digest:
                        raise RuntimeError(
                            "signed actor assertion is already reserved for another decision."
                        )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        result = self.get(value.assertion_digest)
        if result is None:
            raise RuntimeError("signed actor-use reservation disappeared.")
        return result

    def mark_committed(
        self,
        assertion_digest: str,
        *,
        decision_id: str,
        now: float | None = None,
    ) -> SignedActorUseRecord:
        selected = _digest(assertion_digest, "assertion_digest")
        selected_decision = _digest(decision_id, "decision_id")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM signed_review_actor_uses WHERE assertion_digest=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._record(row)
                if current.decision_id != selected_decision:
                    raise RuntimeError("signed actor-use decision identity changed.")
                timestamp = max(timestamp, current.reserved_at, current.updated_at)
                if current.state == "reserved":
                    committed = SignedActorUseRecord(
                        **{
                            **asdict(current),
                            "state": "committed",
                            "committed_at": timestamp,
                            "updated_at": timestamp,
                        }
                    )
                    payload = json.dumps(
                        asdict(committed),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    connection.execute(
                        """
                        UPDATE signed_review_actor_uses
                        SET payload_json=?, state='committed', committed_at=?, updated_at=?
                        WHERE assertion_digest=? AND state='reserved'
                        """,
                        (payload, timestamp, timestamp, selected),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        result = self.get(selected)
        if result is None or result.state != "committed":
            raise RuntimeError("signed actor-use commit was not durable.")
        return result

    def get(self, assertion_digest: str) -> SignedActorUseRecord | None:
        selected = _digest(assertion_digest, "assertion_digest")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM signed_review_actor_uses WHERE assertion_digest=?",
                (selected,),
            ).fetchone()
        return None if row is None else self._record(row)

    def list(
        self,
        *,
        owner_id: str,
        decision_id: str | None = None,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[SignedActorUseRecord, ...]:
        owner = normalize_owner_id(owner_id)
        selected_decision = (
            None if decision_id is None else _digest(decision_id, "decision_id")
        )
        selected_state = None if state is None else _identifier(state, "state", 20)
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("actor-use state filter is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM signed_review_actor_uses
                WHERE owner_id=?
                  AND (? IS NULL OR decision_id=?)
                  AND (? IS NULL OR state=?)
                ORDER BY updated_at, assertion_digest LIMIT ?
                """,
                (
                    owner,
                    selected_decision,
                    selected_decision,
                    selected_state,
                    selected_state,
                    count,
                ),
            ).fetchall()
        return tuple(self._record(row) for row in rows)


__all__ = ["SignedActorUseRecord", "SignedActorUseStore"]
