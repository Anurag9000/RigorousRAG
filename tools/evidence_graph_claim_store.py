"""Immutable scientific-claim proposals and atomically governed terminal reviews."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.evidence_graph_claim_contracts import (
    REVIEW_DECISIONS,
    ClaimEvidenceLocator,
    ClaimReviewAuthorization,
    ClaimReviewDecision,
    ScientificClaimProposal,
    _digest,
    _identifier,
    _integer,
)
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_MAX_BATCH = 10_000


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("claim review database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("claim review database path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("claim review database path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("claim review database path may not contain redirects.")
    return absolute


def _strict_json(value: str, label: str) -> dict[str, Any]:
    if not isinstance(value, str) or len(value) > 20_000_000:
        raise RuntimeError(f"stored {label} is corrupt.")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise RuntimeError(f"stored {label} is corrupt.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"stored {label} is corrupt.")
    return parsed


def _proposal_from_payload(value: str) -> ScientificClaimProposal:
    raw = _strict_json(value, "claim proposal")
    try:
        raw["locator"] = ClaimEvidenceLocator(**raw["locator"])
        return ScientificClaimProposal(**raw)
    except Exception as exc:
        raise RuntimeError("stored claim proposal is corrupt.") from exc


def _decision_from_payload(value: str) -> ClaimReviewDecision:
    try:
        return ClaimReviewDecision(**_strict_json(value, "claim review decision"))
    except Exception as exc:
        raise RuntimeError("stored claim review decision is corrupt.") from exc


def _authorization_from_payload(value: str) -> ClaimReviewAuthorization:
    try:
        return ClaimReviewAuthorization(**_strict_json(value, "claim review authorization"))
    except Exception as exc:
        raise RuntimeError("stored claim review authorization is corrupt.") from exc


def _semantic_decision(value: ClaimReviewDecision) -> tuple[Any, ...]:
    return (
        value.decision_id,
        value.proposal_id,
        value.owner_id,
        value.decision,
        value.reviewer_id,
        value.reason_code,
        value.replacement_proposal_id,
        value.schema_version,
    )


class ScientificClaimReviewStore:
    """Append-only claim proposal/review store with correction-lineage enforcement."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("claim review database parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("claim review database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("claim review database parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("claim review database identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(self.path, timeout=30.0, isolation_level=None) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scientific_claim_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    profile_fingerprint TEXT NOT NULL,
                    supersedes_proposal_id TEXT,
                    proposal_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL,
                    FOREIGN KEY(supersedes_proposal_id)
                        REFERENCES scientific_claim_proposals(proposal_id)
                        ON DELETE RESTRICT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS scientific_claim_single_successor
                    ON scientific_claim_proposals(supersedes_proposal_id)
                    WHERE supersedes_proposal_id IS NOT NULL;
                CREATE INDEX IF NOT EXISTS scientific_claim_scope
                    ON scientific_claim_proposals(
                        owner_id, doc_id, generation, created_at, proposal_id
                    );
                CREATE TABLE IF NOT EXISTS scientific_claim_decisions (
                    proposal_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    decided_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL,
                    FOREIGN KEY(proposal_id)
                        REFERENCES scientific_claim_proposals(proposal_id)
                        ON DELETE RESTRICT
                );
                CREATE TABLE IF NOT EXISTS scientific_claim_authorizations (
                    proposal_id TEXT PRIMARY KEY,
                    authorization_digest TEXT NOT NULL UNIQUE,
                    decision_id TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    authorized_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL,
                    FOREIGN KEY(proposal_id)
                        REFERENCES scientific_claim_proposals(proposal_id)
                        ON DELETE RESTRICT
                );
                """
            )

    @staticmethod
    def _proposal(row: sqlite3.Row) -> ScientificClaimProposal:
        value = _proposal_from_payload(row["payload_json"])
        if value.proposal_digest != row["proposal_digest"]:
            raise RuntimeError("stored claim proposal digest is corrupt.")
        if (
            value.proposal_id != row["proposal_id"]
            or value.owner_id != row["owner_id"]
            or value.doc_id != row["doc_id"]
            or value.generation != int(row["generation"])
            or value.content_sha256 != row["content_sha256"]
            or value.profile_fingerprint != row["profile_fingerprint"]
            or value.supersedes_proposal_id != row["supersedes_proposal_id"]
        ):
            raise RuntimeError("stored claim proposal columns are corrupt.")
        return value

    @staticmethod
    def _decision(row: sqlite3.Row) -> ClaimReviewDecision:
        value = _decision_from_payload(row["payload_json"])
        if (
            value.proposal_id != row["proposal_id"]
            or value.decision_id != row["decision_id"]
            or value.owner_id != row["owner_id"]
            or value.decision != row["decision"]
        ):
            raise RuntimeError("stored claim review decision columns are corrupt.")
        return value

    @staticmethod
    def _authorization(row: sqlite3.Row) -> ClaimReviewAuthorization:
        value = _authorization_from_payload(row["payload_json"])
        if (
            value.proposal_id != row["proposal_id"]
            or value.authorization_digest != row["authorization_digest"]
            or value.decision_id != row["decision_id"]
            or value.owner_id != row["owner_id"]
            or value.reviewer_id != row["reviewer_id"]
        ):
            raise RuntimeError("stored claim review authorization columns are corrupt.")
        return value

    def submit(self, proposal: ScientificClaimProposal) -> ScientificClaimProposal:
        return self.submit_many((proposal,))[0]

    def submit_many(
        self,
        proposals: Iterable[ScientificClaimProposal],
    ) -> tuple[ScientificClaimProposal, ...]:
        if isinstance(proposals, (str, bytes, bytearray)):
            raise ValueError("proposals must be an iterable.")
        values = tuple(proposals)
        if not 1 <= len(values) <= _MAX_BATCH:
            raise ValueError("proposals must contain a bounded non-empty batch.")
        if any(not isinstance(value, ScientificClaimProposal) for value in values):
            raise ValueError("every proposal must be ScientificClaimProposal.")
        if len({value.proposal_id for value in values}) != len(values):
            raise ValueError("claim proposal batch contains duplicate IDs.")
        by_id = {value.proposal_id: value for value in values}
        payloads = {
            value.proposal_id: json.dumps(
                asdict(value),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for value in values
        }
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for proposal in values:
                    predecessor = proposal.supersedes_proposal_id
                    if predecessor is not None:
                        predecessor_row = connection.execute(
                            "SELECT * FROM scientific_claim_proposals WHERE proposal_id=?",
                            (predecessor,),
                        ).fetchone()
                        predecessor_value = (
                            None if predecessor_row is None else self._proposal(predecessor_row)
                        )
                        if predecessor_value is None and predecessor in by_id:
                            predecessor_value = by_id[predecessor]
                        if predecessor_value is None:
                            raise KeyError(predecessor)
                        if predecessor_value.proposal_id == proposal.proposal_id:
                            raise ValueError("claim proposal may not supersede itself.")
                        if (
                            predecessor_value.owner_id != proposal.owner_id
                            or predecessor_value.doc_id != proposal.doc_id
                            or predecessor_value.generation != proposal.generation
                            or predecessor_value.content_sha256 != proposal.content_sha256
                            or predecessor_value.profile_fingerprint != proposal.profile_fingerprint
                        ):
                            raise PermissionError(
                                "claim correction must remain in the same document generation scope."
                            )
                        successor = connection.execute(
                            "SELECT proposal_id FROM scientific_claim_proposals "
                            "WHERE supersedes_proposal_id=?",
                            (predecessor,),
                        ).fetchone()
                        if successor is not None and successor["proposal_id"] != proposal.proposal_id:
                            raise RuntimeError(
                                "claim proposal already has a different correction successor."
                            )
                    row = connection.execute(
                        "SELECT * FROM scientific_claim_proposals WHERE proposal_id=?",
                        (proposal.proposal_id,),
                    ).fetchone()
                    if row is None:
                        connection.execute(
                            """
                            INSERT INTO scientific_claim_proposals VALUES (
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1
                            )
                            """,
                            (
                                proposal.proposal_id,
                                proposal.owner_id,
                                proposal.doc_id,
                                proposal.generation,
                                proposal.content_sha256,
                                proposal.profile_fingerprint,
                                proposal.supersedes_proposal_id,
                                proposal.proposal_digest,
                                payloads[proposal.proposal_id],
                                proposal.created_at,
                            ),
                        )
                    elif self._proposal(row).proposal_digest != proposal.proposal_digest:
                        raise RuntimeError("claim proposal identity collision detected.")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return tuple(self.get_proposal(value.proposal_id) for value in values)

    def get_proposal(self, proposal_id: str) -> ScientificClaimProposal:
        selected = _digest(proposal_id, "proposal_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scientific_claim_proposals WHERE proposal_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return self._proposal(row)

    def get_decision(self, proposal_id: str) -> ClaimReviewDecision | None:
        selected = _digest(proposal_id, "proposal_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scientific_claim_decisions WHERE proposal_id=?",
                (selected,),
            ).fetchone()
        return None if row is None else self._decision(row)

    def get_authorization(self, proposal_id: str) -> ClaimReviewAuthorization | None:
        selected = _digest(proposal_id, "proposal_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scientific_claim_authorizations WHERE proposal_id=?",
                (selected,),
            ).fetchone()
        return None if row is None else self._authorization(row)

    def get_successor(self, proposal_id: str) -> ScientificClaimProposal | None:
        selected = _digest(proposal_id, "proposal_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scientific_claim_proposals "
                "WHERE supersedes_proposal_id=?",
                (selected,),
            ).fetchone()
        return None if row is None else self._proposal(row)

    def governed_decide(
        self,
        decision: ClaimReviewDecision,
        authorization: ClaimReviewAuthorization,
    ) -> tuple[ClaimReviewDecision, ClaimReviewAuthorization]:
        if not isinstance(decision, ClaimReviewDecision):
            raise ValueError("decision must be ClaimReviewDecision.")
        if not isinstance(authorization, ClaimReviewAuthorization):
            raise ValueError("authorization must be ClaimReviewAuthorization.")
        if (
            authorization.proposal_id != decision.proposal_id
            or authorization.decision_id != decision.decision_id
            or authorization.owner_id != decision.owner_id
            or authorization.decision != decision.decision
            or authorization.reviewer_id != decision.reviewer_id
        ):
            raise ValueError("authorization differs from claim review decision scope.")
        decision_payload = json.dumps(
            asdict(decision),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        authorization_payload = json.dumps(
            asdict(authorization),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                proposal_row = connection.execute(
                    "SELECT * FROM scientific_claim_proposals WHERE proposal_id=?",
                    (decision.proposal_id,),
                ).fetchone()
                if proposal_row is None:
                    raise KeyError(decision.proposal_id)
                proposal = self._proposal(proposal_row)
                if (
                    proposal.owner_id != decision.owner_id
                    or proposal.owner_id != authorization.owner_id
                    or proposal.doc_id != authorization.doc_id
                    or proposal.generation != authorization.generation
                ):
                    raise RuntimeError("claim review escaped proposal scope.")
                if proposal.proposer_id == decision.reviewer_id:
                    raise PermissionError("claim proposal authors may not review their own proposal.")

                if decision.decision == "superseded":
                    replacement_row = connection.execute(
                        "SELECT * FROM scientific_claim_proposals WHERE proposal_id=?",
                        (decision.replacement_proposal_id,),
                    ).fetchone()
                    if replacement_row is None:
                        raise KeyError(decision.replacement_proposal_id)
                    replacement = self._proposal(replacement_row)
                    if (
                        replacement.supersedes_proposal_id != proposal.proposal_id
                        or replacement.owner_id != proposal.owner_id
                        or replacement.doc_id != proposal.doc_id
                        or replacement.generation != proposal.generation
                        or replacement.content_sha256 != proposal.content_sha256
                        or replacement.profile_fingerprint != proposal.profile_fingerprint
                    ):
                        raise PermissionError("replacement proposal is outside correction scope.")
                    if replacement.proposer_id == decision.reviewer_id:
                        raise PermissionError(
                            "replacement authors may not authorize their own correction."
                        )
                    if authorization.replacement_scope_validated is not True:
                        raise ValueError("superseded decision lacks replacement-scope authorization.")
                elif authorization.replacement_scope_validated:
                    raise ValueError("non-superseded decision claims replacement validation.")

                if decision.decision == "approved" and proposal.supersedes_proposal_id is not None:
                    predecessor_decision = connection.execute(
                        "SELECT * FROM scientific_claim_decisions WHERE proposal_id=?",
                        (proposal.supersedes_proposal_id,),
                    ).fetchone()
                    if predecessor_decision is None:
                        raise RuntimeError(
                            "corrected claim cannot be approved before predecessor supersession."
                        )
                    predecessor = self._decision(predecessor_decision)
                    if (
                        predecessor.decision != "superseded"
                        or predecessor.replacement_proposal_id != proposal.proposal_id
                    ):
                        raise RuntimeError(
                            "corrected claim predecessor is not superseded by this proposal."
                        )

                existing_decision_row = connection.execute(
                    "SELECT * FROM scientific_claim_decisions WHERE proposal_id=?",
                    (proposal.proposal_id,),
                ).fetchone()
                existing_authorization_row = connection.execute(
                    "SELECT * FROM scientific_claim_authorizations WHERE proposal_id=?",
                    (proposal.proposal_id,),
                ).fetchone()
                if existing_decision_row is None and existing_authorization_row is None:
                    connection.execute(
                        "INSERT INTO scientific_claim_authorizations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            authorization.proposal_id,
                            authorization.authorization_digest,
                            authorization.decision_id,
                            authorization.owner_id,
                            authorization.reviewer_id,
                            authorization_payload,
                            authorization.authorized_at,
                            authorization.schema_version,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO scientific_claim_decisions VALUES (?, ?, ?, ?, ?, ?, 1)",
                        (
                            decision.proposal_id,
                            decision.decision_id,
                            decision.owner_id,
                            decision.decision,
                            decision_payload,
                            decision.decided_at,
                        ),
                    )
                elif existing_decision_row is None or existing_authorization_row is None:
                    raise RuntimeError("claim review decision/authorization atomicity is corrupt.")
                else:
                    existing_decision = self._decision(existing_decision_row)
                    existing_authorization = self._authorization(existing_authorization_row)
                    if _semantic_decision(existing_decision) != _semantic_decision(decision):
                        raise RuntimeError("claim proposal already has a different terminal decision.")
                    if existing_authorization.authorization_digest != authorization.authorization_digest:
                        raise RuntimeError("claim proposal already has a different authorization.")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        stored_decision = self.get_decision(decision.proposal_id)
        stored_authorization = self.get_authorization(decision.proposal_id)
        if stored_decision is None or stored_authorization is None:
            raise RuntimeError("stored claim review state disappeared.")
        return stored_decision, stored_authorization

    def list(
        self,
        *,
        owner_id: str,
        doc_id: str,
        generation: int | None = None,
        decision: str | None = None,
        limit: int = 100,
    ) -> tuple[
        tuple[ScientificClaimProposal, ClaimReviewDecision | None, ClaimReviewAuthorization | None],
        ...,
    ]:
        owner = normalize_owner_id(owner_id)
        document = _identifier(doc_id, "doc_id", 200)
        selected_generation = None if generation is None else _integer(
            generation, "generation", 1, 2**63 - 1
        )
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        selected_decision = None
        if decision is not None:
            selected_decision = _identifier(decision, "decision", 20)
            if selected_decision not in REVIEW_DECISIONS | {"pending"}:
                raise ValueError("decision filter is unsupported.")
        query = """
            SELECT p.*, d.payload_json AS decision_payload,
                   a.payload_json AS authorization_payload
            FROM scientific_claim_proposals p
            LEFT JOIN scientific_claim_decisions d ON d.proposal_id=p.proposal_id
            LEFT JOIN scientific_claim_authorizations a ON a.proposal_id=p.proposal_id
            WHERE p.owner_id=? AND p.doc_id=?
        """
        params: list[Any] = [owner, document]
        if selected_generation is not None:
            query += " AND p.generation=?"
            params.append(selected_generation)
        if selected_decision is not None:
            if selected_decision == "pending":
                query += " AND d.proposal_id IS NULL"
            else:
                query += " AND d.decision=?"
                params.append(selected_decision)
        query += " ORDER BY p.created_at, p.proposal_id LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        result = []
        for row in rows:
            proposal = self._proposal(row)
            review = None if row["decision_payload"] is None else _decision_from_payload(
                row["decision_payload"]
            )
            authorization = (
                None
                if row["authorization_payload"] is None
                else _authorization_from_payload(row["authorization_payload"])
            )
            if (review is None) != (authorization is None):
                raise RuntimeError("claim review decision/authorization atomicity is corrupt.")
            result.append((proposal, review, authorization))
        return tuple(result)


__all__ = ["ScientificClaimReviewStore"]
