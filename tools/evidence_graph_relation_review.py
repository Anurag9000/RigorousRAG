"""Reviewed proposal and decision ledger for cross-document graph relations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tools.evidence_graph_sets import (
    ExplicitCrossDocumentRelation,
    _CROSS_EDGE_TYPES,
)
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_METADATA_ITEMS = 64
_MAX_LIMIT = 10_000
_PROPOSER_KINDS = frozenset({"human", "model", "rule"})
_DECISIONS = frozenset({"approved", "rejected", "superseded"})


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or any(
        ord(character) < 32 or ord(character) == 127 for character in cleaned
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _identifier(value, label, 64).lower()
    if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


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
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping.")
    result: dict[str, Any] = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= _MAX_METADATA_ITEMS:
            raise ValueError("metadata contains too many fields.")
        selected = _identifier(key, "metadata key", 200)
        if item is None or isinstance(item, (bool, int)):
            result[selected] = item
        elif isinstance(item, float) and math.isfinite(item):
            result[selected] = item
        elif isinstance(item, str) and len(item) <= 10_000 and "\x00" not in item:
            result[selected] = item
        else:
            raise ValueError("metadata contains an unsupported value.")
    return result


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


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("relation review database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("relation review database path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or bool(
            int(getattr(info, "st_file_attributes", 0)) & _REPARSE
        ):
            raise ValueError("relation review database path may not contain redirects.")
    return absolute


@dataclass(frozen=True)
class RelationEndpoint:
    doc_id: str
    generation: int
    graph_digest: str
    node_id: str
    provenance_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id", 200))
        object.__setattr__(
            self, "generation", _integer(self.generation, "generation", 1, 2**63 - 1)
        )
        for name in ("graph_digest", "node_id", "provenance_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    @property
    def endpoint_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class CrossDocumentRelationProposal:
    proposal_id: str
    owner_id: str
    graph_set_key: str
    relation_key: str
    source: RelationEndpoint
    target: RelationEndpoint
    edge_type: str
    proposer_kind: str
    proposer_id: str
    evidence_digest: str
    extractor_name: str | None = None
    extractor_version: str | None = None
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    schema_version: int = 1

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        key = _identifier(self.graph_set_key, "graph_set_key", 500)
        relation_key = _identifier(self.relation_key, "relation_key", 2_000)
        if not isinstance(self.source, RelationEndpoint) or not isinstance(
            self.target, RelationEndpoint
        ):
            raise ValueError("proposal endpoints must be RelationEndpoint.")
        if self.source.doc_id == self.target.doc_id:
            raise ValueError("cross-document proposals must connect different documents.")
        edge_type = _identifier(self.edge_type, "edge_type", 50)
        if edge_type not in _CROSS_EDGE_TYPES:
            raise ValueError("proposal edge_type is unsupported.")
        proposer_kind = _identifier(self.proposer_kind, "proposer_kind", 20)
        if proposer_kind not in _PROPOSER_KINDS:
            raise ValueError("proposer_kind is unsupported.")
        proposer_id = _identifier(self.proposer_id, "proposer_id", 200)
        evidence = _digest(self.evidence_digest, "evidence_digest")
        extractor_name = None if self.extractor_name is None else _identifier(
            self.extractor_name, "extractor_name", 200
        )
        extractor_version = None if self.extractor_version is None else _identifier(
            self.extractor_version, "extractor_version", 200
        )
        if proposer_kind == "human" and (extractor_name is not None or extractor_version is not None):
            raise ValueError("human proposals may not claim an extractor.")
        if proposer_kind in {"model", "rule"} and (
            extractor_name is None or extractor_version is None
        ):
            raise ValueError("model/rule proposals require extractor identity and version.")
        if isinstance(self.weight, bool):
            raise ValueError("weight must be finite and between 0 and 1.")
        weight = float(self.weight)
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("weight must be finite and between 0 and 1.")
        metadata = _metadata(self.metadata)
        expected = _sha256(
            {
                "scope": "rigorousrag-cross-document-relation-proposal-v1",
                "owner_id": owner,
                "graph_set_key": key,
                "relation_key": relation_key,
                "source": self.source.endpoint_digest,
                "target": self.target.endpoint_digest,
                "edge_type": edge_type,
                "proposer_kind": proposer_kind,
                "proposer_id": proposer_id,
                "evidence_digest": evidence,
                "extractor_name": extractor_name,
                "extractor_version": extractor_version,
                "weight": weight,
                "metadata": metadata,
            }
        )
        if _digest(self.proposal_id, "proposal_id") != expected:
            raise ValueError("proposal_id does not match deterministic proposal identity.")
        object.__setattr__(self, "proposal_id", expected)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "graph_set_key", key)
        object.__setattr__(self, "relation_key", relation_key)
        object.__setattr__(self, "edge_type", edge_type)
        object.__setattr__(self, "proposer_kind", proposer_kind)
        object.__setattr__(self, "proposer_id", proposer_id)
        object.__setattr__(self, "evidence_digest", evidence)
        object.__setattr__(self, "extractor_name", extractor_name)
        object.__setattr__(self, "extractor_version", extractor_version)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        if self.schema_version != 1:
            raise ValueError("proposal schema is unsupported.")

    @property
    def proposal_digest(self) -> str:
        value = asdict(self)
        value.pop("created_at", None)
        return _sha256(value)

    @classmethod
    def create(cls, *, created_at: float | None = None, **kwargs: Any) -> "CrossDocumentRelationProposal":
        timestamp = time.time() if created_at is None else created_at
        source = kwargs["source"]
        target = kwargs["target"]
        if not isinstance(source, RelationEndpoint) or not isinstance(target, RelationEndpoint):
            raise ValueError("proposal endpoints must be RelationEndpoint.")
        owner = normalize_owner_id(kwargs["owner_id"])
        key = _identifier(kwargs["graph_set_key"], "graph_set_key", 500)
        relation_key = _identifier(kwargs["relation_key"], "relation_key", 2_000)
        edge_type = _identifier(kwargs["edge_type"], "edge_type", 50)
        if edge_type not in _CROSS_EDGE_TYPES:
            raise ValueError("proposal edge_type is unsupported.")
        proposer_kind = _identifier(kwargs["proposer_kind"], "proposer_kind", 20)
        proposer_id = _identifier(kwargs["proposer_id"], "proposer_id", 200)
        evidence = _digest(kwargs["evidence_digest"], "evidence_digest")
        extractor_name = (
            None
            if kwargs.get("extractor_name") is None
            else _identifier(kwargs["extractor_name"], "extractor_name", 200)
        )
        extractor_version = (
            None
            if kwargs.get("extractor_version") is None
            else _identifier(kwargs["extractor_version"], "extractor_version", 200)
        )
        if proposer_kind not in _PROPOSER_KINDS:
            raise ValueError("proposer_kind is unsupported.")
        if isinstance(kwargs.get("weight", 1.0), bool):
            raise ValueError("weight must be finite and between 0 and 1.")
        weight = float(kwargs.get("weight", 1.0))
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("weight must be finite and between 0 and 1.")
        metadata = _metadata(kwargs.get("metadata"))
        stable = {
            "scope": "rigorousrag-cross-document-relation-proposal-v1",
            "owner_id": owner,
            "graph_set_key": key,
            "relation_key": relation_key,
            "source": source.endpoint_digest,
            "target": target.endpoint_digest,
            "edge_type": edge_type,
            "proposer_kind": proposer_kind,
            "proposer_id": proposer_id,
            "evidence_digest": evidence,
            "extractor_name": extractor_name,
            "extractor_version": extractor_version,
            "weight": weight,
            "metadata": metadata,
        }
        return cls(
            proposal_id=_sha256(stable),
            owner_id=owner,
            graph_set_key=key,
            relation_key=relation_key,
            source=source,
            target=target,
            edge_type=edge_type,
            proposer_kind=proposer_kind,
            proposer_id=proposer_id,
            evidence_digest=evidence,
            extractor_name=extractor_name,
            extractor_version=extractor_version,
            weight=weight,
            metadata=metadata,
            created_at=timestamp,
        )


@dataclass(frozen=True)
class RelationReviewDecision:
    decision_id: str
    proposal_id: str
    owner_id: str
    decision: str
    reviewer_id: str
    reason_code: str
    replacement_proposal_id: str | None
    decided_at: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        proposal = _digest(self.proposal_id, "proposal_id")
        owner = normalize_owner_id(self.owner_id)
        decision = _identifier(self.decision, "decision", 20)
        if decision not in _DECISIONS:
            raise ValueError("review decision is unsupported.")
        reviewer = _identifier(self.reviewer_id, "reviewer_id", 200)
        reason = _identifier(self.reason_code, "reason_code", 200)
        replacement = None if self.replacement_proposal_id is None else _digest(
            self.replacement_proposal_id, "replacement_proposal_id"
        )
        if decision == "superseded" and replacement is None:
            raise ValueError("superseded decisions require a replacement proposal.")
        if decision != "superseded" and replacement is not None:
            raise ValueError("only superseded decisions may name a replacement proposal.")
        expected = _sha256(
            {
                "scope": "rigorousrag-relation-review-decision-v1",
                "proposal_id": proposal,
                "owner_id": owner,
                "decision": decision,
                "reviewer_id": reviewer,
                "reason_code": reason,
                "replacement_proposal_id": replacement,
            }
        )
        if _digest(self.decision_id, "decision_id") != expected:
            raise ValueError("decision_id does not match deterministic decision identity.")
        object.__setattr__(self, "decision_id", expected)
        object.__setattr__(self, "proposal_id", proposal)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "reviewer_id", reviewer)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(self, "replacement_proposal_id", replacement)
        object.__setattr__(self, "decided_at", _timestamp(self.decided_at, "decided_at"))
        if self.schema_version != 1:
            raise ValueError("decision schema is unsupported.")

    @classmethod
    def create(cls, *, decided_at: float | None = None, **kwargs: Any) -> "RelationReviewDecision":
        proposal = _digest(kwargs["proposal_id"], "proposal_id")
        owner = normalize_owner_id(kwargs["owner_id"])
        decision = _identifier(kwargs["decision"], "decision", 20)
        reviewer = _identifier(kwargs["reviewer_id"], "reviewer_id", 200)
        reason = _identifier(kwargs["reason_code"], "reason_code", 200)
        replacement = (
            None
            if kwargs.get("replacement_proposal_id") is None
            else _digest(kwargs["replacement_proposal_id"], "replacement_proposal_id")
        )
        stable = {
            "scope": "rigorousrag-relation-review-decision-v1",
            "proposal_id": proposal,
            "owner_id": owner,
            "decision": decision,
            "reviewer_id": reviewer,
            "reason_code": reason,
            "replacement_proposal_id": replacement,
        }
        return cls(
            decision_id=_sha256(stable),
            proposal_id=proposal,
            owner_id=owner,
            decision=decision,
            reviewer_id=reviewer,
            reason_code=reason,
            replacement_proposal_id=replacement,
            decided_at=time.time() if decided_at is None else decided_at,
        )


class RelationReviewLedger:
    """Immutable proposal plus one terminal reviewer decision per proposal."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if not stat.S_ISDIR(parent.st_mode):
            raise ValueError("relation review database parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("relation review database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity:
            raise RuntimeError("relation review database parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("relation review database identity changed.")

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
                CREATE TABLE IF NOT EXISTS relation_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    graph_set_key TEXT NOT NULL,
                    proposal_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS relation_proposal_scope
                    ON relation_proposals(owner_id, graph_set_key, created_at, proposal_id);
                CREATE TABLE IF NOT EXISTS relation_decisions (
                    proposal_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    decided_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL,
                    FOREIGN KEY(proposal_id) REFERENCES relation_proposals(proposal_id)
                        ON DELETE RESTRICT
                );
                """
            )

    @staticmethod
    def _proposal(row: sqlite3.Row) -> CrossDocumentRelationProposal:
        try:
            raw = json.loads(row["payload_json"])
            raw["source"] = RelationEndpoint(**raw["source"])
            raw["target"] = RelationEndpoint(**raw["target"])
            value = CrossDocumentRelationProposal(**raw)
        except Exception as exc:
            raise RuntimeError("stored relation proposal is corrupt.") from exc
        if value.proposal_digest != row["proposal_digest"]:
            raise RuntimeError("stored relation proposal digest is corrupt.")
        return value

    @staticmethod
    def _decision(row: sqlite3.Row) -> RelationReviewDecision:
        try:
            return RelationReviewDecision(**json.loads(row["payload_json"]))
        except Exception as exc:
            raise RuntimeError("stored relation decision is corrupt.") from exc

    def submit(self, proposal: CrossDocumentRelationProposal) -> CrossDocumentRelationProposal:
        if not isinstance(proposal, CrossDocumentRelationProposal):
            raise ValueError("proposal must be CrossDocumentRelationProposal.")
        payload = json.dumps(asdict(proposal), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM relation_proposals WHERE proposal_id=?",
                    (proposal.proposal_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO relation_proposals VALUES (?, ?, ?, ?, ?, ?, 1)",
                        (
                            proposal.proposal_id,
                            proposal.owner_id,
                            proposal.graph_set_key,
                            proposal.proposal_digest,
                            payload,
                            proposal.created_at,
                        ),
                    )
                elif self._proposal(row).proposal_digest != proposal.proposal_digest:
                    raise RuntimeError("relation proposal identity collision detected.")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get_proposal(proposal.proposal_id)

    def decide(self, decision: RelationReviewDecision) -> RelationReviewDecision:
        if not isinstance(decision, RelationReviewDecision):
            raise ValueError("decision must be RelationReviewDecision.")
        payload = json.dumps(asdict(decision), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                proposal_row = connection.execute(
                    "SELECT owner_id FROM relation_proposals WHERE proposal_id=?",
                    (decision.proposal_id,),
                ).fetchone()
                if proposal_row is None:
                    raise KeyError(decision.proposal_id)
                if proposal_row["owner_id"] != decision.owner_id:
                    raise RuntimeError("review decision escaped proposal owner scope.")
                if decision.replacement_proposal_id is not None:
                    replacement = connection.execute(
                        "SELECT owner_id FROM relation_proposals WHERE proposal_id=?",
                        (decision.replacement_proposal_id,),
                    ).fetchone()
                    if replacement is None or replacement["owner_id"] != decision.owner_id:
                        raise RuntimeError("replacement proposal is missing or owner-mismatched.")
                existing = connection.execute(
                    "SELECT * FROM relation_decisions WHERE proposal_id=?",
                    (decision.proposal_id,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO relation_decisions VALUES (?, ?, ?, ?, ?, ?, 1)",
                        (
                            decision.proposal_id,
                            decision.decision_id,
                            decision.owner_id,
                            decision.decision,
                            payload,
                            decision.decided_at,
                        ),
                    )
                elif self._decision(existing) != decision:
                    raise RuntimeError("relation proposal already has a different terminal decision.")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        result = self.get_decision(decision.proposal_id)
        if result is None:
            raise RuntimeError("stored relation decision disappeared.")
        return result

    def get_proposal(self, proposal_id: str) -> CrossDocumentRelationProposal:
        selected = _digest(proposal_id, "proposal_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM relation_proposals WHERE proposal_id=?", (selected,)
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return self._proposal(row)

    def get_decision(self, proposal_id: str) -> RelationReviewDecision | None:
        selected = _digest(proposal_id, "proposal_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM relation_decisions WHERE proposal_id=?", (selected,)
            ).fetchone()
        return None if row is None else self._decision(row)

    def list(
        self,
        *,
        owner_id: str,
        graph_set_key: str,
        decision: str | None = None,
        limit: int = 100,
    ) -> tuple[tuple[CrossDocumentRelationProposal, RelationReviewDecision | None], ...]:
        owner = normalize_owner_id(owner_id)
        key = _identifier(graph_set_key, "graph_set_key", 500)
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        if decision is not None:
            selected = _identifier(decision, "decision", 20)
            if selected not in _DECISIONS | {"pending"}:
                raise ValueError("decision filter is unsupported.")
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.*, d.decision_id AS d_decision_id,
                       d.owner_id AS d_owner_id, d.decision AS d_decision,
                       d.payload_json AS d_payload_json,
                       d.decided_at AS d_decided_at,
                       d.schema_version AS d_schema_version
                FROM relation_proposals p
                LEFT JOIN relation_decisions d ON d.proposal_id=p.proposal_id
                WHERE p.owner_id=? AND p.graph_set_key=?
                  AND (? IS NULL OR (?='pending' AND d.proposal_id IS NULL) OR d.decision=?)
                ORDER BY p.created_at, p.proposal_id LIMIT ?
                """,
                (owner, key, decision, decision, decision, count),
            ).fetchall()
        values = []
        for row in rows:
            proposal = self._proposal(row)
            review = None
            if row["d_decision_id"] is not None:
                review = RelationReviewDecision(**json.loads(row["d_payload_json"]))
            values.append((proposal, review))
        return tuple(values)


def approved_relations(
    *,
    owner_id: str,
    graph_set_key: str,
    proposal_ids: Iterable[str],
    authority_views: Iterable[Any],
    ledger: RelationReviewLedger,
) -> tuple[ExplicitCrossDocumentRelation, ...]:
    owner = normalize_owner_id(owner_id)
    key = _identifier(graph_set_key, "graph_set_key", 500)
    views = tuple(authority_views)
    lookup: dict[tuple[str, str], tuple[int, str, str]] = {}
    for view in views:
        if getattr(view, "authoritative_current", None) is not True:
            raise RuntimeError("approved relations require current authoritative graph views.")
        batch = getattr(view, "batch", None)
        if batch is None or batch.owner_id != owner:
            raise RuntimeError("authority view escaped owner scope.")
        for node in batch.nodes:
            provenance = getattr(node, "provenance_digest", None) or _sha256(asdict(node))
            lookup[(batch.doc_id, node.node_id)] = (
                batch.generation,
                batch.graph_digest,
                provenance,
            )
    results = []
    seen: set[str] = set()
    for raw_id in proposal_ids:
        proposal = ledger.get_proposal(raw_id)
        review = ledger.get_decision(proposal.proposal_id)
        if proposal.owner_id != owner or proposal.graph_set_key != key:
            raise RuntimeError("approved proposal escaped graph-set scope.")
        if review is None or review.decision != "approved":
            raise RuntimeError("proposal is not approved.")
        for endpoint in (proposal.source, proposal.target):
            actual = lookup.get((endpoint.doc_id, endpoint.node_id))
            if actual != (
                endpoint.generation,
                endpoint.graph_digest,
                endpoint.provenance_digest,
            ):
                raise RuntimeError("approved proposal endpoint is stale or missing.")
        if proposal.relation_key in seen:
            raise RuntimeError("approved relation keys must be unique.")
        seen.add(proposal.relation_key)
        results.append(
            ExplicitCrossDocumentRelation(
                relation_key=proposal.relation_key,
                source_doc_id=proposal.source.doc_id,
                source_node_id=proposal.source.node_id,
                target_doc_id=proposal.target.doc_id,
                target_node_id=proposal.target.node_id,
                edge_type=proposal.edge_type,
                weight=proposal.weight,
                metadata={
                    **dict(proposal.metadata),
                    "proposal_id": proposal.proposal_id,
                    "review_decision_id": review.decision_id,
                    "reviewer_id": review.reviewer_id,
                    "evidence_digest": proposal.evidence_digest,
                },
            )
        )
    return tuple(results)


__all__ = [
    "CrossDocumentRelationProposal",
    "RelationEndpoint",
    "RelationReviewDecision",
    "RelationReviewLedger",
    "approved_relations",
]
