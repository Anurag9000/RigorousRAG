"""Fail-closed reviewer authorization for semantic relation decisions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from tools.evidence_graph_relation_review import (
    CrossDocumentRelationProposal,
    RelationReviewDecision,
    RelationReviewLedger,
)
from tools.security import normalize_owner_id

_ALLOWED_DECISIONS = frozenset({"approved", "rejected", "superseded"})
_MAX_POLICY_BYTES = 1_000_000
_MAX_REVIEWERS = 1_000
_MAX_SCOPE_VALUES = 1_000
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


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


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("review policy contains a duplicate JSON key.")
        result[key] = value
    return result


def _scope_values(
    value: Any,
    label: str,
    *,
    owner_scope: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a bounded array.")
    if not 1 <= len(value) <= _MAX_SCOPE_VALUES:
        raise ValueError(f"{label} must contain 1-{_MAX_SCOPE_VALUES} entries.")
    selected: set[str] = set()
    for item in value:
        rendered = _identifier(item, label, 500)
        if rendered == "*":
            selected.add(rendered)
        elif owner_scope:
            selected.add(normalize_owner_id(rendered))
        else:
            selected.add(rendered)
    if "*" in selected and len(selected) != 1:
        raise ValueError(f"{label} wildcard may not be combined with explicit entries.")
    return tuple(sorted(selected))


def _allows(scope: tuple[str, ...], value: str) -> bool:
    return scope == ("*",) or value in scope


@dataclass(frozen=True)
class ReviewerGrant:
    reviewer_id: str
    owners: tuple[str, ...]
    graph_set_keys: tuple[str, ...]
    decisions: tuple[str, ...]
    expires_at: float | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reviewer_id",
            _identifier(self.reviewer_id, "reviewer_id", 200),
        )
        object.__setattr__(
            self,
            "owners",
            _scope_values(self.owners, "owners", owner_scope=True),
        )
        object.__setattr__(
            self,
            "graph_set_keys",
            _scope_values(self.graph_set_keys, "graph_set_keys"),
        )
        decisions = _scope_values(self.decisions, "decisions")
        if decisions == ("*",) or any(value not in _ALLOWED_DECISIONS for value in decisions):
            raise ValueError("decisions contains an unsupported value.")
        object.__setattr__(self, "decisions", decisions)
        if self.expires_at is not None:
            object.__setattr__(
                self,
                "expires_at",
                _timestamp(self.expires_at, "expires_at"),
            )
        if self.schema_version != 1:
            raise ValueError("reviewer grant schema is unsupported.")

    @property
    def grant_digest(self) -> str:
        return _sha256(asdict(self))

    def permits(
        self,
        *,
        owner_id: str,
        graph_set_key: str,
        decision: str,
        now: float,
    ) -> bool:
        return bool(
            _allows(self.owners, owner_id)
            and _allows(self.graph_set_keys, graph_set_key)
            and decision in self.decisions
            and (self.expires_at is None or now <= self.expires_at)
        )


@dataclass(frozen=True)
class RelationReviewPolicy:
    reviewers: tuple[ReviewerGrant, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reviewers, tuple)
            or not 1 <= len(self.reviewers) <= _MAX_REVIEWERS
            or any(not isinstance(value, ReviewerGrant) for value in self.reviewers)
        ):
            raise ValueError("reviewers must be a bounded non-empty tuple of grants.")
        ordered = tuple(sorted(self.reviewers, key=lambda value: value.reviewer_id))
        if len({value.reviewer_id for value in ordered}) != len(ordered):
            raise ValueError("reviewer IDs must be unique.")
        object.__setattr__(self, "reviewers", ordered)
        if self.schema_version != 1:
            raise ValueError("relation review policy schema is unsupported.")

    @property
    def policy_digest(self) -> str:
        return _sha256(asdict(self))

    def grant_for(self, reviewer_id: str) -> ReviewerGrant:
        selected = _identifier(reviewer_id, "reviewer_id", 200)
        for grant in self.reviewers:
            if grant.reviewer_id == selected:
                return grant
        raise PermissionError("reviewer is not authorized by policy.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RelationReviewPolicy":
        if not isinstance(value, Mapping) or set(value) != {"schema_version", "reviewers"}:
            raise ValueError("relation review policy schema is invalid.")
        if value["schema_version"] != 1:
            raise ValueError("relation review policy schema is unsupported.")
        raw_reviewers = value["reviewers"]
        if (
            isinstance(raw_reviewers, (str, bytes, bytearray))
            or not isinstance(raw_reviewers, Sequence)
            or not 1 <= len(raw_reviewers) <= _MAX_REVIEWERS
        ):
            raise ValueError("reviewers must be a bounded non-empty array.")
        grants: list[ReviewerGrant] = []
        allowed = {
            "reviewer_id",
            "owners",
            "graph_set_keys",
            "decisions",
            "expires_at",
        }
        required = allowed - {"expires_at"}
        for raw in raw_reviewers:
            if not isinstance(raw, Mapping) or not required <= set(raw) <= allowed:
                raise ValueError("reviewer grant schema is invalid.")
            grants.append(
                ReviewerGrant(
                    reviewer_id=raw["reviewer_id"],
                    owners=_scope_values(
                        raw["owners"], "owners", owner_scope=True
                    ),
                    graph_set_keys=_scope_values(
                        raw["graph_set_keys"], "graph_set_keys"
                    ),
                    decisions=_scope_values(raw["decisions"], "decisions"),
                    expires_at=raw.get("expires_at"),
                )
            )
        return cls(reviewers=tuple(grants))


@dataclass(frozen=True)
class ReviewAuthorization:
    proposal_id: str
    decision_id: str
    owner_id: str
    graph_set_key: str
    decision: str
    reviewer_id: str
    policy_digest: str
    grant_digest: str
    authorization_digest: str
    authorized_at: float
    separation_of_duties_enforced: bool = True
    replacement_scope_validated: bool = False

    def __post_init__(self) -> None:
        for name in (
            "proposal_id",
            "decision_id",
            "policy_digest",
            "grant_digest",
            "authorization_digest",
        ):
            value = _identifier(getattr(self, name), name, 64).lower()
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a SHA-256 digest.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(
            self,
            "graph_set_key",
            _identifier(self.graph_set_key, "graph_set_key", 500),
        )
        selected_decision = _identifier(self.decision, "decision", 20)
        if selected_decision not in _ALLOWED_DECISIONS:
            raise ValueError("authorization decision is unsupported.")
        object.__setattr__(self, "decision", selected_decision)
        object.__setattr__(
            self,
            "reviewer_id",
            _identifier(self.reviewer_id, "reviewer_id", 200),
        )
        object.__setattr__(
            self,
            "authorized_at",
            _timestamp(self.authorized_at, "authorized_at"),
        )
        if self.separation_of_duties_enforced is not True:
            raise ValueError("separation_of_duties_enforced must remain true.")
        if not isinstance(self.replacement_scope_validated, bool):
            raise ValueError("replacement_scope_validated must be boolean.")


class GovernedRelationReviewService:
    """Authorize, journal and persist one immutable terminal relation decision."""

    def __init__(
        self,
        *,
        ledger: RelationReviewLedger,
        policy: RelationReviewPolicy,
        authorization_store: Any,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(ledger, RelationReviewLedger):
            raise ValueError("ledger must be RelationReviewLedger.")
        if not isinstance(policy, RelationReviewPolicy):
            raise ValueError("policy must be RelationReviewPolicy.")
        if not all(
            callable(getattr(authorization_store, name, None))
            for name in ("prepare", "mark_committed", "get")
        ):
            raise ValueError("authorization_store lacks the required journal boundary.")
        if not callable(clock):
            raise ValueError("clock must be callable.")
        self.ledger = ledger
        self.policy = policy
        self.authorization_store = authorization_store
        self.clock = clock

    def _authorization(
        self,
        proposal: CrossDocumentRelationProposal,
        decision: RelationReviewDecision,
        *,
        now: float,
    ) -> ReviewAuthorization:
        grant = self.policy.grant_for(decision.reviewer_id)
        if not grant.permits(
            owner_id=proposal.owner_id,
            graph_set_key=proposal.graph_set_key,
            decision=decision.decision,
            now=now,
        ):
            raise PermissionError("reviewer grant does not permit this decision scope.")
        if decision.reviewer_id == proposal.proposer_id:
            raise PermissionError("proposal authors may not review their own proposal.")

        replacement_scope_validated = False
        if decision.replacement_proposal_id is not None:
            replacement = self.ledger.get_proposal(decision.replacement_proposal_id)
            if replacement.proposal_id == proposal.proposal_id:
                raise ValueError("replacement proposal must differ from the original.")
            if (
                replacement.owner_id != proposal.owner_id
                or replacement.graph_set_key != proposal.graph_set_key
                or replacement.relation_key != proposal.relation_key
            ):
                raise PermissionError(
                    "replacement proposal must remain in the same relation scope."
                )
            if decision.reviewer_id == replacement.proposer_id:
                raise PermissionError(
                    "replacement authors may not authorize their own replacement."
                )
            replacement_scope_validated = True

        stable = {
            "scope": "rigorousrag-relation-review-authorization-v1",
            "proposal_id": proposal.proposal_id,
            "decision_id": decision.decision_id,
            "owner_id": proposal.owner_id,
            "graph_set_key": proposal.graph_set_key,
            "decision": decision.decision,
            "reviewer_id": decision.reviewer_id,
            "grant_digest": grant.grant_digest,
            "separation_of_duties_enforced": True,
            "replacement_scope_validated": replacement_scope_validated,
        }
        return ReviewAuthorization(
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            owner_id=proposal.owner_id,
            graph_set_key=proposal.graph_set_key,
            decision=decision.decision,
            reviewer_id=decision.reviewer_id,
            policy_digest=self.policy.policy_digest,
            grant_digest=grant.grant_digest,
            authorization_digest=_sha256(stable),
            authorized_at=now,
            replacement_scope_validated=replacement_scope_validated,
        )

    def decide(
        self,
        decision: RelationReviewDecision,
    ) -> tuple[RelationReviewDecision, Any]:
        if not isinstance(decision, RelationReviewDecision):
            raise ValueError("decision must be RelationReviewDecision.")
        proposal = self.ledger.get_proposal(decision.proposal_id)
        if proposal.owner_id != decision.owner_id:
            raise PermissionError("review decision escaped proposal owner scope.")

        existing = self.ledger.get_decision(decision.proposal_id)
        if existing is not None:
            if existing != decision:
                raise RuntimeError(
                    "relation proposal already has a different terminal decision."
                )
            receipt = self.authorization_store.get(decision.decision_id)
            if receipt is None:
                raise RuntimeError(
                    "existing relation decision lacks a governed authorization receipt."
                )
            authorization = receipt.authorization
            if (
                authorization.proposal_id != proposal.proposal_id
                or authorization.decision_id != decision.decision_id
                or authorization.owner_id != proposal.owner_id
                or authorization.graph_set_key != proposal.graph_set_key
                or authorization.decision != decision.decision
                or authorization.reviewer_id != decision.reviewer_id
            ):
                raise RuntimeError(
                    "governed authorization receipt differs from the terminal decision."
                )
            return existing, self.authorization_store.mark_committed(
                decision.decision_id, now=self.clock()
            )

        now = _timestamp(self.clock(), "authorization time")
        authorization = self._authorization(proposal, decision, now=now)
        prepared = self.authorization_store.prepare(authorization, now=now)
        if prepared.authorization.authorization_digest != authorization.authorization_digest:
            raise RuntimeError("prepared authorization differs from current reviewer grant.")
        stored = self.ledger.decide(decision)
        if stored != decision:
            raise RuntimeError("stored relation decision differs from authorized decision.")
        committed = self.authorization_store.mark_committed(
            decision.decision_id, now=self.clock()
        )
        return stored, committed


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _policy_path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("review policy path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("review policy path is invalid.")
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
            raise ValueError("review policy path may not contain redirects.")
    return absolute


def _read_policy_file(path: str | os.PathLike[str]) -> str:
    selected = _policy_path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(selected, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_POLICY_BYTES:
            raise ValueError("review policy file is invalid or too large.")
        chunks: list[bytes] = []
        remaining = _MAX_POLICY_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > _MAX_POLICY_BYTES:
            raise ValueError("review policy file is too large.")
        return payload.decode("utf-8")
    finally:
        os.close(descriptor)


def load_relation_review_policy(
    *,
    json_text: str | None = None,
    path: str | os.PathLike[str] | None = None,
) -> RelationReviewPolicy:
    if json_text is not None and path is not None:
        raise ValueError("configure either inline or file review policy, not both.")
    if json_text is None and path is None:
        inline = os.getenv("EVIDENCE_GRAPH_REVIEW_POLICY_JSON")
        configured_path = os.getenv("EVIDENCE_GRAPH_REVIEW_POLICY_PATH")
        if inline and configured_path:
            raise RuntimeError("multiple relation review policy sources are configured.")
        json_text = inline or None
        path = configured_path or None
    if json_text is None and path is None:
        raise RuntimeError("relation review policy is not configured.")
    payload = _read_policy_file(path) if path is not None else json_text
    if not isinstance(payload, str) or not payload or len(payload.encode("utf-8")) > _MAX_POLICY_BYTES:
        raise ValueError("relation review policy JSON is invalid or too large.")
    try:
        raw = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("relation review policy JSON is invalid.") from exc
    return RelationReviewPolicy.from_mapping(raw)


__all__ = [
    "GovernedRelationReviewService",
    "RelationReviewPolicy",
    "ReviewAuthorization",
    "ReviewerGrant",
    "load_relation_review_policy",
]
