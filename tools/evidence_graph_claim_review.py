"""Governed claim review, policy loading, and approved GraphAnnotation conversion."""

from __future__ import annotations

import json
import os
import stat
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

from tools.evidence_graph_builder import GraphAnnotation
from tools.evidence_graph_claim_contracts import (
    ClaimReviewAuthorization,
    ClaimReviewDecision,
    ClaimReviewPolicy,
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.evidence_graph_claim_store import ScientificClaimReviewStore
from tools.evidence_graph_relation_actor import (
    ReviewActorBinding,
    require_relation_review_actor,
)
from tools.security import normalize_owner_id

_MAX_POLICY_BYTES = 1_000_000
_MAX_PATH = 4096
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("claim review policy contains a duplicate JSON key.")
        result[key] = value
    return result


def _policy_path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("claim review policy path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("claim review policy path is invalid.")
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
            raise ValueError("claim review policy path may not contain redirects.")
    return absolute


def _read_policy(path: str | os.PathLike[str]) -> bytes:
    selected = _policy_path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(selected, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= _MAX_POLICY_BYTES:
            raise ValueError("claim review policy file is invalid or too large.")
        payload = os.read(descriptor, _MAX_POLICY_BYTES + 1)
        if len(payload) > _MAX_POLICY_BYTES:
            raise ValueError("claim review policy file is too large.")
        return payload
    finally:
        os.close(descriptor)


def load_claim_review_policy(
    *,
    path: str | os.PathLike[str] | None = None,
    policy_json: str | None = None,
) -> ClaimReviewPolicy:
    """Load exactly one strict claim-review policy source; absence fails closed."""

    if path is None and policy_json is None:
        configured_path = os.getenv("EVIDENCE_GRAPH_CLAIM_REVIEW_POLICY_PATH")
        configured_json = os.getenv("EVIDENCE_GRAPH_CLAIM_REVIEW_POLICY_JSON")
        path = configured_path if configured_path else None
        policy_json = configured_json if configured_json else None
    if (path is None) == (policy_json is None):
        raise RuntimeError("configure exactly one claim review policy source.")
    payload = _read_policy(path) if path is not None else policy_json.encode("utf-8")
    if not 1 <= len(payload) <= _MAX_POLICY_BYTES:
        raise ValueError("claim review policy size is invalid.")
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("claim review policy JSON is invalid.") from exc
    return ClaimReviewPolicy.from_mapping(raw)


class GovernedScientificClaimReviewService:
    """Actor-bound, policy-scoped, atomic terminal review of claim proposals."""

    def __init__(
        self,
        *,
        store: ScientificClaimReviewStore,
        policy: ClaimReviewPolicy,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(store, ScientificClaimReviewStore):
            raise ValueError("store must be ScientificClaimReviewStore.")
        if not isinstance(policy, ClaimReviewPolicy):
            raise ValueError("policy must be ClaimReviewPolicy.")
        if not callable(clock):
            raise ValueError("clock must be callable.")
        self.store = store
        self.policy = policy
        self.clock = clock

    def decide(
        self,
        decision: ClaimReviewDecision,
        *,
        actor_binding: ReviewActorBinding,
    ) -> tuple[ClaimReviewDecision, ClaimReviewAuthorization]:
        if not isinstance(decision, ClaimReviewDecision):
            raise ValueError("decision must be ClaimReviewDecision.")
        if not isinstance(actor_binding, ReviewActorBinding):
            raise ValueError("actor_binding must be ReviewActorBinding.")
        require_relation_review_actor(decision.reviewer_id, binding=actor_binding)
        proposal = self.store.get_proposal(decision.proposal_id)
        if proposal.owner_id != decision.owner_id:
            raise PermissionError("claim decision owner differs from proposal owner.")
        now = _timestamp(self.clock(), "review time")
        grant = self.policy.grant_for(decision.reviewer_id)
        if not grant.permits(
            owner_id=proposal.owner_id,
            doc_id=proposal.doc_id,
            decision=decision.decision,
            now=now,
        ):
            raise PermissionError("claim reviewer grant does not permit this scope.")
        if proposal.proposer_id == decision.reviewer_id:
            raise PermissionError("claim proposal authors may not review their own proposal.")

        replacement_scope_validated = False
        if decision.replacement_proposal_id is not None:
            replacement = self.store.get_proposal(decision.replacement_proposal_id)
            if (
                replacement.supersedes_proposal_id != proposal.proposal_id
                or replacement.owner_id != proposal.owner_id
                or replacement.doc_id != proposal.doc_id
                or replacement.generation != proposal.generation
                or replacement.content_sha256 != proposal.content_sha256
                or replacement.profile_fingerprint != proposal.profile_fingerprint
            ):
                raise PermissionError("claim replacement is outside correction scope.")
            if replacement.proposer_id == decision.reviewer_id:
                raise PermissionError(
                    "claim replacement authors may not authorize their own correction."
                )
            replacement_scope_validated = True

        authorization = ClaimReviewAuthorization.create(
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            owner_id=proposal.owner_id,
            doc_id=proposal.doc_id,
            generation=proposal.generation,
            decision=decision.decision,
            reviewer_id=decision.reviewer_id,
            policy_digest=self.policy.policy_digest,
            grant_digest=grant.grant_digest,
            replacement_scope_validated=replacement_scope_validated,
            authorized_at=now,
        )
        return self.store.governed_decide(decision, authorization)


def approved_claim_annotations(
    *,
    owner_id: str,
    doc_id: str,
    generation: int,
    content_sha256: str,
    profile_fingerprint: str,
    proposal_ids: Iterable[str],
    store: ScientificClaimReviewStore,
) -> tuple[GraphAnnotation, ...]:
    """Convert only exact authorized approvals into explicit claim annotations."""

    owner = normalize_owner_id(owner_id)
    document = _identifier(doc_id, "doc_id", 200)
    sequence = _integer(generation, "generation", 1, 2**63 - 1)
    content = _digest(content_sha256, "content_sha256")
    profile = _digest(profile_fingerprint, "profile_fingerprint")
    if not isinstance(store, ScientificClaimReviewStore):
        raise ValueError("store must be ScientificClaimReviewStore.")
    if isinstance(proposal_ids, (str, bytes, bytearray)):
        raise ValueError("proposal_ids must be an iterable.")
    selected_ids = tuple(_digest(value, "proposal_id") for value in proposal_ids)
    if not selected_ids or len(selected_ids) > 10_000:
        raise ValueError("proposal_ids must be a bounded non-empty iterable.")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("proposal_ids contains duplicates.")

    annotations: list[GraphAnnotation] = []
    for proposal_id in selected_ids:
        proposal = store.get_proposal(proposal_id)
        decision = store.get_decision(proposal_id)
        authorization = store.get_authorization(proposal_id)
        if decision is None or authorization is None:
            raise RuntimeError("claim proposal lacks a governed terminal review.")
        if decision.decision != "approved":
            raise PermissionError("only approved claim proposals may become annotations.")
        if (
            proposal.owner_id != owner
            or proposal.doc_id != document
            or proposal.generation != sequence
            or proposal.content_sha256 != content
            or proposal.profile_fingerprint != profile
        ):
            raise PermissionError("claim proposal differs from requested graph generation scope.")
        if (
            authorization.proposal_id != proposal.proposal_id
            or authorization.decision_id != decision.decision_id
            or authorization.owner_id != owner
            or authorization.doc_id != document
            or authorization.generation != sequence
            or authorization.decision != "approved"
            or authorization.reviewer_id != decision.reviewer_id
        ):
            raise RuntimeError("claim review authorization differs from approved decision.")
        if proposal.supersedes_proposal_id is not None:
            predecessor = store.get_decision(proposal.supersedes_proposal_id)
            if (
                predecessor is None
                or predecessor.decision != "superseded"
                or predecessor.replacement_proposal_id != proposal.proposal_id
            ):
                raise RuntimeError("approved claim correction lineage is incomplete.")
        successor = store.get_successor(proposal.proposal_id)
        if successor is not None:
            successor_decision = store.get_decision(successor.proposal_id)
            if successor_decision is not None and successor_decision.decision == "approved":
                raise RuntimeError("approved claim is obsolete because an approved correction exists.")
        annotations.append(
            GraphAnnotation(
                annotation_key=f"claim:{proposal.proposal_id}",
                node_type="claim",
                label=proposal.claim_text,
                text=proposal.claim_text,
                section_index=proposal.locator.section_index,
                page_number=proposal.locator.page_number,
                metadata={
                    "claim_proposal_id": proposal.proposal_id,
                    "claim_proposal_digest": proposal.proposal_digest,
                    "claim_decision_id": decision.decision_id,
                    "claim_authorization_digest": authorization.authorization_digest,
                    "claim_policy_digest": authorization.policy_digest,
                    "claim_grant_digest": authorization.grant_digest,
                    "claim_type": proposal.claim_type,
                    "claim_modality": proposal.modality,
                    "claim_confidence": proposal.confidence,
                    "evidence_sha256": proposal.locator.evidence_sha256,
                    "evidence_locator_digest": proposal.locator.locator_digest,
                    "evidence_char_start": proposal.locator.char_start,
                    "evidence_char_end": proposal.locator.char_end,
                    "extractor_name": proposal.extractor_name,
                    "extractor_version": proposal.extractor_version,
                    "supersedes_proposal_id": proposal.supersedes_proposal_id,
                    "explicit_reviewed_claim": True,
                    "semantic_relation_inference_performed": False,
                },
            )
        )
    return tuple(annotations)


__all__ = [
    "GovernedScientificClaimReviewService",
    "approved_claim_annotations",
    "load_claim_review_policy",
]
