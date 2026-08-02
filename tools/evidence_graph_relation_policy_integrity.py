"""Deterministic identity boundary for semantic relation authorizations."""

from __future__ import annotations

from typing import Any

from tools import evidence_graph_relation_policy as _policy

_MARKER = "_deterministic_review_authorization_integrity_installed"


def deterministic_review_authorization_digest(
    *,
    proposal_id: str,
    decision_id: str,
    owner_id: str,
    graph_set_key: str,
    decision: str,
    reviewer_id: str,
    policy_digest: str,
    grant_digest: str,
    separation_of_duties_enforced: bool = True,
    replacement_scope_validated: bool = False,
) -> str:
    return _policy._sha256(
        {
            "scope": "rigorousrag-relation-review-authorization-v1",
            "proposal_id": proposal_id,
            "decision_id": decision_id,
            "owner_id": owner_id,
            "graph_set_key": graph_set_key,
            "decision": decision,
            "reviewer_id": reviewer_id,
            "policy_digest": policy_digest,
            "grant_digest": grant_digest,
            "separation_of_duties_enforced": separation_of_duties_enforced,
            "replacement_scope_validated": replacement_scope_validated,
        }
    )


def install_relation_review_authorization_integrity() -> None:
    authorization_class = _policy.ReviewAuthorization
    service_class = _policy.GovernedRelationReviewService
    if getattr(authorization_class, _MARKER, False):
        return
    original_post_init = authorization_class.__post_init__

    def checked_post_init(self: Any) -> None:
        original_post_init(self)
        expected = deterministic_review_authorization_digest(
            proposal_id=self.proposal_id,
            decision_id=self.decision_id,
            owner_id=self.owner_id,
            graph_set_key=self.graph_set_key,
            decision=self.decision,
            reviewer_id=self.reviewer_id,
            policy_digest=self.policy_digest,
            grant_digest=self.grant_digest,
            separation_of_duties_enforced=self.separation_of_duties_enforced,
            replacement_scope_validated=self.replacement_scope_validated,
        )
        if self.authorization_digest != expected:
            raise ValueError(
                "authorization_digest does not match deterministic authorization identity."
            )

    def create_authorization(
        self: Any,
        proposal: Any,
        decision: Any,
        *,
        now: float,
    ) -> Any:
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

        digest = deterministic_review_authorization_digest(
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            owner_id=proposal.owner_id,
            graph_set_key=proposal.graph_set_key,
            decision=decision.decision,
            reviewer_id=decision.reviewer_id,
            policy_digest=self.policy.policy_digest,
            grant_digest=grant.grant_digest,
            separation_of_duties_enforced=True,
            replacement_scope_validated=replacement_scope_validated,
        )
        return authorization_class(
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            owner_id=proposal.owner_id,
            graph_set_key=proposal.graph_set_key,
            decision=decision.decision,
            reviewer_id=decision.reviewer_id,
            policy_digest=self.policy.policy_digest,
            grant_digest=grant.grant_digest,
            authorization_digest=digest,
            authorized_at=now,
            replacement_scope_validated=replacement_scope_validated,
        )

    authorization_class.__post_init__ = checked_post_init
    service_class._authorization = create_authorization
    setattr(authorization_class, _MARKER, True)


install_relation_review_authorization_integrity()

__all__ = [
    "deterministic_review_authorization_digest",
    "install_relation_review_authorization_integrity",
]
