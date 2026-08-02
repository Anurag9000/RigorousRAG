from __future__ import annotations

import pytest

from tools.evidence_graph_relation_policy import ReviewAuthorization
from tools.evidence_graph_relation_policy_integrity import (
    deterministic_review_authorization_digest,
    install_relation_review_authorization_integrity,
)


def values() -> dict[str, object]:
    return {
        "proposal_id": "1" * 64,
        "decision_id": "2" * 64,
        "owner_id": "alice",
        "graph_set_key": "review",
        "decision": "approved",
        "reviewer_id": "reviewer",
        "policy_digest": "3" * 64,
        "grant_digest": "4" * 64,
        "separation_of_duties_enforced": True,
        "replacement_scope_validated": False,
    }


def test_authorization_digest_is_recomputed_from_all_governed_fields():
    payload = values()
    digest = deterministic_review_authorization_digest(**payload)
    authorization = ReviewAuthorization(
        **payload,
        authorization_digest=digest,
        authorized_at=1.0,
    )
    assert authorization.authorization_digest == digest

    for field, changed in (
        ("owner_id", "bob"),
        ("graph_set_key", "other"),
        ("reviewer_id", "other-reviewer"),
        ("policy_digest", "5" * 64),
        ("grant_digest", "6" * 64),
        ("replacement_scope_validated", True),
    ):
        tampered = {**payload, field: changed}
        with pytest.raises(ValueError, match="deterministic authorization identity"):
            ReviewAuthorization(
                **tampered,
                authorization_digest=digest,
                authorized_at=1.0,
            )


def test_integrity_installation_is_idempotent():
    before = ReviewAuthorization.__post_init__
    install_relation_review_authorization_integrity()
    install_relation_review_authorization_integrity()
    assert ReviewAuthorization.__post_init__ is before
