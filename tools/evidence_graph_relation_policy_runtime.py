"""Runtime loader for fail-closed evidence-graph reviewer policy."""

from __future__ import annotations

from tools.evidence_graph_relation_policy import (
    RelationReviewPolicy,
    load_relation_review_policy,
)


def get_relation_review_policy() -> RelationReviewPolicy:
    """Load current policy on every decision so revocation is immediately effective."""

    return load_relation_review_policy()


__all__ = ["get_relation_review_policy"]
