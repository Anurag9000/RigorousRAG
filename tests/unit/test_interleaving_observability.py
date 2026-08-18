from __future__ import annotations

import hashlib

from evaluation.interleaving_observability import observations_from_interleaving_promotion
from evaluation.quality_observability import QualityProvenance, QualitySLO, QualitySnapshot, QualityWindow, build_quality_dashboard
from evaluation.retrieval_interleaving import InterleavingOutcome, InterleavingSpec, RankedIdentity, build_team_draft_interleaving
from evaluation.retrieval_interleaving_promotion import InterleavingPromotionPolicy, build_interleaving_evidence, qualify_interleaving_experiment


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_interleaving_promotion_metrics_are_dashboard_compatible_and_content_safe() -> None:
    spec = InterleavingSpec(sha("experiment"), sha("baseline"), sha("candidate"), max_positions=4)
    impressions = []
    outcomes = []
    for index in range(8):
        ranking_a = tuple(RankedIdentity(f"a-{index}-{row}", f"sa-{index}-{row}") for row in range(4))
        ranking_b = tuple(RankedIdentity(f"b-{index}-{row}", f"sb-{index}-{row}") for row in range(4))
        impression = build_team_draft_interleaving(spec, query_sha256=sha(f"query-{index}"), impression_index=index, ranking_a=ranking_a, ranking_b=ranking_b)
        b_position = next(item.position for item in impression.items if item.contributed_by == "b")
        impressions.append(impression)
        outcomes.append(InterleavingOutcome.build(impression, (b_position,)))
    evidence = build_interleaving_evidence(spec, tuple(impressions), tuple(outcomes))
    receipt = qualify_interleaving_experiment(
        spec,
        evidence,
        policy=InterleavingPromotionPolicy(candidate_team="b", min_impressions=8, min_decisive=8, min_candidate_preference_rate=0.5, min_candidate_wilson_low=0.0, max_sign_test_p_value=1.0, max_tie_fraction=0.0),
    )
    metrics = observations_from_interleaving_promotion(receipt)
    assert {item.name for item in metrics} >= {"retrieval.interleaving.preference_rate", "retrieval.interleaving.sign_test_p_value", "retrieval.interleaving.eligible"}
    for item in metrics:
        tags = dict(item.tags)
        assert tags["policy_digest"] == receipt.candidate_policy_sha256
        assert "query" not in tags and "document" not in tags and "source" not in tags

    snapshot = QualitySnapshot(
        QualityWindow(1.0, 2.0, 3.0),
        QualityProvenance("interleave-run", "rigorousrag", "science", sha("dataset"), sha("split"), sha("evaluation"), "0123456789abcdef0123456789abcdef01234567", retrieval_stack_digest=sha("stack")),
        metrics,
    )
    dashboard = build_quality_dashboard(snapshot, (QualitySLO("interleaving eligible", "retrieval.interleaving.eligible", ">=", 1.0, tag_match=(("metric_family", "retrieval_interleaving"),)),))
    assert dashboard.healthy
