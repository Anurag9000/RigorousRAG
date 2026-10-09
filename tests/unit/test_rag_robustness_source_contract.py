"""Digest-bound RAG robustness and defensive release-gate tests.

All cases are controlled, synthetic measurements. These are not adversarial
payloads, real retrieval benchmarks, or production security certifications.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from evaluation import rag_robustness as robustness


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _binding(name="case-a", *, attack=robustness.RagAttackKind.CORPUS_POISONING):
    return robustness.RobustnessCaseBinding(
        case_id=name,
        attack_kind=attack,
        clean_query_sha256=_sha("clean-query-" + name),
        attacked_query_sha256=_sha("attacked-query-" + name),
        clean_corpus_sha256=_sha("clean-corpus"),
        attacked_corpus_sha256=_sha("attacked-corpus"),
        benchmark_manifest_sha256=_sha("benchmark-manifest"),
        split_sha256=_sha("heldout"),
    )


def _observation(
    name="case-a",
    *,
    attack=robustness.RagAttackKind.CORPUS_POISONING,
    compromised=False,
    abstained=False,
    supported=True,
    clean_abstained=False,
    cited=False,
    clean_support=0.9,
    attacked_support=0.9,
    clean_contradiction=0.1,
    attacked_contradiction=0.1,
):
    return robustness.MatchedRobustnessObservation(
        binding=_binding(name, attack=attack),
        clean_retrieval_success=True,
        attacked_retrieval_success=True,
        clean_answer_supported=True,
        attacked_answer_supported=supported,
        clean_abstained=clean_abstained,
        attacked_abstained=abstained,
        attack_target_retrieved=cited,
        attack_target_cited=cited,
        attack_changed_answer=compromised,
        clean_support_score=clean_support,
        attacked_support_score=attacked_support,
        clean_contradiction_score=clean_contradiction,
        attacked_contradiction_score=attacked_contradiction,
        attacked_topk_suspicious_fraction=0.1,
        attacked_duplicate_cluster_fraction=0.1,
        attacked_largest_source_fraction=0.4,
        attacked_independent_support_groups=3,
    )


def _signal(i: int, *, group=None, cluster=None, trust=0.9,
            injection=0.1, contradiction=0.1, integrity=1.0):
    return robustness.CandidateSecuritySignal(
        candidate_sha256=_sha(f"candidate-{i}"),
        content_sha256=_sha(f"content-{i}"),
        source_group_sha256=_sha(group or f"group-{i}"),
        duplicate_cluster_sha256=_sha(cluster or f"cluster-{i}"),
        source_trust=trust,
        injection_risk=injection,
        contradiction_risk=contradiction,
        provenance_integrity=integrity,
    )


def test_digest_bound_matched_report_is_deterministic_and_promotion_eligible():
    observations = (_observation("one"), _observation("two", attack=robustness.RagAttackKind.CITATION_SPOOFING))
    first = robustness.build_robustness_report(observations)
    second = robustness.build_robustness_report(tuple(reversed(observations)))
    assert first == second
    assert first.count == 2
    assert first.attacked_supported_answer_rate == 1
    assert first.support_retention == pytest.approx(1)
    assert tuple(first.per_attack) == ("citation_spoofing", "corpus_poisoning")
    approved = robustness.evaluate_robustness_promotion(first, robustness.RobustnessPromotionPolicy())
    assert approved.eligible and not approved.reasons
    assert len(approved.decision_sha256) == 64


def test_attack_success_support_loss_and_abstention_are_not_conflated():
    cases = (
        _observation("compromised", compromised=True, supported=False, clean_support=0.9, attacked_support=0.1),
        _observation("safe-abstained", compromised=True, supported=False, abstained=True),
    )
    report = robustness.build_robustness_report(cases)
    assert report.answer_attack_success_rate == pytest.approx(0.5)
    assert report.attacked_abstention_rate == pytest.approx(0.5)
    assert report.support_retention < 1
    assert report.robust_or_abstain_rate == pytest.approx(0.5)


def test_clean_zero_support_retention_is_not_falsely_degraded():
    report = robustness.build_robustness_report(
        (_observation(clean_support=0.0, attacked_support=0.0),)
    )
    assert report.support_retention == 1.0


def test_report_refuses_duplicate_or_empty_case_sets():
    with pytest.raises(ValueError, match="non-empty"):
        robustness.build_robustness_report(())
    duplicate = _observation()
    with pytest.raises(ValueError, match="unique"):
        robustness.build_robustness_report((duplicate, duplicate))
    with pytest.raises(ValueError, match="matched-observation"):
        robustness.build_robustness_report(("not an observation",))


def test_digest_bound_report_rejects_tampered_metrics_and_case_provenance():
    report = robustness.build_robustness_report((_observation(),))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(report, attacked_supported_answer_rate=0)
    original = _binding()
    with pytest.raises(ValueError, match="SHA-256"):
        replace(original, clean_query_sha256="not-a-digest")
    assert original.case_sha256 == _binding().case_sha256


def test_poisoning_signal_policy_yields_allow_only_for_diverse_trusted_evidence():
    policy = robustness.PoisoningDefensePolicy()
    candidates = tuple(_signal(i) for i in range(4))
    first = robustness.assess_poisoning_risk(candidates, policy=policy)
    second = robustness.assess_poisoning_risk(tuple(reversed(candidates)), policy=policy)
    assert first == second
    assert first.decision == robustness.RobustnessDecision.ALLOW
    assert first.independent_source_groups == 4
    assert not first.reasons
    assert first.largest_source_fraction == pytest.approx(0.25)


def test_poisoning_integrity_failure_blocks_even_with_otherwise_diverse_sources():
    policy = robustness.PoisoningDefensePolicy()
    items = (_signal(0), _signal(1, integrity=0.1), _signal(2), _signal(3))
    result = robustness.assess_poisoning_risk(items, policy=policy)
    assert result.decision == robustness.RobustnessDecision.BLOCK
    assert "provenance_integrity_failure" in result.reasons
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(result, integrity_failure_fraction=0)


def test_insufficient_independent_source_groups_is_blocking():
    policy = robustness.PoisoningDefensePolicy(minimum_independent_source_groups=3)
    items = (_signal(0, group="same"), _signal(1, group="same"))
    result = robustness.assess_poisoning_risk(items, policy=policy)
    assert result.decision == robustness.RobustnessDecision.BLOCK
    assert "insufficient_independent_sources" in result.reasons


def test_concentration_injection_and_low_trust_trigger_review():
    policy = robustness.PoisoningDefensePolicy()
    items = tuple(_signal(i, group="same", cluster="same", trust=0.1,
                          injection=0.95, contradiction=0.95)
                  for i in range(3))
    result = robustness.assess_poisoning_risk(items, policy=policy)
    assert result.decision == robustness.RobustnessDecision.REVIEW
    assert "duplicate_cluster_concentration" in result.reasons
    assert "single_source_concentration" in result.reasons
    assert "low_source_trust" in result.reasons
    assert "retrieved_instruction_risk" in result.reasons
    assert "contradiction_risk" in result.reasons


def test_risk_policy_switch_can_exclude_integrity_block_but_remain_review():
    policy = robustness.PoisoningDefensePolicy(block_on_integrity_failure=False)
    items = (_signal(0, integrity=0.1), _signal(1), _signal(2))
    result = robustness.assess_poisoning_risk(items, policy=policy)
    assert result.decision == robustness.RobustnessDecision.ALLOW
    assert result.integrity_failure_fraction > 0
    assert "provenance_integrity_failure" not in result.reasons


def test_all_robustness_promotion_failures_are_enforced():
    observation = _observation(
        compromised=True, supported=False, cited=True, clean_abstained=True,
        clean_support=1.0, attacked_support=0.1,
        clean_contradiction=0.0, attacked_contradiction=1.0,
    )
    report = robustness.build_robustness_report((observation,))
    policy = robustness.RobustnessPromotionPolicy()
    decision = robustness.evaluate_robustness_promotion(report, policy)
    assert not decision.eligible
    assert set(decision.reasons) == {
        "answer_attack_success_rate", "attack_target_citation_rate",
        "robust_or_abstain_rate", "support_retention",
        "clean_abstention_rate", "contradiction_increase",
    }
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(decision, eligible=True)


@pytest.mark.parametrize("field,bad", (
    ("source_trust", -0.1),
    ("injection_risk", 1.1),
    ("contradiction_risk", float("nan")),
    ("provenance_integrity", True),
))
def test_security_signals_refuse_out_of_range_or_nonfinite_risks(field, bad):
    with pytest.raises(ValueError):
        replace(_signal(0), **{field: bad})


@pytest.mark.parametrize("field,bad", (
    ("maximum_largest_source_fraction", -0.1),
    ("maximum_duplicate_cluster_fraction", 1.2),
    ("minimum_independent_source_groups", 0),
    ("block_on_integrity_failure", 1),
))
def test_security_policy_admission_is_bounded(field, bad):
    with pytest.raises(ValueError):
        replace(robustness.PoisoningDefensePolicy(), **{field: bad})


def test_poisoning_assessment_rejects_untrusted_input_shapes_and_duplicates():
    with pytest.raises(ValueError):
        robustness.assess_poisoning_risk((), policy=robustness.PoisoningDefensePolicy())
    with pytest.raises(ValueError, match="unique"):
        robustness.assess_poisoning_risk(
            (_signal(0), _signal(0)), policy=robustness.PoisoningDefensePolicy()
        )
    with pytest.raises(ValueError, match="policy"):
        robustness.assess_poisoning_risk((_signal(0),), policy="allow")


def test_promotion_policy_rejects_nonfinite_threshold_and_wrong_types():
    with pytest.raises(ValueError):
        robustness.RobustnessPromotionPolicy(maximum_contradiction_increase=-0.1)
    with pytest.raises(ValueError, match="invalid types"):
        robustness.evaluate_robustness_promotion("not-a-report", robustness.RobustnessPromotionPolicy())
