from dataclasses import replace

import pytest

from tools.migration_benchmark import (
    BenchmarkCase,
    BenchmarkRun,
    PromotionBenchmarkFixture,
    run_promotion_benchmark,
)
from tools.migration_promotion import PromotionReport, ResourceMetrics
from tools.migration_statistical_gate import (
    StatisticalGatePolicy,
    attach_statistical_assessment,
    evaluate_statistical_gate,
    statistical_policy_from_mapping,
)

TASK = "a" * 64
VALIDATION = "b" * 64
CONTENT = "c" * 64
SOURCE_PROFILE = "d" * 64
TARGET_PROFILE = "e" * 64
POLICY = "f" * 64


def resources():
    return ResourceMetrics(
        p95_latency_ms=100.0,
        peak_memory_bytes=1000,
        index_bytes=2000,
        estimated_cost_units=10.0,
    )


def relevant_case(*, degraded=False):
    return BenchmarkCase(
        query_id="q1",
        relevant_ids=("d1", "d3"),
        current_ranked_ids=("d1", "d3"),
        shadow_ranked_ids=("x", "y") if degraded else ("d1", "d3"),
        support_total=2,
        current_support_found=2,
        shadow_support_found=0 if degraded else 2,
        current_citation_count=2,
        current_valid_citation_count=2,
        shadow_citation_count=2,
        shadow_valid_citation_count=0 if degraded else 2,
        should_abstain=False,
        current_abstained=False,
        shadow_abstained=False,
    )


def abstention_case():
    return BenchmarkCase(
        query_id="q2",
        relevant_ids=(),
        current_ranked_ids=(),
        shadow_ranked_ids=(),
        support_total=0,
        current_support_found=0,
        shadow_support_found=0,
        current_citation_count=0,
        current_valid_citation_count=0,
        shadow_citation_count=0,
        shadow_valid_citation_count=0,
        should_abstain=True,
        current_abstained=True,
        shadow_abstained=True,
    )


def benchmark(*, degraded=False, run_count=3):
    runs = tuple(
        BenchmarkRun(
            seed=seed,
            cases=(relevant_case(degraded=degraded), abstention_case()),
            current_resources=resources(),
            shadow_resources=resources(),
        )
        for seed in range(1, run_count + 1)
    )
    fixture = PromotionBenchmarkFixture(
        task_id=TASK,
        validation_digest=VALIDATION,
        source_sequence=4,
        source_content_sha256=CONTENT,
        vector_count=4,
        sparse_count=4,
        rank_cutoff=2,
        runs=runs,
    )
    return run_promotion_benchmark(fixture)


def report(evidence):
    return PromotionReport(
        task_id=evidence.task_id,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=evidence.source_sequence,
        source_profile_fingerprint=SOURCE_PROFILE,
        target_profile_fingerprint=TARGET_PROFILE,
        validation_digest=evidence.validation_digest,
        benchmark_fingerprint=evidence.benchmark_fingerprint,
        evidence_digest=evidence.evidence_digest,
        policy_id="policy-v1",
        policy_digest=POLICY,
        decision="eligible",
        reason_codes=(),
        quality_deltas={
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "mrr": 0.0,
            "support_recall": 0.0,
            "citation_precision": 0.0,
            "abstention_accuracy": 0.0,
        },
        resource_ratios={
            "p95_latency_ms": 1.0,
            "peak_memory_bytes": 1.0,
            "index_bytes": 1.0,
            "estimated_cost_units": 1.0,
        },
        evaluated_at=1.0,
    )


def test_paired_nondegradation_passes_and_attaches_deterministically():
    result = benchmark()
    assessment = evaluate_statistical_gate(result, now=1)
    assert assessment.decision == "passed"
    assert assessment.reason_codes == ()
    attached = attach_statistical_assessment(report(result.evidence), assessment)
    assert attached.decision == "eligible"
    assert attached.policy_id == "paired-promotion-v1"
    assert attached.evidence_digest != result.evidence.evidence_digest
    assert attached.policy_digest != POLICY


def test_lower_bound_beyond_margin_blocks_report():
    result = benchmark(degraded=True)
    assessment = evaluate_statistical_gate(result, now=1)
    assert assessment.decision == "blocked"
    assert "paired_recall_at_k_noninferiority_failed" in assessment.reason_codes
    attached = attach_statistical_assessment(report(result.evidence), assessment)
    assert attached.decision == "blocked"
    assert set(assessment.reason_codes).issubset(attached.reason_codes)


def test_configured_practical_gain_uses_interval_lower_bound():
    result = benchmark()
    assessment = evaluate_statistical_gate(
        result,
        policy=StatisticalGatePolicy(minimum_recall_gain=0.1),
        now=1,
    )
    assert "paired_recall_at_k_practical_gain_failed" in assessment.reason_codes
    metric = assessment.metrics["recall_at_k"]
    assert metric.practical_gain_threshold == 0.1
    assert metric.practical_gain_satisfied is False


def test_run_seed_and_confidence_minimums_block():
    assessment = evaluate_statistical_gate(benchmark(run_count=1), now=1)
    assert "paired_runs_below_minimum" in assessment.reason_codes
    assert "paired_seed_count_below_minimum" in assessment.reason_codes


def test_assessment_report_identity_mismatch_is_refused():
    result = benchmark()
    assessment = evaluate_statistical_gate(result, now=1)
    mismatched = replace(report(result.evidence), benchmark_fingerprint=CONTENT)
    with pytest.raises(RuntimeError, match="does not match"):
        attach_statistical_assessment(mismatched, assessment)


def test_policy_mapping_is_closed_schema():
    policy = statistical_policy_from_mapping({"minimum_recall_gain": 0.01})
    assert policy.minimum_recall_gain == 0.01
    with pytest.raises(ValueError, match="unsupported"):
        statistical_policy_from_mapping({"unknown": 1})


def test_composite_report_digests_commit_to_statistical_assessment():
    result = benchmark()
    assessment = evaluate_statistical_gate(result, now=1)
    first = attach_statistical_assessment(report(result.evidence), assessment)
    second = attach_statistical_assessment(report(result.evidence), assessment)
    assert first.report_digest == second.report_digest
    changed = evaluate_statistical_gate(
        result,
        policy=StatisticalGatePolicy(minimum_recall_gain=0.1),
        now=1,
    )
    third = attach_statistical_assessment(report(result.evidence), changed)
    assert third.evidence_digest != first.evidence_digest
    assert third.policy_digest != first.policy_digest
