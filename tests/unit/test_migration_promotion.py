from types import SimpleNamespace

import pytest

from tools.migration_promotion import (
    PromotionEvidence,
    PromotionPolicy,
    ResourceMetrics,
    RetrievalMetrics,
    evaluate_promotion,
    evidence_from_mapping,
)

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64


def quality(**changes):
    values = dict(
        query_count=100,
        recall_at_k=0.8,
        ndcg_at_k=0.75,
        mrr=0.72,
        support_recall=0.82,
        citation_precision=0.96,
        abstention_accuracy=0.9,
    )
    values.update(changes)
    return RetrievalMetrics(**values)


def resources(**changes):
    values = dict(
        p95_latency_ms=100.0,
        peak_memory_bytes=1000,
        index_bytes=2000,
        estimated_cost_units=10.0,
    )
    values.update(changes)
    return ResourceMetrics(**values)


def evidence(**changes):
    values = dict(
        task_id=E,
        validation_digest=D,
        benchmark_fingerprint=B,
        source_sequence=4,
        source_content_sha256=C,
        vector_count=3,
        sparse_count=3,
        repeated_runs=5,
        seed_count=5,
        confidence_interval_level=0.95,
        current_quality=quality(),
        shadow_quality=quality(recall_at_k=0.81),
        current_resources=resources(),
        shadow_resources=resources(p95_latency_ms=120.0),
    )
    values.update(changes)
    return PromotionEvidence(**values)


def fixtures(state="validated"):
    task = SimpleNamespace(
        task_id=E,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=A,
        target_profile_fingerprint=B,
        validation_digest=D,
        state=state,
    )
    manifest = SimpleNamespace(
        task_id=E,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=A,
        target_profile_fingerprint=B,
        validation_digest=D,
        content_sha256=C,
        vector_count=3,
        sparse_count=3,
    )
    generation = SimpleNamespace(
        sequence=4,
        state="active",
        profile_fingerprint=A,
        content_sha256=C,
    )
    return task, manifest, generation


def test_eligible_report_is_deterministic_except_timestamp():
    task, manifest, generation = fixtures()
    first = evaluate_promotion(
        task=task,
        manifest=manifest,
        generation=generation,
        evidence=evidence(),
        now=10,
    )
    second = evaluate_promotion(
        task=task,
        manifest=manifest,
        generation=generation,
        evidence=evidence(),
        now=20,
    )
    assert first.decision == "eligible"
    assert first.reason_codes == ()
    assert first.report_digest == second.report_digest
    assert first.evaluated_at != second.evaluated_at
    assert first.quality_deltas["recall_at_k"] == pytest.approx(0.01)
    assert first.resource_ratios["p95_latency_ms"] == pytest.approx(1.2)


def test_quality_floor_regression_and_resource_limit_block():
    task, manifest, generation = fixtures()
    evaluated = evidence(
        shadow_quality=quality(recall_at_k=0.5, citation_precision=0.8),
        shadow_resources=resources(p95_latency_ms=200.0, index_bytes=5000),
    )
    report = evaluate_promotion(
        task=task,
        manifest=manifest,
        generation=generation,
        evidence=evaluated,
        now=1,
    )
    assert report.decision == "blocked"
    assert "recall_at_k_below_floor" in report.reason_codes
    assert "recall_at_k_regression_exceeds_limit" in report.reason_codes
    assert "citation_precision_below_floor" in report.reason_codes
    assert "p95_latency_ms_ratio_exceeds_limit" in report.reason_codes
    assert "index_bytes_ratio_exceeds_limit" in report.reason_codes
    assert report.reason_codes == tuple(sorted(report.reason_codes))


def test_generation_and_evidence_alignment_are_mandatory():
    task, manifest, generation = fixtures()
    generation.sequence = 5
    report = evaluate_promotion(
        task=task,
        manifest=manifest,
        generation=generation,
        evidence=evidence(source_sequence=5, vector_count=2),
        now=1,
    )
    assert "source_generation_sequence_changed" in report.reason_codes
    assert "evidence_source_sequence_mismatch" in report.reason_codes
    assert "evidence_vector_count_mismatch" in report.reason_codes
    assert "vector_sparse_count_mismatch" in report.reason_codes


def test_zero_resource_baseline_fails_closed_when_shadow_nonzero():
    task, manifest, generation = fixtures()
    evaluated = evidence(
        current_resources=resources(
            p95_latency_ms=0,
            peak_memory_bytes=0,
            index_bytes=0,
            estimated_cost_units=0,
        ),
        shadow_resources=resources(),
    )
    report = evaluate_promotion(
        task=task,
        manifest=manifest,
        generation=generation,
        evidence=evaluated,
        now=1,
    )
    assert report.decision == "blocked"
    assert "p95_latency_ms_has_zero_baseline" in report.reason_codes
    assert "peak_memory_bytes_has_zero_baseline" in report.reason_codes


def test_benchmark_minimums_and_confidence_are_enforced():
    task, manifest, generation = fixtures()
    evaluated = evidence(
        repeated_runs=1,
        seed_count=1,
        confidence_interval_level=0.8,
        current_quality=quality(query_count=10),
        shadow_quality=quality(query_count=10),
    )
    report = evaluate_promotion(
        task=task,
        manifest=manifest,
        generation=generation,
        evidence=evaluated,
        now=1,
    )
    assert "benchmark_query_count_below_minimum" in report.reason_codes
    assert "benchmark_repeated_runs_below_minimum" in report.reason_codes
    assert "benchmark_seed_count_below_minimum" in report.reason_codes
    assert "benchmark_confidence_level_below_minimum" in report.reason_codes


def test_unvalidated_or_mismatched_identity_is_rejected_before_report():
    task, manifest, generation = fixtures(state="running")
    with pytest.raises(ValueError, match="validated"):
        evaluate_promotion(
            task=task,
            manifest=manifest,
            generation=generation,
            evidence=evidence(),
            now=1,
        )
    task.state = "validated"
    manifest.doc_id = "other"
    with pytest.raises(RuntimeError, match="do not match"):
        evaluate_promotion(
            task=task,
            manifest=manifest,
            generation=generation,
            evidence=evidence(),
            now=1,
        )


def test_mapping_loader_is_closed_schema():
    evaluated = evidence()
    payload = {
        "task_id": evaluated.task_id,
        "validation_digest": evaluated.validation_digest,
        "benchmark_fingerprint": evaluated.benchmark_fingerprint,
        "source_sequence": evaluated.source_sequence,
        "source_content_sha256": evaluated.source_content_sha256,
        "vector_count": 3,
        "sparse_count": 3,
        "repeated_runs": 5,
        "seed_count": 5,
        "confidence_interval_level": 0.95,
        "current_quality": evaluated.current_quality.__dict__,
        "shadow_quality": evaluated.shadow_quality.__dict__,
        "current_resources": evaluated.current_resources.__dict__,
        "shadow_resources": evaluated.shadow_resources.__dict__,
    }
    assert evidence_from_mapping(payload).evidence_digest == evaluated.evidence_digest
    payload["extra"] = 1
    with pytest.raises(ValueError, match="incomplete or unsupported"):
        evidence_from_mapping(payload)


def test_policy_refuses_nonpositive_ratios_and_invalid_probability():
    with pytest.raises(ValueError):
        PromotionPolicy(max_latency_ratio=0)
    with pytest.raises(ValueError):
        RetrievalMetrics(
            query_count=1,
            recall_at_k=1.1,
            ndcg_at_k=0,
            mrr=0,
            support_recall=0,
            citation_precision=0,
            abstention_accuracy=0,
        )
