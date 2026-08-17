from __future__ import annotations

import json
import os

import pytest

from evaluation.benchmark_suite import BenchmarkRow, BenchmarkSuiteResult
from evaluation.conformal_retrieval import SelectiveRiskMetrics
from evaluation.drift import DriftReport
from evaluation.efficiency import LatencySummary
from evaluation.quality_observability import (
    MetricObservation,
    QualityProvenance,
    QualitySLO,
    QualitySnapshot,
    QualityWindow,
    build_quality_dashboard,
    compare_quality_snapshots,
    observations_from_benchmark_suite,
    observations_from_drift_report,
    observations_from_generation_metrics,
    observations_from_latency_summary,
    observations_from_retrieval_metrics,
    observations_from_selective_risk,
    observations_from_semantic_metrics,
    write_quality_dashboard,
    write_quality_snapshot,
)
from evaluation.resource_measurement import ProviderUsage, ResourceUsage
from evaluation.semantic_support import CitationSupportMetrics, SemanticMetrics


def provenance(
    *,
    run_id: str = "run-1",
    system_id: str = "rag",
    model_digest: str = "e" * 64,
    retrieval_stack_digest: str = "d" * 64,
) -> QualityProvenance:
    return QualityProvenance(
        run_id=run_id,
        system_id=system_id,
        domain_id="research",
        dataset_manifest_digest="a" * 64,
        split_digest="b" * 64,
        evaluation_contract_digest="c" * 64,
        code_revision="deadbeef",
        retrieval_stack_digest=retrieval_stack_digest,
        model_digest=model_digest,
        environment_digest="f" * 64,
    )


def snapshot(*metrics: MetricObservation, run_id: str = "run-1") -> QualitySnapshot:
    return QualitySnapshot(
        QualityWindow(10.0, 20.0, 21.0),
        provenance(run_id=run_id),
        tuple(metrics),
    )


def test_snapshot_is_canonical_order_independent_and_rejects_duplicate_identity() -> None:
    first = MetricObservation(
        "retrieval.ndcg@10",
        0.8,
        "higher",
        "ratio",
        100,
        "evaluation.retrieval",
        {"k": "10", "route": "hybrid"},
    )
    second = MetricObservation(
        "latency.p95_ms",
        50.0,
        "lower",
        "ms",
        100,
        "evaluation.efficiency",
    )
    left = snapshot(first, second)
    right = snapshot(second, first)

    assert left.snapshot_digest == right.snapshot_digest
    assert left.to_dict() == right.to_dict()
    assert [row.name for row in left.metrics] == ["latency.p95_ms", "retrieval.ndcg@10"]

    with pytest.raises(ValueError, match="duplicate metric identities"):
        snapshot(first, first)


def test_metric_contract_rejects_nonfinite_values_and_content_dimensions() -> None:
    with pytest.raises(ValueError, match="finite"):
        MetricObservation(
            "retrieval.ndcg@10",
            float("nan"),
            "higher",
            "ratio",
            1,
            "test",
        )
    with pytest.raises(ValueError, match="raw content"):
        MetricObservation(
            "retrieval.ndcg@10",
            0.5,
            "higher",
            "ratio",
            1,
            "test",
            {"query_text": "secret query"},
        )
    with pytest.raises(ValueError, match="approved non-content dimension"):
        MetricObservation(
            "retrieval.ndcg@10",
            0.5,
            "higher",
            "ratio",
            1,
            "test",
            {"arbitrary": "secret query"},
        )


def test_slo_evaluation_passes_fails_and_fails_closed_on_missing_or_ambiguous() -> None:
    metrics = (
        MetricObservation(
            "retrieval.ndcg@10",
            0.82,
            "higher",
            "ratio",
            100,
            "test",
            {"route": "hybrid"},
        ),
        MetricObservation(
            "retrieval.ndcg@10",
            0.70,
            "higher",
            "ratio",
            100,
            "test",
            {"route": "dense"},
        ),
        MetricObservation("latency.p95_ms", 95.0, "lower", "ms", 100, "test"),
    )
    snap = snapshot(*metrics)
    dashboard = build_quality_dashboard(
        snap,
        (
            QualitySLO(
                "hybrid quality",
                "retrieval.ndcg@10",
                ">=",
                0.80,
                tag_match={"route": "hybrid"},
            ),
            QualitySLO("latency", "latency.p95_ms", "<=", 100.0),
            QualitySLO("required support", "citation.supported_claim_rate", ">=", 0.90),
            QualitySLO(
                "optional cost",
                "resource.cost_units",
                "<=",
                1.0,
                required=False,
            ),
            QualitySLO("ambiguous retrieval", "retrieval.ndcg@10", ">=", 0.60),
        ),
    )
    by_name = {row.name: row for row in dashboard.slo_results}

    assert by_name["hybrid quality"].status == "passed"
    assert by_name["latency"].passed is True
    assert by_name["required support"].status == "missing"
    assert by_name["required support"].passed is False
    assert by_name["optional cost"].status == "optional_missing"
    assert by_name["optional cost"].passed is True
    assert by_name["ambiguous retrieval"].status == "ambiguous"
    assert dashboard.healthy is False
    assert set(dashboard.failed_slos) == {"required support", "ambiguous retrieval"}


def test_comparison_is_scope_bound_direction_aware_and_tracks_presence_changes() -> None:
    base = snapshot(
        MetricObservation("retrieval.ndcg@10", 0.70, "higher", "ratio", 100, "test"),
        MetricObservation("latency.p95_ms", 120.0, "lower", "ms", 100, "test"),
        MetricObservation("resource.cost_units", 0.1, "lower", "cost_units", 100, "test"),
        run_id="baseline",
    )
    current = QualitySnapshot(
        QualityWindow(30.0, 40.0, 41.0),
        provenance(
            run_id="candidate",
            model_digest="1" * 64,
            retrieval_stack_digest="2" * 64,
        ),
        (
            MetricObservation("retrieval.ndcg@10", 0.75, "higher", "ratio", 100, "test"),
            MetricObservation("latency.p95_ms", 90.0, "lower", "ms", 100, "test"),
            MetricObservation("citation.supported_claim_rate", 0.95, "higher", "ratio", 100, "test"),
        ),
    )
    comparison = compare_quality_snapshots(base, current)
    rows = {row.name: row for row in comparison.deltas}

    assert rows["retrieval.ndcg@10"].state == "improved"
    assert rows["retrieval.ndcg@10"].normalized_delta == pytest.approx(0.05)
    assert rows["latency.p95_ms"].state == "improved"
    assert rows["latency.p95_ms"].normalized_delta == pytest.approx(30.0)
    assert rows["resource.cost_units"].state == "missing"
    assert rows["citation.supported_claim_rate"].state == "new"
    assert comparison.improved_count == 2
    assert comparison.regressed_count == 0

    other_scope = QualitySnapshot(
        current.window,
        provenance(run_id="other", system_id="different"),
        current.metrics,
    )
    with pytest.raises(ValueError, match="comparison scope"):
        compare_quality_snapshots(base, other_scope)


def test_registered_adapters_cover_existing_metric_families_without_raw_records() -> None:
    retrieval = observations_from_retrieval_metrics(
        {"precision@10": 0.4, "recall@10": 0.8, "ndcg@10": 0.7},
        sample_count=50,
        tags={"route": "hybrid"},
    )
    generation = observations_from_generation_metrics(
        {"rouge_l": 0.6, "chrf": 0.7, "unsupported_claim_rate": 0.05},
        sample_count=50,
    )
    semantic = observations_from_semantic_metrics(
        SemanticMetrics(
            count=50,
            coverage=0.96,
            accuracy_on_covered=0.90,
            entailment_recall=0.91,
            contradiction_recall=0.88,
            contradiction_false_negative_rate=0.12,
            multiclass_brier=0.08,
            expected_calibration_error=0.03,
        ),
        CitationSupportMetrics(
            claim_count=80,
            citation_count=100,
            claim_coverage=0.95,
            mean_best_entailment=0.91,
            supported_claim_rate=0.93,
            contradicted_claim_rate=0.01,
            unsupported_claim_rate=0.06,
        ),
    )
    selective = observations_from_selective_risk(
        SelectiveRiskMetrics(
            total=50,
            covered=45,
            coverage=0.9,
            error_rate_on_covered=0.05,
            abstention_rate=0.1,
        )
    )
    latency = observations_from_latency_summary(
        LatencySummary(50, 10.0, 9.0, 20.0, 30.0, 4.0, 35.0)
    )
    resource = ResourceUsage(
        wall_ms=12.0,
        cpu_ms=10.0,
        python_peak_allocated_bytes=1_000,
        process_peak_rss_bytes=2_000,
        provider=ProviderUsage(prompt_tokens=100, completion_tokens=20, cost_units=0.02),
    )
    from evaluation.quality_observability import observations_from_resource_usage

    resources = observations_from_resource_usage(resource, sample_count=50)
    drift = observations_from_drift_report(
        DriftReport(
            score_psi=0.1,
            route_jsd=0.02,
            calibration_shift=0.01,
            latency_relative=-0.05,
            cost_relative=0.02,
            alerts=("cost_drift",),
        ),
        sample_count=50,
    )

    all_rows = retrieval + generation + semantic + selective + latency + resources + drift
    snap = QualitySnapshot(QualityWindow(1.0, 2.0, 3.0), provenance(), all_rows)

    names = {row.name for row in snap.metrics}
    assert {
        "retrieval.ndcg@10",
        "generation.unsupported_claim_rate",
        "semantic.expected_calibration_error",
        "citation.supported_claim_rate",
        "selective.error_rate_on_covered",
        "latency.p95_ms",
        "resource.process_peak_rss_bytes",
        "drift.score_psi",
    }.issubset(names)
    rendered = json.dumps(snap.to_dict()).lower()
    assert "secret query" not in rendered
    assert "claim_text" not in rendered
    assert "evidence_text" not in rendered


def test_benchmark_suite_adapter_exports_aggregates_not_generated_answers() -> None:
    result = BenchmarkSuiteResult(
        rows=(
            BenchmarkRow(
                example_id="example-1",
                retrieval_metrics={"ndcg@10": 0.8},
                retrieval_latency_ms=10.0,
                generated_answer="private generated answer",
                generation_latency_ms=20.0,
                generation_metrics={"rouge_l": 0.7},
            ),
        ),
        aggregate={
            "ndcg@10": 0.8,
            "rouge_l": 0.7,
            "retrieval_latency_ms": 10.0,
            "generation_latency_ms": 20.0,
        },
    )
    rows = observations_from_benchmark_suite(result)
    rendered = json.dumps([row.to_dict() for row in rows]).lower()

    assert {row.name for row in rows} == {
        "retrieval.ndcg@10",
        "generation.rouge_l",
        "benchmark_latency.retrieval_mean_ms",
        "benchmark_latency.generation_mean_ms",
    }
    assert "private generated answer" not in rendered
    assert all(row.sample_count == 1 for row in rows)


def test_atomic_exports_emit_verifiable_digests(tmp_path) -> None:
    snap = snapshot(
        MetricObservation("retrieval.ndcg@10", 0.8, "higher", "ratio", 10, "test")
    )
    dashboard = build_quality_dashboard(
        snap,
        (QualitySLO("retrieval quality", "retrieval.ndcg@10", ">=", 0.7),),
    )
    snapshot_path = tmp_path / "quality-snapshot.json"
    dashboard_path = tmp_path / "quality-dashboard.json"

    write_quality_snapshot(snapshot_path, snap)
    write_quality_dashboard(dashboard_path, dashboard)

    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    dashboard_payload = json.loads(dashboard_path.read_text(encoding="utf-8"))
    assert snapshot_payload["snapshot_digest"] == snap.snapshot_digest
    assert dashboard_payload["dashboard_digest"] == dashboard.dashboard_digest
    assert dashboard_payload["snapshot_digest"] == snap.snapshot_digest
    if os.name != "nt":
        assert snapshot_path.stat().st_mode & 0o777 == 0o600
        assert dashboard_path.stat().st_mode & 0o777 == 0o600
