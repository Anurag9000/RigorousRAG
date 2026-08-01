from dataclasses import replace

import pytest

from tools.migration_benchmark import (
    BenchmarkCase,
    BenchmarkRun,
    DeltaInterval,
    PromotionBenchmarkFixture,
    fixture_from_mapping,
    run_promotion_benchmark,
)
from tools.migration_promotion import ResourceMetrics

TASK = "a" * 64
VALIDATION = "b" * 64
CONTENT = "c" * 64


def resources(latency=100.0, memory=1000, storage=2000, cost=10.0):
    return ResourceMetrics(
        p95_latency_ms=latency,
        peak_memory_bytes=memory,
        index_bytes=storage,
        estimated_cost_units=cost,
    )


def case(
    query="q1",
    current=("d1", "d2"),
    shadow=("d1", "d3"),
    *,
    relevant=("d1", "d3"),
    support_total=2,
    current_support=1,
    shadow_support=2,
    should_abstain=False,
    current_abstained=False,
    shadow_abstained=False,
    current_citations=2,
    current_valid=2,
    shadow_citations=2,
    shadow_valid=2,
):
    return BenchmarkCase(
        query_id=query,
        relevant_ids=tuple(relevant),
        current_ranked_ids=tuple(current),
        shadow_ranked_ids=tuple(shadow),
        support_total=support_total,
        current_support_found=current_support,
        shadow_support_found=shadow_support,
        current_citation_count=current_citations,
        current_valid_citation_count=current_valid,
        shadow_citation_count=shadow_citations,
        shadow_valid_citation_count=shadow_valid,
        should_abstain=should_abstain,
        current_abstained=current_abstained,
        shadow_abstained=shadow_abstained,
    )


def run(seed=1, first=None, latency=100.0):
    first = first or case()
    abstain = case(
        query="q2",
        current=(),
        shadow=(),
        relevant=(),
        support_total=0,
        current_support=0,
        shadow_support=0,
        should_abstain=True,
        current_abstained=False,
        shadow_abstained=True,
        current_citations=0,
        current_valid=0,
        shadow_citations=0,
        shadow_valid=0,
    )
    return BenchmarkRun(
        seed=seed,
        cases=(first, abstain),
        current_resources=resources(latency=latency),
        shadow_resources=resources(
            latency=latency * 1.2,
            memory=1100,
            storage=2200,
            cost=11,
        ),
    )


def fixture(runs=None):
    return PromotionBenchmarkFixture(
        task_id=TASK,
        validation_digest=VALIDATION,
        source_sequence=4,
        source_content_sha256=CONTENT,
        vector_count=4,
        sparse_count=4,
        rank_cutoff=2,
        runs=tuple(runs or (run(1), run(2), run(3))),
    )


def test_benchmark_produces_paired_aggregate_promotion_evidence():
    result = run_promotion_benchmark(fixture())
    evidence = result.evidence
    assert evidence.task_id == TASK
    assert evidence.repeated_runs == 3
    assert evidence.seed_count == 3
    assert evidence.current_quality.query_count == 2
    assert evidence.shadow_quality.recall_at_k > evidence.current_quality.recall_at_k
    assert evidence.shadow_quality.support_recall > evidence.current_quality.support_recall
    assert (
        evidence.shadow_quality.abstention_accuracy
        > evidence.current_quality.abstention_accuracy
    )
    assert evidence.current_resources.p95_latency_ms == 100.0
    assert evidence.shadow_resources.p95_latency_ms == 120.0
    assert result.delta_intervals["recall_at_k"].mean > 0
    assert isinstance(result.delta_intervals["recall_at_k"], DeltaInterval)


def test_benchmark_fingerprint_identifies_contract_not_outputs_or_resources():
    base = fixture()
    changed_output = fixture(
        (
            run(1, first=case(current=("x", "y"), shadow=("d3", "d1")), latency=999),
            run(2, first=case(current=("x", "y"), shadow=("d3", "d1")), latency=999),
            run(3, first=case(current=("x", "y"), shadow=("d3", "d1")), latency=999),
        )
    )
    assert base.benchmark_fingerprint == changed_output.benchmark_fingerprint
    changed_contract = replace(base, rank_cutoff=1)
    assert base.benchmark_fingerprint != changed_contract.benchmark_fingerprint


def test_all_runs_must_share_ordered_query_contract():
    inconsistent = run(2, first=case(relevant=("d9",)))
    with pytest.raises(ValueError, match="same ordered benchmark contract"):
        fixture((run(1), inconsistent))


def test_duplicate_ranked_ids_and_invalid_counts_are_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        case(current=("d1", "d1"))
    with pytest.raises(ValueError, match="exceeds support_total"):
        case(current_support=3)
    with pytest.raises(ValueError, match="relevant identifiers"):
        case(relevant=(), should_abstain=False)


def test_no_citations_fail_closed_to_zero_precision():
    no_citations = case(
        current_citations=0,
        current_valid=0,
        shadow_citations=0,
        shadow_valid=0,
    )
    result = run_promotion_benchmark(fixture((run(1, first=no_citations),)))
    assert result.evidence.current_quality.citation_precision == 0.0
    assert result.evidence.shadow_quality.citation_precision == 0.0


def test_single_run_intervals_are_degenerate_and_bounded():
    result = run_promotion_benchmark(fixture((run(1),)))
    interval = result.current_intervals["recall_at_k"]
    assert interval.lower == interval.mean == interval.upper
    delta = result.delta_intervals["recall_at_k"]
    assert delta.lower == delta.mean == delta.upper


def test_mapping_loader_is_closed_schema_and_reconstructs_fixture():
    value = {
        "task_id": TASK,
        "validation_digest": VALIDATION,
        "source_sequence": 4,
        "source_content_sha256": CONTENT,
        "vector_count": 4,
        "sparse_count": 4,
        "rank_cutoff": 2,
        "runs": [
            {
                "seed": 1,
                "cases": [case().__dict__, run(1).cases[1].__dict__],
                "current_resources": resources().__dict__,
                "shadow_resources": resources(latency=120).__dict__,
            }
        ],
    }
    loaded = fixture_from_mapping(value)
    assert loaded.task_id == TASK
    value["raw_query"] = "private"
    with pytest.raises(ValueError, match="incomplete or unsupported"):
        fixture_from_mapping(value)


def test_duplicate_seeds_are_counted_once_but_runs_remain_distinct():
    result = run_promotion_benchmark(fixture((run(1), run(1), run(2))))
    assert result.evidence.repeated_runs == 3
    assert result.evidence.seed_count == 2
