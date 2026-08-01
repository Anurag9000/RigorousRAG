from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools.migration_cutover_preflight import build_cutover_preflight
from tools.migration_promotion import PromotionReport

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64
F = "f" * 64


def report(decision="eligible", policy_id="paired-promotion-v1"):
    return PromotionReport(
        task_id=E,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=A,
        target_profile_fingerprint=B,
        validation_digest=D,
        benchmark_fingerprint=F,
        evidence_digest="1" * 64,
        policy_id=policy_id,
        policy_digest="2" * 64,
        decision=decision,
        reason_codes=() if decision == "eligible" else ("blocked",),
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


def task(state="validated"):
    return SimpleNamespace(
        task_id=E,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=A,
        target_profile_fingerprint=B,
        validation_digest=D,
        state=state,
    )


def manifest():
    return SimpleNamespace(
        task_id=E,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=A,
        target_profile_fingerprint=B,
        validation_digest=D,
        content_sha256=C,
        vector_count=2,
        sparse_count=2,
        vector_sha256="3" * 64,
        sparse_sha256="4" * 64,
    )


def snapshot():
    vector = SimpleNamespace(
        owner_id="alice",
        doc_id="doc-1",
        ids=("v1", "v2"),
        documents=("one", "two"),
        metadatas=(
            {"owner_id": "alice", "doc_id": "doc-1", "page": 1},
            {"owner_id": "alice", "doc_id": "doc-1", "page": 2},
        ),
    )
    fields = (
        SimpleNamespace(
            field_id="f1",
            field_type="body",
            text="one",
            position=0,
            token_count=1,
            page_number=1,
            section="A",
            metadata={},
        ),
        SimpleNamespace(
            field_id="f2",
            field_type="body",
            text="two",
            position=1,
            token_count=1,
            page_number=2,
            section="B",
            metadata={},
        ),
    )
    sparse = SimpleNamespace(
        owner_id="alice",
        doc_id="doc-1",
        generation=7,
        profile_fingerprint=A,
        metadata={"content_sha256": C},
        fields=fields,
    )
    generation = SimpleNamespace(
        owner_id="alice",
        doc_id="doc-1",
        sequence=4,
        state="active",
        content_sha256=C,
        profile_fingerprint=A,
        vector_rows=2,
        sparse_generation=7,
    )
    return SimpleNamespace(
        owner_id="alice",
        doc_id="doc-1",
        stores=SimpleNamespace(vector=vector, sparse=sparse),
        generation=generation,
    )


def test_preflight_binds_source_rollback_target_and_promotion_identities():
    result = build_cutover_preflight(
        task=task(),
        shadow_manifest=manifest(),
        promotion_report=report(),
        authoritative_snapshot=snapshot(),
        now=10,
    )
    assert result.source_vector_rows == 2
    assert result.source_sparse_fields == 2
    assert result.target_vector_rows == 2
    assert result.promotion_report_digest == report().report_digest
    assert len(result.rollback_identity_digest) == 64
    assert len(result.target_artifact_digest) == 64
    assert len(result.preflight_digest) == 64


def test_preflight_digest_excludes_creation_time_only():
    first = build_cutover_preflight(
        task=task(),
        shadow_manifest=manifest(),
        promotion_report=report(),
        authoritative_snapshot=snapshot(),
        now=1,
    )
    second = build_cutover_preflight(
        task=task(),
        shadow_manifest=manifest(),
        promotion_report=report(),
        authoritative_snapshot=snapshot(),
        now=2,
    )
    assert first.preflight_digest == second.preflight_digest
    assert first.created_at != second.created_at


def test_blocked_or_aggregate_only_report_is_refused():
    with pytest.raises(RuntimeError, match="eligible"):
        build_cutover_preflight(
            task=task(),
            shadow_manifest=manifest(),
            promotion_report=report(decision="blocked"),
            authoritative_snapshot=snapshot(),
        )
    with pytest.raises(RuntimeError, match="paired statistical"):
        build_cutover_preflight(
            task=task(),
            shadow_manifest=manifest(),
            promotion_report=report(policy_id="conservative-v1"),
            authoritative_snapshot=snapshot(),
        )


def test_source_generation_or_shadow_content_change_is_refused():
    changed = snapshot()
    changed.generation.sequence = 5
    with pytest.raises(RuntimeError, match="generation changed"):
        build_cutover_preflight(
            task=task(),
            shadow_manifest=manifest(),
            promotion_report=report(),
            authoritative_snapshot=changed,
        )
    bad_manifest = manifest()
    bad_manifest.content_sha256 = "9" * 64
    with pytest.raises(RuntimeError, match="content"):
        build_cutover_preflight(
            task=task(),
            shadow_manifest=bad_manifest,
            promotion_report=report(),
            authoritative_snapshot=snapshot(),
        )


def test_vector_and_sparse_snapshot_scope_and_counts_are_exact():
    changed = snapshot()
    changed.stores.vector.metadatas = (
        {"owner_id": "bob", "doc_id": "doc-1"},
        {"owner_id": "alice", "doc_id": "doc-1"},
    )
    with pytest.raises(RuntimeError, match="escaped task scope"):
        build_cutover_preflight(
            task=task(),
            shadow_manifest=manifest(),
            promotion_report=report(),
            authoritative_snapshot=changed,
        )
    changed = snapshot()
    changed.generation.vector_rows = 3
    with pytest.raises(RuntimeError, match="row count"):
        build_cutover_preflight(
            task=task(),
            shadow_manifest=manifest(),
            promotion_report=report(),
            authoritative_snapshot=changed,
        )
    changed = snapshot()
    changed.stores.sparse.generation = 8
    with pytest.raises(RuntimeError, match="sparse rollback generation"):
        build_cutover_preflight(
            task=task(),
            shadow_manifest=manifest(),
            promotion_report=report(),
            authoritative_snapshot=changed,
        )


def test_task_shadow_report_identity_mismatch_is_refused():
    mismatched = replace(report(), benchmark_fingerprint="8" * 64)
    mismatched_manifest = manifest()
    mismatched_manifest.task_id = "7" * 64
    with pytest.raises(RuntimeError, match="identities"):
        build_cutover_preflight(
            task=task(),
            shadow_manifest=mismatched_manifest,
            promotion_report=mismatched,
            authoritative_snapshot=snapshot(),
        )
