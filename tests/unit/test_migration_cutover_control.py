from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools.migration_cutover_control import build_cutover_preparation
from tools.migration_promotion import PromotionReport
from tools.migration_rollback_artifact import (
    RollbackEncryptionKey,
    capture_rollback_payload,
)
from tools.migration_rollback_reconstruction import reconstruct_rollback_snapshots
from tools.migration_rollback_staging import verify_in_isolated_staging
from tools.migration_rollback_store import MigrationRollbackStore
from tests.unit.test_migration_rollback_artifact import aligned_preflight


def promotion(preflight):
    return PromotionReport(
        task_id=preflight.task_id,
        owner_id=preflight.owner_id,
        doc_id=preflight.doc_id,
        source_sequence=preflight.source_sequence,
        source_profile_fingerprint=preflight.source_profile_fingerprint,
        target_profile_fingerprint=preflight.target_profile_fingerprint,
        validation_digest=preflight.validation_digest,
        benchmark_fingerprint=preflight.benchmark_fingerprint,
        evidence_digest="6" * 64,
        policy_id="paired-promotion-v1",
        policy_digest="7" * 64,
        decision="eligible",
        reason_codes=(),
        quality_deltas={
            name: 0.0
            for name in (
                "recall_at_k",
                "ndcg_at_k",
                "mrr",
                "support_recall",
                "citation_precision",
                "abstention_accuracy",
            )
        },
        resource_ratios={
            "p95_latency_ms": 1.0,
            "peak_memory_bytes": 1.0,
            "index_bytes": 1.0,
            "estimated_cost_units": 1.0,
        },
        evaluated_at=1,
    )


def prerequisites(tmp_path):
    original, snapshot = aligned_preflight()
    report = promotion(original)
    preflight = replace(original, promotion_report_digest=report.report_digest)
    payload = capture_rollback_payload(preflight, snapshot)
    store = MigrationRollbackStore(tmp_path / "rollbacks")
    rollback_manifest = store.write(
        preflight=preflight,
        payload=payload,
        key=RollbackEncryptionKey("key-1", b"k" * 32),
        now=1,
    )
    reconstructed = reconstruct_rollback_snapshots(preflight, payload)
    staging = verify_in_isolated_staging(preflight, reconstructed, now=1)
    task = SimpleNamespace(
        task_id=preflight.task_id,
        owner_id=preflight.owner_id,
        doc_id=preflight.doc_id,
        source_sequence=preflight.source_sequence,
        source_profile_fingerprint=preflight.source_profile_fingerprint,
        target_profile_fingerprint=preflight.target_profile_fingerprint,
        validation_digest=preflight.validation_digest,
        state="validated",
    )
    return task, preflight, report, rollback_manifest, staging, snapshot.generation


def test_preparation_binds_all_immutable_prerequisites(tmp_path):
    values = prerequisites(tmp_path)
    first = build_cutover_preparation(
        task=values[0],
        preflight=values[1],
        promotion=values[2],
        rollback_manifest=values[3],
        staging=values[4],
        generation=values[5],
        now=1,
    )
    second = build_cutover_preparation(
        task=values[0],
        preflight=values[1],
        promotion=values[2],
        rollback_manifest=values[3],
        staging=values[4],
        generation=values[5],
        now=2,
    )
    assert first.operation_id == second.operation_id
    assert first.prepared_at != second.prepared_at
    assert first.rollback_artifact_digest == values[3].artifact_digest
    assert first.staging_verification_digest == values[4].verification_digest


def test_blocked_report_or_mismatched_staging_is_refused(tmp_path):
    values = list(prerequisites(tmp_path))
    values[2] = replace(values[2], decision="blocked", reason_codes=("blocked",))
    with pytest.raises(RuntimeError, match="eligible paired"):
        build_cutover_preparation(
            task=values[0], preflight=values[1], promotion=values[2],
            rollback_manifest=values[3], staging=values[4], generation=values[5]
        )
    values = list(prerequisites(tmp_path / "second"))
    values[4] = replace(values[4], vector_rows=values[4].vector_rows + 1)
    with pytest.raises(RuntimeError, match="inconsistent"):
        build_cutover_preparation(
            task=values[0], preflight=values[1], promotion=values[2],
            rollback_manifest=values[3], staging=values[4], generation=values[5]
        )


def test_stale_generation_is_refused(tmp_path):
    values = list(prerequisites(tmp_path))
    values[5] = SimpleNamespace(
        **{**vars(values[5]), "sequence": values[5].sequence + 1}
    )
    with pytest.raises(RuntimeError, match="changed"):
        build_cutover_preparation(
            task=values[0], preflight=values[1], promotion=values[2],
            rollback_manifest=values[3], staging=values[4], generation=values[5]
        )
