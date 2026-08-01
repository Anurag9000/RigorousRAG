import json

import pytest

from tools.migration_promotion import PromotionReport
from tools.migration_promotion_runtime import (
    clear_migration_promotion_store_cache,
    get_migration_promotion_store,
)
from tools.migration_promotion_store import MigrationPromotionStore

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64
F = "f" * 64


def report(decision="eligible", reasons=(), evaluated_at=1.0, policy_digest=F):
    return PromotionReport(
        task_id=E,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=A,
        target_profile_fingerprint=B,
        validation_digest=C,
        benchmark_fingerprint=D,
        evidence_digest=F,
        policy_id="policy-v1",
        policy_digest=policy_digest,
        decision=decision,
        reason_codes=tuple(reasons),
        quality_deltas={
            "recall_at_k": 0.01,
            "ndcg_at_k": 0.01,
            "mrr": 0.01,
            "support_recall": 0.01,
            "citation_precision": 0.0,
            "abstention_accuracy": 0.01,
        },
        resource_ratios={
            "p95_latency_ms": 1.1,
            "peak_memory_bytes": 1.0,
            "index_bytes": 1.2,
            "estimated_cost_units": 1.0,
        },
        evaluated_at=evaluated_at,
    )


def test_append_only_write_current_pointer_and_timestamp_reuse(tmp_path):
    store = MigrationPromotionStore(tmp_path / "promotions")
    first = store.write(report(evaluated_at=1))
    same = store.write(report(evaluated_at=9))
    assert same == first
    assert store.read(E) == first
    task_dir = store.root / E
    assert (task_dir / f"{first.report_digest}.json").is_file()
    pointer = json.loads((task_dir / "current.json").read_text())
    assert pointer == {"report_digest": first.report_digest}


def test_multiple_reports_are_retained_and_history_is_newest_first(tmp_path):
    store = MigrationPromotionStore(tmp_path / "promotions")
    first = store.write(report(evaluated_at=1))
    blocked = store.write(
        report(
            decision="blocked",
            reasons=("recall_at_k_below_floor",),
            evaluated_at=2,
            policy_digest="1" * 64,
        )
    )
    assert store.read(E) == blocked
    assert store.read(E, report_digest=first.report_digest) == first
    assert store.history(E) == (blocked, first)
    assert store.history(E, limit=1) == (blocked,)


def test_report_and_pointer_tampering_are_detected(tmp_path):
    store = MigrationPromotionStore(tmp_path / "promotions")
    item = store.write(report())
    task_dir = store.root / E
    path = task_dir / f"{item.report_digest}.json"
    payload = json.loads(path.read_text())
    payload["doc_id"] = "other"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="digest"):
        store.read(E, report_digest=item.report_digest)

    path.write_text(json.dumps(item.__dict__))
    (task_dir / "current.json").write_text('{"report_digest":"x"}')
    with pytest.raises(ValueError):
        store.read(E)


def test_history_and_reports_contain_no_private_source_path(tmp_path):
    store = MigrationPromotionStore(tmp_path / "promotions")
    item = store.write(report())
    raw = (store.root / E / f"{item.report_digest}.json").read_text()
    assert "source_path" not in raw
    assert "/private/" not in raw


def test_remove_and_path_scoped_runtime_cache(tmp_path):
    store = MigrationPromotionStore(tmp_path / "promotions")
    store.write(report())
    assert store.remove_task(E) is True
    assert store.remove_task(E) is False
    clear_migration_promotion_store_cache()
    one = get_migration_promotion_store(tmp_path / "one")
    again = get_migration_promotion_store(tmp_path / "one")
    two = get_migration_promotion_store(tmp_path / "two")
    assert one is again
    assert one is not two


def test_symlink_and_root_replacement_fail_closed(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="redirects"):
        MigrationPromotionStore(link)

    root = tmp_path / "root"
    store = MigrationPromotionStore(root)
    root.rename(tmp_path / "old")
    root.mkdir()
    with pytest.raises(RuntimeError, match="identity changed"):
        store.write(report())
