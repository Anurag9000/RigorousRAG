import json

import pytest

from tools.migration_cutover_preflight import CutoverPreflight
from tools.migration_cutover_preflight_runtime import (
    clear_migration_cutover_preflight_store_cache,
    get_migration_cutover_preflight_store,
)
from tools.migration_cutover_preflight_store import MigrationCutoverPreflightStore

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64
F = "f" * 64


def value(created_at=1.0, promotion=A):
    return CutoverPreflight(
        task_id=E,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=A,
        target_profile_fingerprint=B,
        source_content_sha256=C,
        validation_digest=D,
        promotion_report_digest=promotion,
        benchmark_fingerprint=F,
        vector_snapshot_digest="1" * 64,
        sparse_snapshot_digest="2" * 64,
        rollback_identity_digest="3" * 64,
        target_artifact_digest="4" * 64,
        source_vector_rows=2,
        source_sparse_generation=7,
        source_sparse_fields=2,
        target_vector_rows=3,
        target_sparse_rows=3,
        created_at=created_at,
    )


def test_append_only_current_pointer_and_timestamp_reuse(tmp_path):
    store = MigrationCutoverPreflightStore(tmp_path / "preflights")
    first = store.write(value(created_at=1))
    same = store.write(value(created_at=9))
    assert same == first
    assert store.read(E) == first
    pointer = json.loads((store.root / E / "current.json").read_text())
    assert pointer == {"preflight_digest": first.preflight_digest}


def test_history_retains_changed_promotion_identity(tmp_path):
    store = MigrationCutoverPreflightStore(tmp_path / "preflights")
    first = store.write(value(created_at=1))
    second = store.write(value(created_at=2, promotion="9" * 64))
    assert store.read(E) == second
    assert store.read(E, preflight_digest=first.preflight_digest) == first
    assert store.history(E) == (second, first)


def test_tampering_is_detected_and_records_are_path_free(tmp_path):
    store = MigrationCutoverPreflightStore(tmp_path / "preflights")
    item = store.write(value())
    path = store.root / E / f"{item.preflight_digest}.json"
    raw = path.read_text()
    assert "source_path" not in raw and "document" not in raw
    payload = json.loads(raw)
    payload["doc_id"] = "other"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="digest"):
        store.read(E, preflight_digest=item.preflight_digest)


def test_remove_and_path_scoped_runtime_cache(tmp_path):
    store = MigrationCutoverPreflightStore(tmp_path / "preflights")
    store.write(value())
    assert store.remove_task(E) is True
    assert store.remove_task(E) is False
    clear_migration_cutover_preflight_store_cache()
    one = get_migration_cutover_preflight_store(tmp_path / "one")
    again = get_migration_cutover_preflight_store(tmp_path / "one")
    two = get_migration_cutover_preflight_store(tmp_path / "two")
    assert one is again and one is not two


def test_symlink_and_replaced_root_fail_closed(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="redirects"):
        MigrationCutoverPreflightStore(link)
    root = tmp_path / "root"
    store = MigrationCutoverPreflightStore(root)
    root.rename(tmp_path / "old")
    root.mkdir()
    with pytest.raises(RuntimeError, match="identity changed"):
        store.write(value())
