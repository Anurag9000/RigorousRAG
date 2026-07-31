import json
import os
import sqlite3
from types import SimpleNamespace

import pytest

import tools.sparse_index as sparse_module
from tools.sparse_index import SparseField, SparseIndex


FINGERPRINT = "a" * 64


def field(identifier, field_type, text, position, **kwargs):
    return SparseField(identifier, field_type, text, position, **kwargs)


def test_transactional_replace_field_weighting_provenance_and_owner_isolation(tmp_path):
    index = SparseIndex(tmp_path / "sparse.sqlite3")
    assert index.replace_document(
        owner_id="alice",
        doc_id="doc-title",
        profile_fingerprint=FINGERPRINT,
        metadata={"kind": "paper"},
        fields=[
            field("title", "title", "rare target", 0, page_number=1, section="Title"),
            field("body", "body", "background", 1),
        ],
    ) == 1
    index.replace_document(
        owner_id="alice",
        doc_id="doc-body",
        fields=[field("body", "body", "rare target target", 0, page_number=7)],
    )
    index.replace_document(
        owner_id="bob",
        doc_id="secret",
        fields=[field("body", "body", "rare target target target", 0)],
    )

    hits = index.search("rare target", owner_id="alice", limit=10)
    assert [hit.doc_id for hit in hits] == ["doc-title", "doc-body"]
    assert all(hit.doc_id != "secret" for hit in hits)
    assert hits[0].profile_fingerprint == FINGERPRINT
    assert hits[0].metadata == {"kind": "paper"}
    assert hits[0].matches[0].field_type == "title"
    assert hits[0].matches[0].page_number == 1
    assert hits[0].matches[0].positions["target"] == (1,)
    assert index.search("target", owner_id="bob")[0].doc_id == "secret"


def test_exact_document_and_field_filters_and_custom_weights(tmp_path):
    index = SparseIndex(tmp_path / "sparse.sqlite3")
    index.replace_document(
        owner_id="alice",
        doc_id="d1",
        fields=[
            field("heading", "heading", "target heading", 0),
            field("custom", "custom:methods", "target method", 1),
        ],
    )
    index.replace_document(
        owner_id="alice",
        doc_id="d2",
        fields=[field("body", "body", "target body", 0)],
    )
    assert [hit.doc_id for hit in index.search("target", owner_id="alice", doc_id="d2")] == ["d2"]
    assert [hit.doc_id for hit in index.search("target", owner_id="alice", field_types=["heading"])] == ["d1"]
    hit = index.search(
        "target",
        owner_id="alice",
        field_types=["custom:methods"],
        field_weights={"custom:methods": 4.0},
    )[0]
    assert hit.doc_id == "d1"
    assert hit.matches[0].field_type == "custom:methods"


def test_generation_snapshot_restore_and_absent_restore(tmp_path):
    index = SparseIndex(tmp_path / "sparse.sqlite3")
    generation_one = index.replace_document(
        owner_id="alice",
        doc_id="d1",
        fields=[field("body", "body", "first generation", 0)],
        metadata={"version": 1},
    )
    snapshot = index.snapshot_document(owner_id="alice", doc_id="d1")
    assert snapshot is not None and snapshot.generation == generation_one == 1
    generation_two = index.replace_document(
        owner_id="alice",
        doc_id="d1",
        fields=[field("body", "body", "second replacement", 0)],
        metadata={"version": 2},
        expected_generation=1,
    )
    assert generation_two == 2
    assert index.search("second", owner_id="alice")[0].generation == 2
    index.restore_document(owner_id="alice", doc_id="d1", snapshot=snapshot)
    restored = index.snapshot_document(owner_id="alice", doc_id="d1")
    assert restored is not None and restored.generation == 1
    assert restored.metadata == {"version": 1}
    assert index.search("first", owner_id="alice")[0].doc_id == "d1"
    assert index.search("second", owner_id="alice") == []
    index.restore_document(owner_id="alice", doc_id="d1", snapshot=None)
    assert not index.document_exists(owner_id="alice", doc_id="d1")


def test_replacement_rolls_back_completely_after_posting_failure(tmp_path, monkeypatch):
    index = SparseIndex(tmp_path / "sparse.sqlite3")
    index.replace_document(
        owner_id="alice",
        doc_id="d1",
        fields=[field("body", "body", "stable evidence", 0)],
    )
    before = index.snapshot_document(owner_id="alice", doc_id="d1")
    original = index._insert_posting
    calls = {"count": 0}

    def fail_second(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise sqlite3.OperationalError("forced posting failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(index, "_insert_posting", fail_second)
    with pytest.raises(sqlite3.OperationalError, match="forced"):
        index.replace_document(
            owner_id="alice",
            doc_id="d1",
            fields=[field("body", "body", "new terms here", 0)],
        )
    assert index.snapshot_document(owner_id="alice", doc_id="d1") == before
    assert index.search("stable", owner_id="alice")[0].doc_id == "d1"
    assert index.search("new", owner_id="alice") == []


def test_expected_generation_prevents_lost_updates(tmp_path):
    index = SparseIndex(tmp_path / "sparse.sqlite3")
    index.replace_document(
        owner_id="alice",
        doc_id="d1",
        fields=[field("body", "body", "evidence", 0)],
    )
    with pytest.raises(RuntimeError, match="expected 0, found 1"):
        index.replace_document(
            owner_id="alice",
            doc_id="d1",
            fields=[field("body", "body", "replacement", 0)],
            expected_generation=0,
        )
    assert index.snapshot_document(owner_id="alice", doc_id="d1").generation == 1


def test_delete_list_and_duplicate_fields(tmp_path):
    index = SparseIndex(tmp_path / "sparse.sqlite3")
    with pytest.raises(ValueError, match="Duplicate"):
        index.replace_document(
            owner_id="alice",
            doc_id="d1",
            fields=[
                field("same", "body", "one", 0),
                field("same", "heading", "two", 1),
            ],
        )
    index.replace_document(
        owner_id="alice",
        doc_id="d1",
        fields=[field("body", "body", "one", 0)],
    )
    index.replace_document(
        owner_id="alice",
        doc_id="d2",
        fields=[field("body", "body", "two", 0)],
    )
    assert index.list_document_ids(owner_id="alice") == ("d1", "d2")
    assert index.delete_document(owner_id="alice", doc_id="d1") is True
    assert index.delete_document(owner_id="alice", doc_id="d1") is False
    assert index.list_document_ids(owner_id="alice") == ("d2",)


def test_nonfinite_metadata_exact_numbers_and_invalid_identifiers_fail_closed(tmp_path):
    index = SparseIndex(tmp_path / "sparse.sqlite3")
    with pytest.raises(ValueError, match="unsupported"):
        index.replace_document(
            owner_id="alice",
            doc_id="d1",
            fields=[field("body", "body", "text", 0)],
            metadata={"bad": float("nan")},
        )
    for invalid in (True, 1.5, "1"):
        with pytest.raises(ValueError):
            SparseField("body", "body", "text", invalid)
    with pytest.raises(ValueError):
        index.replace_document(
            owner_id=" alice ",
            doc_id="bad id",
            fields=[field("body", "body", "text", 0)],
        )


def test_corrupt_snapshot_refuses_publication_and_corrupt_search_hit_is_skipped(tmp_path):
    index = SparseIndex(tmp_path / "sparse.sqlite3")
    index.replace_document(
        owner_id="alice",
        doc_id="good",
        fields=[field("body", "body", "target good", 0)],
    )
    index.replace_document(
        owner_id="alice",
        doc_id="bad",
        fields=[field("body", "body", "target bad", 0)],
    )
    with sqlite3.connect(index.path) as connection:
        connection.execute(
            "UPDATE sparse_documents SET metadata_json='not-json' WHERE owner_id='alice' AND doc_id='bad'"
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        index.snapshot_document(owner_id="alice", doc_id="bad")
    assert [hit.doc_id for hit in index.search("target", owner_id="alice")] == ["good"]


def test_parent_and_database_replacement_fail_closed(tmp_path):
    parent = tmp_path / "index-root"
    index = SparseIndex(parent / "sparse.sqlite3")
    parent_backup = tmp_path / "index-root-old"
    parent.rename(parent_backup)
    parent.mkdir()
    (parent / "sparse.sqlite3").touch()
    with pytest.raises(RuntimeError, match="parent directory was replaced"):
        index.list_document_ids(owner_id="alice")

    db_parent = tmp_path / "db-root"
    second = SparseIndex(db_parent / "sparse.sqlite3")
    second.path.rename(db_parent / "old.sqlite3")
    second.path.touch()
    with pytest.raises(RuntimeError, match="database file was replaced"):
        second.list_document_ids(owner_id="alice")


def test_symlink_and_windows_reparse_detection(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Symbolic links unavailable")
    with pytest.raises(ValueError, match="links|reparse"):
        SparseIndex(link / "sparse.sqlite3")
    fake = SimpleNamespace(st_mode=stat_mode_regular(), st_file_attributes=0x400)
    assert sparse_module._is_redirecting(fake) is True


def stat_mode_regular():
    return 0o100644


def test_ping_and_restore_identity_checks(tmp_path):
    index = SparseIndex(tmp_path / "sparse.sqlite3")
    assert index.ping() is True
    index.replace_document(
        owner_id="alice",
        doc_id="d1",
        fields=[field("body", "body", "text", 0)],
    )
    snapshot = index.snapshot_document(owner_id="alice", doc_id="d1")
    with pytest.raises(ValueError, match="identity"):
        index.restore_document(owner_id="bob", doc_id="d1", snapshot=snapshot)


def test_hostile_or_oversized_metadata_mapping_is_bounded_without_materialization(tmp_path):
    class HugeMapping(dict):
        def items(self):
            for index in range(10_000_000):
                yield str(index), index

        def __len__(self):
            raise AssertionError("len() must not be called")

    index = SparseIndex(tmp_path / "sparse.sqlite3")
    with pytest.raises(ValueError, match="too many fields"):
        index.replace_document(
            owner_id="alice",
            doc_id="d1",
            fields=[field("body", "body", "text", 0)],
            metadata=HugeMapping(),
        )


def test_corrupt_posting_positions_skip_the_entire_document(tmp_path):
    index = SparseIndex(tmp_path / "sparse.sqlite3")
    index.replace_document(
        owner_id="alice",
        doc_id="good",
        fields=[field("body", "body", "target good", 0)],
    )
    index.replace_document(
        owner_id="alice",
        doc_id="bad",
        fields=[field("body", "body", "target bad", 0)],
    )
    with sqlite3.connect(index.path) as connection:
        connection.execute(
            "UPDATE sparse_postings SET positions_json='[999]', frequency=2 "
            "WHERE owner_id='alice' AND doc_id='bad' AND term='target'"
        )
    assert [hit.doc_id for hit in index.search("target", owner_id="alice")] == ["good"]
