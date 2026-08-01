from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.migration_shadow_store import MigrationShadowStore, ShadowBuild
from tools.migration_types import MigrationTask

SOURCE = "a" * 64
TARGET = "b" * 64
CONTENT = "c" * 64
PARSER = "d" * 64


def task(state="running", validation_digest=None):
    return MigrationTask(
        task_id="e" * 64,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=3,
        source_profile_fingerprint=SOURCE,
        target_profile_name="bge-m3",
        target_profile_fingerprint=TARGET,
        state=state,
        attempt=1,
        created_at=1.0,
        updated_at=2.0,
        lease_owner="worker" if state in {"running", "validated"} else None,
        lease_expires_at=100.0 if state in {"running", "validated"} else None,
        validation_digest=validation_digest,
    )


def build(vector_value=1.0):
    return ShadowBuild(
        content_sha256=CONTENT,
        parser_fingerprint=PARSER,
        vector_rows=(
            {
                "row_id": "chunk-1",
                "text": "bounded source text",
                "embedding": [vector_value, 0.5],
                "metadata": {"page_number": 1},
            },
        ),
        sparse_rows=(
            {
                "field_id": "field-1",
                "field_type": "body",
                "text": "bounded source text",
                "page_number": 1,
            },
        ),
    )


def test_manifest_last_shadow_write_and_validation(tmp_path):
    store = MigrationShadowStore(tmp_path / "shadows")
    manifest = store.write(task=task(), build=build(), now=5.0)
    directory = store.root / task().task_id
    assert directory.is_dir()
    assert (directory / "vectors.json").is_file()
    assert (directory / "sparse.json").is_file()
    assert (directory / "manifest.json").is_file()
    assert manifest.vector_count == 1
    assert manifest.sparse_count == 1
    assert len(manifest.validation_digest) == 64
    assert store.validate(task().task_id) == manifest


def test_identical_existing_shadow_is_reused_across_retry_times(tmp_path):
    store = MigrationShadowStore(tmp_path / "shadows")
    first = store.write(task=task(), build=build(), now=5.0)
    second = store.write(task=task(), build=build(), now=9.0)
    assert second == first
    assert second.validation_digest == first.validation_digest


def test_changed_existing_shadow_is_refused(tmp_path):
    store = MigrationShadowStore(tmp_path / "shadows")
    store.write(task=task(), build=build(), now=5.0)
    with pytest.raises(RuntimeError, match="different artifacts"):
        store.write(task=task(), build=build(0.1), now=5.0)


def test_vector_or_manifest_tampering_is_detected(tmp_path):
    store = MigrationShadowStore(tmp_path / "shadows")
    store.write(task=task(), build=build(), now=5.0)
    directory = store.root / task().task_id
    (directory / "vectors.json").write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="vector artifact digest"):
        store.validate(task().task_id)

    store.remove(task().task_id)
    store.write(task=task(), build=build(), now=5.0)
    manifest_path = directory / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["task_id"] = "f" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="task identity"):
        store.validate(task().task_id)


def test_nonfinite_deep_and_empty_artifacts_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        build(float("nan"))
    with pytest.raises(ValueError, match="at least one"):
        ShadowBuild(
            content_sha256=CONTENT,
            parser_fingerprint=PARSER,
            vector_rows=(),
            sparse_rows=({"field_id": "one"},),
        )


def test_symlink_root_and_replaced_root_identity_are_refused(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links unavailable")
    with pytest.raises(ValueError, match="redirects"):
        MigrationShadowStore(link)

    root = tmp_path / "root"
    store = MigrationShadowStore(root)
    root.rename(tmp_path / "old-root")
    root.mkdir()
    with pytest.raises(RuntimeError, match="identity changed"):
        store.write(task=task(), build=build(), now=5.0)
