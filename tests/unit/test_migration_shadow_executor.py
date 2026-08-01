from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.migration_shadow_executor import (
    build_and_validate_shadow,
    execute_shadow_task,
)
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
        lease_owner="worker",
        lease_expires_at=100.0,
        validation_digest=validation_digest,
    )


def generation(sequence=3, profile=SOURCE, content=CONTENT, state="active"):
    return SimpleNamespace(
        sequence=sequence,
        profile_fingerprint=profile,
        content_sha256=content,
        state=state,
    )


def build():
    return ShadowBuild(
        content_sha256=CONTENT,
        parser_fingerprint=PARSER,
        vector_rows=({"row_id": "one", "embedding": [0.1, 0.2]},),
        sparse_rows=({"field_id": "one", "text": "source"},),
    )


class Generations:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def current(self, *, owner_id, doc_id):
        index = min(self.calls, len(self.values) - 1)
        self.calls += 1
        return self.values[index]


class Journal:
    def __init__(self):
        self.validated = []
        self.failed = []

    def mark_validated(
        self,
        *,
        task_id,
        worker_id,
        validation_digest,
        now=None,
    ):
        self.validated.append((task_id, worker_id, validation_digest, now))
        return task(state="validated", validation_digest=validation_digest)

    def mark_failed(self, *, task_id, worker_id, failure_type, now=None):
        self.failed.append((task_id, worker_id, failure_type, now))
        return MigrationTask(
            task_id=task_id,
            owner_id="alice",
            doc_id="doc-1",
            source_sequence=3,
            source_profile_fingerprint=SOURCE,
            target_profile_name="bge-m3",
            target_profile_fingerprint=TARGET,
            state="failed",
            attempt=1,
            created_at=1.0,
            updated_at=3.0,
            failure_type=failure_type,
        )


def test_running_task_builds_validates_and_records_digest(tmp_path):
    store = MigrationShadowStore(tmp_path / "shadows")
    journal = Journal()
    generations = Generations([generation(), generation()])
    calls = []
    result = build_and_validate_shadow(
        task(),
        worker_id="worker",
        journal=journal,
        generations=generations,
        shadows=store,
        builder=lambda migration: calls.append(migration.task_id) or build(),
        now=5.0,
    )
    assert result.outcome == "validated"
    assert result.task_state == "validated"
    assert result.vector_count == 1
    assert result.sparse_count == 1
    assert len(result.validation_digest) == 64
    assert calls == [task().task_id]
    assert journal.validated[0][2] == result.validation_digest
    assert journal.failed == []


def test_source_generation_must_match_before_and_after_build(tmp_path):
    store = MigrationShadowStore(tmp_path / "shadows")
    with pytest.raises(RuntimeError, match="before shadow"):
        build_and_validate_shadow(
            task(),
            worker_id="worker",
            journal=Journal(),
            generations=Generations([generation(sequence=4)]),
            shadows=store,
            builder=lambda migration: build(),
        )

    with pytest.raises(RuntimeError, match="during shadow"):
        build_and_validate_shadow(
            task(),
            worker_id="worker",
            journal=Journal(),
            generations=Generations([generation(), generation(sequence=4)]),
            shadows=MigrationShadowStore(tmp_path / "second"),
            builder=lambda migration: build(),
        )


def test_content_hash_and_worker_lease_are_enforced(tmp_path):
    store = MigrationShadowStore(tmp_path / "shadows")
    with pytest.raises(RuntimeError, match="content hash"):
        build_and_validate_shadow(
            task(),
            worker_id="worker",
            journal=Journal(),
            generations=Generations([generation(), generation()]),
            shadows=store,
            builder=lambda migration: ShadowBuild(
                content_sha256="f" * 64,
                parser_fingerprint=PARSER,
                vector_rows=({"row_id": "one"},),
                sparse_rows=({"field_id": "one"},),
            ),
        )
    with pytest.raises(ValueError, match="does not own"):
        build_and_validate_shadow(
            task(),
            worker_id="other",
            journal=Journal(),
            generations=Generations([generation()]),
            shadows=store,
            builder=lambda migration: build(),
        )


def test_validated_task_reuses_exact_artifacts_without_builder(tmp_path):
    store = MigrationShadowStore(tmp_path / "shadows")
    manifest = store.write(task=task(), build=build(), now=5.0)
    validated = task(state="validated", validation_digest=manifest.validation_digest)
    result = build_and_validate_shadow(
        validated,
        worker_id="worker",
        journal=Journal(),
        generations=Generations([generation()]),
        shadows=store,
        builder=lambda migration: pytest.fail("builder must not run"),
    )
    assert result.outcome == "already_validated"
    assert result.validation_digest == manifest.validation_digest


def test_contained_execution_records_only_generic_failure(tmp_path):
    journal = Journal()
    result = execute_shadow_task(
        task(),
        worker_id="worker",
        journal=journal,
        generations=Generations([generation()]),
        shadows=MigrationShadowStore(tmp_path / "shadows"),
        builder=lambda migration: (_ for _ in ()).throw(
            RuntimeError("private retained source path")
        ),
    )
    assert result.outcome == "failed"
    assert result.failure_type == "RuntimeError"
    assert journal.failed[0][2] == "RuntimeError"
    assert "private retained source path" not in repr(result)
